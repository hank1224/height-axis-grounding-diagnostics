#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import queue
import re
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from height_axis_grounding_diagnostics.common.io_utils import (  # noqa: E402
    JsonParseError,
    detect_mime_type,
    load_env_file,
    parse_json_text,
    read_json,
    read_text,
    write_csv,
    write_json,
    write_text,
)
from height_axis_grounding_diagnostics.common.providers import (  # noqa: E402
    ProviderClient,
    ProviderError,
    add_provider_arguments,
    build_provider_client,
    get_installed_sdk_versions,
    get_provider_doc_sources,
    resolve_provider_runtime,
)
from height_axis_grounding_diagnostics.common.retry import (  # noqa: E402
    ApiErrorGuardError,
    RetryPolicy,
    add_retry_arguments,
    assert_no_api_errors,
    build_api_error_rows,
    build_retry_policy_from_args,
    calculate_retry_delay,
    classify_api_error,
    normalize_status_code,
)
from height_axis_grounding_diagnostics.tasks.pcb_face_axis_mapping.schema import (  # noqa: E402
    PcbFaceAxisSchemaError,
    compare_pcb_face_axis_outputs,
    load_answer_map,
    validate_ground_truth_payload,
)


TASK_NAME = "pcb_face_axis_mapping"
DEFAULT_ENV_PATH = ROOT / ".env"
DEFAULT_CASES_PATH = ROOT / "data" / "tasks" / TASK_NAME / "cases.json"
DEFAULT_GROUND_TRUTH_PATH = ROOT / "data" / "tasks" / TASK_NAME / "ground_truth.json"
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / TASK_NAME


class BenchmarkError(Exception):
    pass


class DryRunClient:
    def __init__(self, model: str) -> None:
        self.model = model


def sanitize_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")


def path_for_record(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def filter_cases(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    case_ids = set(args.case_id or [])
    image_ids = set(args.image_id or [])
    package_slugs = set(args.package_slug or [])
    variant_slugs = set(args.variant_slug or [])

    for case in cases:
        if case_ids and case["case_id"] not in case_ids:
            continue
        if image_ids and case["image_id"] not in image_ids:
            continue
        if package_slugs and case["package_slug"] not in package_slugs:
            continue
        if variant_slugs and case["variant_slug"] not in variant_slugs:
            continue
        selected.append(case)

    if args.max_cases is not None:
        selected = selected[: args.max_cases]
    if not selected:
        raise BenchmarkError("No cases matched the requested filters")
    return selected


def write_response_artifacts(
    *,
    run_dir: Path,
    provider: str,
    case_id: str,
    repeat_index: int,
    response_text: Any,
    raw_response_text: Any,
    response_json: Any,
) -> dict[str, str]:
    attempt_dir = run_dir / "attempts" / provider / case_id
    stem = f"run-{repeat_index:03d}"
    artifact_paths: dict[str, str] = {}

    if isinstance(response_text, str):
        response_text_path = attempt_dir / f"{stem}.response.txt"
        write_text(response_text_path, response_text)
        artifact_paths["response_text"] = path_for_record(response_text_path)

    if isinstance(raw_response_text, str):
        raw_response_path = attempt_dir / f"{stem}.raw.txt"
        write_text(raw_response_path, raw_response_text)
        artifact_paths["raw_response_text"] = path_for_record(raw_response_path)

    if response_json is not None:
        response_json_path = attempt_dir / f"{stem}.sdk.json"
        write_json(response_json_path, response_json)
        artifact_paths["response_json"] = path_for_record(response_json_path)

    return artifact_paths


SLOT_SYNTHETIC_BBOXES = {
    "upper_left": [0, 0, 500, 500],
    "upper_right": [0, 500, 500, 1000],
    "lower_left": [500, 0, 1000, 500],
    "lower_right": [500, 500, 1000, 1000],
}


def make_dry_run_prediction(expected_output: dict[str, Any]) -> dict[str, Any]:
    prediction = json.loads(json.dumps(expected_output))
    for view in prediction.get("views", []):
        if isinstance(view, dict) and "bounding_box_2d" not in view:
            slot = view.get("slot")
            if slot in SLOT_SYNTHETIC_BBOXES:
                view["bounding_box_2d"] = SLOT_SYNTHETIC_BBOXES[slot]
    return prediction


def execute_provider_attempt(
    *,
    provider: str,
    client: ProviderClient,
    case: dict[str, Any],
    prompt_text: str,
    image_path: Path,
) -> tuple[dict[str, Any], Any | None, str | None]:
    try:
        provider_result = client.run(prompt_text=prompt_text, image_path=image_path)
        request_summary = provider_result.get("request_summary")
        if isinstance(request_summary, dict):
            request_summary["task_name"] = TASK_NAME
            request_summary["case_id"] = case["case_id"]
            request_summary["image_id"] = case["image_id"]
        response_text = provider_result["response_text"] or ""
        raw_prediction, parse_error = (
            parse_json_text(response_text) if response_text else (None, "Empty response text")
        )
        return provider_result, raw_prediction, parse_error
    except Exception as exc:  # pragma: no cover - defensive against SDK/network/runtime failures
        error_text = f"{exc.__class__.__name__}: {exc}"
        status_code = normalize_status_code(getattr(exc, "status_code", None), error_text)
        provider_result = {
            "status_code": status_code,
            "raw_response_text": str(exc),
            "response_json": None,
            "response_text": None,
            "request_summary": {
                "transport": f"{provider} API",
                "endpoint": "request_failed_before_response",
                "model": client.model,
                "image_path": image_path.relative_to(ROOT).as_posix(),
                "mime_type": detect_mime_type(image_path),
                "task_name": TASK_NAME,
                "case_id": case["case_id"],
                "image_id": case["image_id"],
                "temperature": client.temperature,
                "structured_output": False,
            },
        }
        return provider_result, None, error_text


def process_attempt(
    *,
    run_dir: Path,
    provider: str,
    client: ProviderClient | DryRunClient,
    case: dict[str, Any],
    expected_group: dict[str, Any],
    prompt_text: str,
    repeat_index: int,
    temperature: float,
    dry_run: bool,
    retry_policy: RetryPolicy,
) -> dict[str, Any]:
    image_path = ROOT / case["image_path"]
    attempt_id = f"{provider}__{case['case_id']}__run-{repeat_index:03d}"
    started = time.perf_counter()
    retry_attempts: list[dict[str, Any]] = []

    if dry_run:
        raw_prediction = make_dry_run_prediction(expected_group["ground_truth"])
        response_text = json.dumps(raw_prediction, ensure_ascii=False)
        provider_result = {
            "status_code": 200,
            "raw_response_text": response_text,
            "response_json": {"dry_run": True},
            "response_text": response_text,
            "request_summary": {
                "endpoint": "dry-run",
                "model": client.model,
                "image_path": image_path.relative_to(ROOT).as_posix(),
                "mime_type": detect_mime_type(image_path),
                "task_name": TASK_NAME,
                "case_id": case["case_id"],
                "image_id": case["image_id"],
                "temperature": temperature,
                "structured_output": False,
            },
        }
        parse_error = None
        retry_attempts.append(
            {
                "attempt_number": 1,
                "status_code": 200,
                "outcome": "success",
                "error_type": None,
                "error_message": None,
                "retryable": False,
                "retry_reason": None,
                "sleep_seconds": 0.0,
            }
        )
    else:
        provider_result = {}
        raw_prediction = None
        parse_error = None
        if not isinstance(client, ProviderClient):
            raise BenchmarkError("DryRunClient cannot be used for real provider attempts")
        for attempt_number in range(1, retry_policy.max_attempts + 1):
            provider_result, raw_prediction, parse_error = execute_provider_attempt(
                provider=provider,
                client=client,
                case=case,
                prompt_text=prompt_text,
                image_path=image_path,
            )
            status_code = provider_result.get("status_code")
            error_type = None
            retryable = False
            if parse_error:
                request_summary = provider_result.get("request_summary") or {}
                if request_summary.get("endpoint") == "request_failed_before_response":
                    error_type = "api_error"
                    retry_decision = classify_api_error(
                        status_code=status_code if isinstance(status_code, int) else None,
                        error_text=parse_error,
                        policy=retry_policy,
                    )
                    retryable = (
                        retry_policy.enabled
                        and attempt_number < retry_policy.max_attempts
                        and retry_decision.retryable
                    )
                    retry_reason = retry_decision.reason
                else:
                    error_type = "parse_error"
                    retry_reason = None
            else:
                retry_reason = None

            sleep_seconds = calculate_retry_delay(attempt_number, retry_policy) if parse_error and retryable else 0.0
            retry_attempts.append(
                {
                    "attempt_number": attempt_number,
                    "status_code": status_code,
                    "outcome": "success" if not parse_error else "failed",
                    "error_type": error_type,
                    "error_message": parse_error,
                    "retryable": retryable,
                    "retry_reason": retry_reason,
                    "sleep_seconds": sleep_seconds,
                }
            )
            if not parse_error or not retryable:
                break
            time.sleep(sleep_seconds)

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    expected_output = expected_group["ground_truth"]
    comparison = (
        compare_pcb_face_axis_outputs({"_parse_error": parse_error}, expected_output)
        if parse_error
        else compare_pcb_face_axis_outputs(raw_prediction, expected_output)
    )

    retry_count = max(0, len(retry_attempts) - 1)
    retried = retry_count > 0
    if parse_error:
        final_outcome = "failed_after_retries" if retried else "failed_without_retry"
    else:
        final_outcome = "success_after_retry" if retried else "success_first_try"

    attempt_record = {
        "attempt_id": attempt_id,
        "task_name": TASK_NAME,
        "provider": provider,
        "model": client.model,
        "case_id": case["case_id"],
        "image_id": case["image_id"],
        "answer_key": case["answer_key"],
        "package_name": case["package_name"],
        "package_slug": case["package_slug"],
        "shape_class": case.get("shape_class", ""),
        "variant_name": case["variant_name"],
        "variant_slug": case["variant_slug"],
        "prompt_path": case["prompt_path"],
        "rendered_prompt": prompt_text,
        "repeat_index": repeat_index,
        "retried": retried,
        "retry_count": retry_count,
        "retry_summary": {
            "retry_policy": retry_policy.to_dict(),
            "attempt_count": len(retry_attempts),
            "final_outcome": final_outcome,
            "retryable_error_count": sum(1 for item in retry_attempts if item.get("retryable")),
        },
        "retry_attempts": retry_attempts,
        "latency_ms": latency_ms,
        "status_code": provider_result["status_code"],
        "response_text": provider_result["response_text"],
        "raw_prediction": raw_prediction,
        "parse_error": parse_error,
        "schema_valid": comparison["schema_valid"],
        "answer_schema_valid": comparison["answer_schema_valid"],
        "validation_errors": comparison["validation_errors"],
        "layout_exact_match": comparison["layout_exact_match"],
        "layout_slot_correct_count": comparison["layout_slot_correct_count"],
        "layout_slot_total_count": comparison["layout_slot_total_count"],
        "layout_slot_accuracy": comparison["layout_slot_accuracy"],
        "occupied_slot_precision": comparison["occupied_slot_precision"],
        "occupied_slot_recall": comparison["occupied_slot_recall"],
        "occupied_slot_f1": comparison["occupied_slot_f1"],
        "occupied_slot_matched_count": comparison["occupied_slot_matched_count"],
        "expected_occupied_slot_count": comparison["expected_occupied_slot_count"],
        "predicted_occupied_slot_count": comparison["predicted_occupied_slot_count"],
        "bbox_output_valid": comparison["bbox_output_valid"],
        "bbox_validation_errors": comparison["bbox_validation_errors"],
        "axis_correct_count": comparison["axis_correct_count"],
        "axis_total_count": comparison["axis_total_count"],
        "axis_accuracy": comparison["axis_accuracy"],
        "axis_precision": comparison["axis_precision"],
        "axis_recall": comparison["axis_recall"],
        "axis_f1": comparison["axis_f1"],
        "axis_exact_match": comparison["axis_exact_match"],
        "pcb_mounting_face_axis_correct_count": comparison["pcb_mounting_face_axis_correct_count"],
        "exact_match": comparison["exact_match"],
        "normalized_output": comparison["normalized_output"],
        "expected_output": expected_output,
        "request_summary": provider_result["request_summary"],
        "raw_response_text": provider_result["raw_response_text"],
        "response_json": provider_result["response_json"],
    }
    attempt_record["artifact_paths"] = write_response_artifacts(
        run_dir=run_dir,
        provider=provider,
        case_id=case["case_id"],
        repeat_index=repeat_index,
        response_text=provider_result["response_text"],
        raw_response_text=provider_result["raw_response_text"],
        response_json=provider_result["response_json"],
    )
    write_json(
        run_dir / "attempts" / provider / case["case_id"] / f"run-{repeat_index:03d}.json",
        attempt_record,
    )
    return attempt_record


def summarize_attempts(attempts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_provider_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_provider: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        by_provider_case[(attempt["provider"], attempt["case_id"])].append(attempt)
        by_provider[attempt["provider"]].append(attempt)

    metric_names = [
        "schema_valid",
        "answer_schema_valid",
        "layout_exact_match",
        "layout_slot_accuracy",
        "occupied_slot_f1",
        "bbox_output_valid",
        "axis_accuracy",
        "axis_precision",
        "axis_recall",
        "axis_f1",
        "axis_exact_match",
        "exact_match",
    ]

    case_summary_rows: list[dict[str, Any]] = []
    for (provider, case_id), rows in sorted(by_provider_case.items()):
        row = {
            "provider": provider,
            "case_id": case_id,
            "image_id": rows[0]["image_id"],
            "package_slug": rows[0]["package_slug"],
            "shape_class": rows[0].get("shape_class", ""),
            "variant_slug": rows[0]["variant_slug"],
            "repeats": len(rows),
            "axis_correct_count": sum(item["axis_correct_count"] for item in rows),
            "axis_total_count": sum(item["axis_total_count"] for item in rows),
            "pcb_mounting_face_axis_correct_count": sum(
                item["pcb_mounting_face_axis_correct_count"] for item in rows
            ),
        }
        for metric in metric_names:
            row[f"mean_{metric}"] = round(sum(float(item[metric]) for item in rows) / len(rows), 4)
        case_summary_rows.append(row)

    provider_summary_rows: list[dict[str, Any]] = []
    for provider, rows in sorted(by_provider.items()):
        row = {
            "provider": provider,
            "model": rows[0]["model"],
            "attempt_count": len(rows),
            "mean_latency_ms": round(sum(item["latency_ms"] for item in rows) / len(rows), 2),
            "axis_correct_count": sum(item["axis_correct_count"] for item in rows),
            "axis_total_count": sum(item["axis_total_count"] for item in rows),
            "pcb_mounting_face_axis_correct_count": sum(
                item["pcb_mounting_face_axis_correct_count"] for item in rows
            ),
        }
        for metric in metric_names:
            row[f"mean_{metric}"] = round(sum(float(item[metric]) for item in rows) / len(rows), 4)
        provider_summary_rows.append(row)
    return case_summary_rows, provider_summary_rows


def sort_attempts(attempts: list[dict[str, Any]], providers: list[str]) -> list[dict[str, Any]]:
    provider_order = {provider: index for index, provider in enumerate(providers)}
    return sorted(
        attempts,
        key=lambda item: (
            provider_order.get(item["provider"], len(provider_order)),
            item["case_id"],
            item["repeat_index"],
        ),
    )


def run_benchmark(args: argparse.Namespace) -> Path:
    load_env_file(args.env_file)
    retry_policy = build_retry_policy_from_args(args)
    cases_payload = read_json(args.cases)
    ground_truth_payload = read_json(args.ground_truth)
    all_cases = cases_payload.get("cases")
    if not isinstance(all_cases, list):
        raise BenchmarkError("Cases file must contain a `cases` array")
    selected_cases = filter_cases(all_cases, args)
    selected_answer_keys = {case["answer_key"] for case in selected_cases}
    selected_cases_payload = {
        **cases_payload,
        "cases": selected_cases,
        "case_count": len(selected_cases),
    }
    selected_ground_truth_payload = {
        **ground_truth_payload,
        "answer_groups": [
            group
            for group in ground_truth_payload.get("answer_groups", [])
            if isinstance(group, dict) and group.get("answer_key") in selected_answer_keys
        ],
    }
    gt_errors = validate_ground_truth_payload(
        cases_payload=selected_cases_payload,
        ground_truth_payload=selected_ground_truth_payload,
        require_complete=True,
    )
    if gt_errors:
        raise BenchmarkError("Ground truth is not ready for benchmarking:\n- " + "\n- ".join(gt_errors))

    answer_map = load_answer_map(selected_ground_truth_payload)
    missing_ground_truth = sorted({case["answer_key"] for case in selected_cases} - set(answer_map))
    if missing_ground_truth:
        raise BenchmarkError(f"Missing ground truth for answer keys: {missing_ground_truth}")

    provider_runtime = resolve_provider_runtime(
        args,
        dry_run=args.dry_run,
        env_file=args.env_file,
    )
    models = provider_runtime.models
    batch_sizes = provider_runtime.batch_sizes

    prompt_text = read_text(args.prompt)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{TASK_NAME}-{timestamp}"
    run_dir = args.output_root / sanitize_slug(run_name)
    if run_dir.exists():
        raise BenchmarkError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    run_config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_name": TASK_NAME,
        "transport": "provider APIs",
        "sdk_versions": get_installed_sdk_versions(),
        "retry_policy": retry_policy.to_dict(),
        "providers": args.providers,
        "models": {provider: models[provider] for provider in args.providers},
        "batch_sizes": {provider: batch_sizes[provider] for provider in args.providers},
        "base_urls": {
            provider: provider_runtime.base_urls[provider]
            for provider in args.providers
            if provider_runtime.base_urls.get(provider)
        },
        "repeats": args.repeats,
        "temperature": args.temperature,
        "timeout_seconds": args.timeout_seconds,
        "delay_seconds": args.delay_seconds,
        "selected_case_count": len(selected_cases),
        "selected_case_ids": [case["case_id"] for case in selected_cases],
        "cases_path": path_for_record(args.cases),
        "ground_truth_path": path_for_record(args.ground_truth),
        "prompt_path": path_for_record(args.prompt),
        "api_doc_sources": get_provider_doc_sources(args.providers),
        "dry_run": args.dry_run,
    }
    write_json(run_dir / "config.json", run_config)

    clients: dict[str, ProviderClient | DryRunClient] = {}
    for provider in args.providers:
        if args.dry_run:
            clients[provider] = DryRunClient(models[provider])
            continue
        clients[provider] = build_provider_client(
            provider,
            model=models[provider],
            api_key=provider_runtime.api_keys[provider],
            timeout_seconds=args.timeout_seconds,
            temperature=args.temperature,
            base_url=provider_runtime.base_urls.get(provider),
        )

    attempts: list[dict[str, Any]] = []
    attempts_lock = threading.Lock()
    worker_errors: list[str] = []
    work_queues: dict[str, queue.Queue[tuple[dict[str, Any], int] | None]] = {}
    worker_threads: list[threading.Thread] = []

    def provider_worker(provider: str, worker_index: int) -> None:
        client = clients[provider]
        task_queue = work_queues[provider]
        while True:
            item = task_queue.get()
            try:
                if item is None:
                    break
                case, repeat_index = item
                expected_group = answer_map[case["answer_key"]]
                attempt_record = process_attempt(
                    run_dir=run_dir,
                    provider=provider,
                    client=client,
                    case=case,
                    expected_group=expected_group,
                    prompt_text=prompt_text,
                    repeat_index=repeat_index,
                    temperature=args.temperature,
                    dry_run=args.dry_run,
                    retry_policy=retry_policy,
                )
                with attempts_lock:
                    attempts.append(attempt_record)
                print(
                    f"[{provider}] {case['case_id']} run {repeat_index}/{args.repeats} "
                    f"schema_valid={attempt_record['schema_valid']} "
                    f"axis_exact={attempt_record['axis_exact_match']} "
                    f"worker={worker_index}",
                    flush=True,
                )
                if args.delay_seconds > 0:
                    time.sleep(args.delay_seconds)
            except Exception as exc:  # pragma: no cover - defensive against worker runtime failures
                with attempts_lock:
                    worker_errors.append(f"{provider} worker {worker_index}: {exc}")
            finally:
                task_queue.task_done()

    for provider in args.providers:
        provider_queue: queue.Queue[tuple[dict[str, Any], int] | None] = queue.Queue()
        work_queues[provider] = provider_queue
        for case in selected_cases:
            for repeat_index in range(1, args.repeats + 1):
                provider_queue.put((case, repeat_index))
        for _ in range(batch_sizes[provider]):
            provider_queue.put(None)

        for worker_index in range(1, batch_sizes[provider] + 1):
            thread = threading.Thread(
                target=provider_worker,
                args=(provider, worker_index),
                name=f"{provider}-worker-{worker_index}",
            )
            thread.start()
            worker_threads.append(thread)

    for provider in args.providers:
        work_queues[provider].join()
    for thread in worker_threads:
        thread.join()
    if worker_errors:
        raise BenchmarkError(f"Parallel worker failures: {worker_errors}")

    attempts = sort_attempts(attempts, args.providers)
    attempts_csv_rows = [
        {
            "attempt_id": item["attempt_id"],
            "provider": item["provider"],
            "model": item["model"],
            "case_id": item["case_id"],
            "image_id": item["image_id"],
            "package_slug": item["package_slug"],
            "shape_class": item.get("shape_class", ""),
            "variant_slug": item["variant_slug"],
            "repeat_index": item["repeat_index"],
            "retried": item["retried"],
            "retry_count": item["retry_count"],
            "retry_final_outcome": item["retry_summary"]["final_outcome"],
            "latency_ms": item["latency_ms"],
            "status_code": item["status_code"],
            "schema_valid": item["schema_valid"],
            "answer_schema_valid": item["answer_schema_valid"],
            "layout_exact_match": item["layout_exact_match"],
            "occupied_slot_f1": item["occupied_slot_f1"],
            "bbox_output_valid": item["bbox_output_valid"],
            "axis_exact_match": item["axis_exact_match"],
            "pcb_mounting_face_axis_correct_count": item["pcb_mounting_face_axis_correct_count"],
            "axis_correct_count": item["axis_correct_count"],
            "axis_total_count": item["axis_total_count"],
            "axis_accuracy": item["axis_accuracy"],
            "axis_precision": item["axis_precision"],
            "axis_recall": item["axis_recall"],
            "axis_f1": item["axis_f1"],
            "exact_match": item["exact_match"],
            "parse_error": item["parse_error"] or "",
            "validation_errors": "; ".join(item["validation_errors"]),
            "bbox_validation_errors": "; ".join(item["bbox_validation_errors"]),
        }
        for item in attempts
    ]
    case_summary_rows, provider_summary_rows = summarize_attempts(attempts)
    api_error_rows = build_api_error_rows(attempts)
    write_json(
        run_dir / "summary.json",
        {
            "config": run_config,
            "api_error_count": len(api_error_rows),
            "provider_summary": provider_summary_rows,
            "case_summary": case_summary_rows,
        },
    )
    write_csv(run_dir / "attempts.csv", attempts_csv_rows)
    write_csv(run_dir / "case_summary.csv", case_summary_rows)
    write_csv(run_dir / "provider_summary.csv", provider_summary_rows)
    write_csv(run_dir / "api_errors.csv", api_error_rows)
    assert_no_api_errors(api_error_rows, allow_api_errors=args.allow_api_errors, run_dir=run_dir)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PCB face axis mapping benchmarks."
    )
    add_provider_arguments(parser)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--image-id", action="append")
    parser.add_argument("--package-slug", action="append")
    parser.add_argument("--variant-slug", action="append")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--run-name")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH_PATH)
    parser.add_argument("--prompt", type=Path, default=ROOT / "data" / "tasks" / TASK_NAME / "prompt.md")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    add_retry_arguments(parser)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    try:
        run_dir = run_benchmark(args)
    except (BenchmarkError, JsonParseError, PcbFaceAxisSchemaError, ProviderError, ApiErrorGuardError, ValueError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    print(f"Benchmark outputs written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
