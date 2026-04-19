#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import queue
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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

DEFAULT_ENV_PATH = ROOT / ".env"
DEFAULT_CASES_PATH = ROOT / "data" / "tasks" / "package_target_extraction" / "cases.json"
DEFAULT_GROUND_TRUTH_PATH = ROOT / "data" / "tasks" / "package_target_extraction" / "ground_truth.json"
DEFAULT_OUTPUT_ROOT = ROOT / "runs" / "package_target_extraction"
DEFAULT_VIEW_SEMANTICS_WARNING_PATH = (
    ROOT / "data" / "tasks" / "package_target_extraction" / "prompts" / "view_semantics_warning.md"
)
ID_PATTERN = re.compile(r"^ID\d+$")
SUPPORTED_PROMPT_CONTEXT_MODES = ("none", "package_name")
SUPPORTED_PROMPT_VARIANTS = ("baseline", "view_semantics_warning", "both")
CONCRETE_PROMPT_VARIANTS = ("baseline", "view_semantics_warning")
PACKAGE_CONTEXT_PLACEHOLDER = "{{PACKAGE_CONTEXT_BLOCK}}"
VIEW_SEMANTICS_WARNING_PLACEHOLDER = "{{VIEW_SEMANTICS_WARNING}}"


class BenchmarkError(Exception):
    pass


class DryRunClient:
    def __init__(self, model: str) -> None:
        self.model = model


def load_env_file(env_path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not env_path.exists():
        return env

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        env[key] = value
        os.environ.setdefault(key, value)
    return env


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BenchmarkError(f"Missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def path_for_record(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_output(prompt_name: str, data: Any) -> tuple[Any | None, list[str]]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return None, ["Output must be a JSON object"]

    if prompt_name == "extract_number":
        expected_keys = {
            "body_long_side",
            "body_short_side",
            "maximum_terminal_to_terminal_span",
            "overall_package_height",
        }
        actual_keys = set(data.keys())
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            if missing:
                errors.append(f"missing keys: {missing}")
            if extra:
                errors.append(f"unexpected keys: {extra}")
        normalized: dict[str, float | int | None] = {}
        for key in expected_keys:
            value = data.get(key)
            if value is None:
                normalized[key] = None
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"{key} must be a number or null")
            else:
                normalized[key] = value
        return (normalized if not errors else None), errors

    if prompt_name == "extract_id":
        expected_keys = {
            "body_side_dimensions",
            "maximum_terminal_to_terminal_span",
            "overall_package_height",
        }
        actual_keys = set(data.keys())
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            if missing:
                errors.append(f"missing keys: {missing}")
            if extra:
                errors.append(f"unexpected keys: {extra}")

        normalized: dict[str, Any] = {}
        body_dims = data.get("body_side_dimensions")
        if body_dims is None:
            normalized["body_side_dimensions"] = None
        elif not isinstance(body_dims, list) or len(body_dims) != 2:
            errors.append("body_side_dimensions must be null or a 2-item array")
        else:
            if not all(isinstance(item, str) and ID_PATTERN.fullmatch(item) for item in body_dims):
                errors.append("body_side_dimensions items must be strings like ID7")
            else:
                normalized["body_side_dimensions"] = sorted(body_dims)

        for key in ("maximum_terminal_to_terminal_span", "overall_package_height"):
            value = data.get(key)
            if value is None:
                normalized[key] = None
            elif isinstance(value, str) and ID_PATTERN.fullmatch(value):
                normalized[key] = value
            else:
                errors.append(f"{key} must be a string like ID7 or null")

        return (normalized if not errors else None), errors

    return None, [f"Unsupported prompt name: {prompt_name}"]


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compare_outputs(prompt_name: str, predicted: Any, expected: Any) -> dict[str, Any]:
    predicted_norm, predicted_errors = normalize_output(prompt_name, predicted)
    expected_norm, expected_errors = normalize_output(prompt_name, expected)
    if expected_errors:
        raise BenchmarkError(f"Ground truth failed validation for {prompt_name}: {expected_errors}")

    field_names = list(expected_norm.keys())
    if predicted_norm is None:
        field_matches = {field: False for field in field_names}
        return {
            "schema_valid": False,
            "validation_errors": predicted_errors,
            "field_matches": field_matches,
            "matched_field_count": 0,
            "field_count": len(field_names),
            "field_match_rate": 0.0,
            "exact_match": False,
            "normalized_output": None,
        }

    field_matches = {field: predicted_norm[field] == expected_norm[field] for field in field_names}
    matched_field_count = sum(1 for matched in field_matches.values() if matched)
    field_count = len(field_names)
    return {
        "schema_valid": True,
        "validation_errors": [],
        "field_matches": field_matches,
        "matched_field_count": matched_field_count,
        "field_count": field_count,
        "field_match_rate": matched_field_count / field_count if field_count else 1.0,
        "exact_match": matched_field_count == field_count,
        "normalized_output": predicted_norm,
    }


def get_nuisance_dimensions_for_case(
    expected_group: dict[str, Any],
    *,
    variant_slug: str,
) -> tuple[bool, list[Any] | None]:
    evaluation_metadata = expected_group.get("evaluation_metadata")
    if not isinstance(evaluation_metadata, dict):
        return False, None

    nuisance_dimensions = evaluation_metadata.get("nuisance_dimensions")
    if not variant_slug.startswith("rotated-"):
        return False, None
    if nuisance_dimensions is None:
        return True, None
    if isinstance(nuisance_dimensions, list):
        return True, nuisance_dimensions
    return True, [nuisance_dimensions]


def get_predicted_height_value(comparison: dict[str, Any], raw_prediction: Any) -> Any | None:
    normalized_output = comparison.get("normalized_output")
    if isinstance(normalized_output, dict) and "overall_package_height" in normalized_output:
        return normalized_output["overall_package_height"]
    if isinstance(raw_prediction, dict) and "overall_package_height" in raw_prediction:
        return raw_prediction["overall_package_height"]
    return None


def build_height_nuisance_analysis(
    *,
    case: dict[str, Any],
    expected_group: dict[str, Any],
    comparison: dict[str, Any],
    raw_prediction: Any,
) -> dict[str, Any]:
    has_metadata_slot, nuisance_dimensions = get_nuisance_dimensions_for_case(
        expected_group,
        variant_slug=case["variant_slug"],
    )
    predicted_value = get_predicted_height_value(comparison, raw_prediction)
    matches_target = comparison["field_matches"].get("overall_package_height")

    if not has_metadata_slot:
        result = "not_applicable"
        matches_nuisance = None
    elif nuisance_dimensions is None:
        result = "metadata_missing"
        matches_nuisance = None
    elif matches_target:
        result = "target_match"
        matches_nuisance = False
    elif predicted_value is None:
        result = "no_prediction"
        matches_nuisance = False
    elif predicted_value in nuisance_dimensions:
        result = "nuisance_match"
        matches_nuisance = True
    else:
        result = "other_mismatch"
        matches_nuisance = False

    return {
        "expected_target": expected_group["ground_truth"].get("overall_package_height"),
        "predicted_value": predicted_value,
        "nuisance_dimensions": nuisance_dimensions,
        "matches_target": matches_target,
        "matches_nuisance": matches_nuisance,
        "result": result,
    }


def extract_json_candidate(text: str) -> str:
    cleaned = text.strip()
    fenced_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    start_positions = [idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx != -1]
    if not start_positions:
        return cleaned
    start = min(start_positions)

    end_positions = [idx for idx in (cleaned.rfind("}"), cleaned.rfind("]")) if idx != -1]
    if not end_positions:
        return cleaned[start:]
    end = max(end_positions)
    if end >= start:
        return cleaned[start : end + 1]
    return cleaned


def parse_json_text(text: str) -> tuple[Any | None, str | None]:
    candidate = extract_json_candidate(text)
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as exc:
        return None, f"JSON parse failed: {exc}"


def detect_mime_type(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    return mime_type or "application/octet-stream"


def write_response_artifacts(
    *,
    run_dir: Path,
    provider: str,
    case_id: str,
    prompt_variant: str,
    repeat_index: int,
    response_text: Any,
    raw_response_text: Any,
    response_json: Any,
) -> dict[str, str]:
    attempt_dir = run_dir / "attempts" / provider / case_id / prompt_variant
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


def execute_provider_attempt(
    *,
    provider: str,
    client: "ProviderClient",
    case: dict[str, Any],
    prompt_text: str,
    prompt_context_mode: str,
    prompt_variant: str,
    view_semantics_warning_enabled: bool,
    image_path: Path,
    temperature: float,
) -> tuple[dict[str, Any], Any | None, str | None]:
    try:
        provider_result = client.run(
            prompt_text=prompt_text,
            image_path=image_path,
        )
        response_text = provider_result["response_text"] or ""
        raw_prediction, parse_error = (
            parse_json_text(response_text) if response_text else (None, "Empty response text")
        )
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
                "prompt_name": case["prompt_name"],
                "prompt_context_mode": prompt_context_mode,
                "prompt_variant": prompt_variant,
                "view_semantics_warning_enabled": view_semantics_warning_enabled,
                "temperature": temperature,
                "structured_output": False,
            },
        }
        raw_prediction = None
        parse_error = error_text

    request_summary = provider_result.get("request_summary")
    if isinstance(request_summary, dict):
        request_summary["prompt_name"] = case["prompt_name"]
        request_summary["prompt_context_mode"] = prompt_context_mode
        request_summary["prompt_variant"] = prompt_variant
        request_summary["view_semantics_warning_enabled"] = view_semantics_warning_enabled
    return provider_result, raw_prediction, parse_error


def sanitize_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-")


def filter_cases(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = []
    provider_set = set(args.providers)
    if not provider_set:
        raise BenchmarkError("At least one provider must be selected")

    case_ids = set(args.case_id or [])
    answer_keys = set(args.answer_key or [])
    package_slugs = set(args.package_slug or [])
    prompt_names = set(args.prompt_name or [])

    for case in cases:
        if case_ids and case["case_id"] not in case_ids:
            continue
        if answer_keys and case["answer_key"] not in answer_keys:
            continue
        if package_slugs and case["package_slug"] not in package_slugs:
            continue
        if prompt_names and case["prompt_name"] not in prompt_names:
            continue
        selected.append(case)

    if args.max_cases is not None:
        selected = selected[: args.max_cases]

    if not selected:
        raise BenchmarkError("No cases matched the requested filters")
    return selected


def build_prompt_template_cache(cases: list[dict[str, Any]]) -> dict[str, str]:
    cache: dict[str, str] = {}
    for case in cases:
        prompt_path = ROOT / case["prompt_path"]
        cache.setdefault(case["prompt_name"], read_text(prompt_path))
    return cache


def build_package_context_block(case: dict[str, Any], prompt_context_mode: str) -> str:
    if prompt_context_mode == "none":
        return ""
    if prompt_context_mode == "package_name":
        return (
            "Known package context:\n"
            f"- Package name: {case['package_name']}\n\n"
            "Use this package context only to identify the package family and interpret the drawing.\n"
            "If a requested dimension cannot be determined from the image, return null."
        )
    raise BenchmarkError(f"Unsupported prompt context mode: {prompt_context_mode}")


def resolve_prompt_variants(prompt_variant: str) -> list[str]:
    if prompt_variant == "both":
        return list(CONCRETE_PROMPT_VARIANTS)
    if prompt_variant in CONCRETE_PROMPT_VARIANTS:
        return [prompt_variant]
    raise BenchmarkError(f"Unsupported prompt variant: {prompt_variant}")


def build_view_semantics_warning_block(prompt_variant: str, view_semantics_warning_text: str) -> str:
    if prompt_variant == "baseline":
        return ""
    if prompt_variant == "view_semantics_warning":
        return view_semantics_warning_text.strip()
    raise BenchmarkError(f"Unsupported prompt variant: {prompt_variant}")


def render_prompt(
    prompt_template: str,
    *,
    case: dict[str, Any],
    prompt_context_mode: str,
    prompt_variant: str,
    view_semantics_warning_text: str,
) -> str:
    if PACKAGE_CONTEXT_PLACEHOLDER not in prompt_template:
        raise BenchmarkError(
            f"Prompt template for `{case['prompt_name']}` is missing {PACKAGE_CONTEXT_PLACEHOLDER}"
        )
    if VIEW_SEMANTICS_WARNING_PLACEHOLDER not in prompt_template:
        raise BenchmarkError(
            f"Prompt template for `{case['prompt_name']}` is missing {VIEW_SEMANTICS_WARNING_PLACEHOLDER}"
        )
    prompt_context_block = build_package_context_block(case, prompt_context_mode)
    view_semantics_warning_block = build_view_semantics_warning_block(
        prompt_variant,
        view_semantics_warning_text,
    )
    rendered = prompt_template.replace(PACKAGE_CONTEXT_PLACEHOLDER, prompt_context_block)
    rendered = rendered.replace(VIEW_SEMANTICS_WARNING_PLACEHOLDER, view_semantics_warning_block)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered).strip()
    return rendered + "\n"


def load_ground_truth_map(ground_truth_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    groups = ground_truth_payload.get("answer_groups")
    if not isinstance(groups, list):
        raise BenchmarkError("Ground truth file must contain `answer_groups`")
    answer_map = {}
    for group in groups:
        answer_key = group["answer_key"]
        answer_map[answer_key] = group
    return answer_map


def summarize_attempts(attempts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_summary_rows: list[dict[str, Any]] = []
    provider_summary_rows: list[dict[str, Any]] = []

    by_provider_case_variant: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_provider_variant: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        case_key = (attempt["provider"], attempt["case_id"], attempt["prompt_variant"])
        provider_key = (attempt["provider"], attempt["prompt_variant"])
        by_provider_case_variant[case_key].append(attempt)
        by_provider_variant[provider_key].append(attempt)

    for (provider, case_id, prompt_variant), rows in sorted(by_provider_case_variant.items()):
        successful_rows = [row for row in rows if row["schema_valid"]]
        exact_rows = [row for row in rows if row["exact_match"]]
        normalized_counter = Counter(
            canonical_json(row["normalized_output"])
            for row in successful_rows
            if row["normalized_output"] is not None
        )
        most_common_count = normalized_counter.most_common(1)[0][1] if normalized_counter else 0
        case_summary_rows.append(
            {
                "provider": provider,
                "case_id": case_id,
                "image_id": rows[0]["image_id"],
                "answer_key": rows[0]["answer_key"],
                "package_slug": rows[0]["package_slug"],
                "prompt_name": rows[0]["prompt_name"],
                "prompt_variant": prompt_variant,
                "view_semantics_warning_enabled": rows[0]["view_semantics_warning_enabled"],
                "repeats": len(rows),
                "successful_schema_runs": len(successful_rows),
                "schema_valid_rate": round(len(successful_rows) / len(rows), 4),
                "exact_match_runs": len(exact_rows),
                "exact_match_rate": round(len(exact_rows) / len(rows), 4),
                "mean_field_match_rate": round(
                    sum(row["field_match_rate"] for row in rows) / len(rows), 4
                ),
                "unique_valid_outputs": len(normalized_counter),
                "stability_rate": round(
                    most_common_count / len(successful_rows), 4
                ) if successful_rows else 0.0,
            }
        )

    for (provider, prompt_variant), rows in sorted(by_provider_variant.items()):
        provider_summary_rows.append(
            {
                "provider": provider,
                "model": rows[0]["model"],
                "prompt_variant": prompt_variant,
                "view_semantics_warning_enabled": rows[0]["view_semantics_warning_enabled"],
                "attempt_count": len(rows),
                "schema_valid_rate": round(
                    sum(1 for row in rows if row["schema_valid"]) / len(rows), 4
                ),
                "exact_match_rate": round(
                    sum(1 for row in rows if row["exact_match"]) / len(rows), 4
                ),
                "mean_field_match_rate": round(
                    sum(row["field_match_rate"] for row in rows) / len(rows), 4
                ),
                "mean_latency_ms": round(
                    sum(row["latency_ms"] for row in rows) / len(rows), 2
                ),
            }
        )

    return case_summary_rows, provider_summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_attempt(
    *,
    run_dir: Path,
    provider: str,
    client: ProviderClient | DryRunClient,
    case: dict[str, Any],
    expected_group: dict[str, Any],
    prompt_text: str,
    prompt_context_mode: str,
    prompt_variant: str,
    view_semantics_warning_enabled: bool,
    repeat_index: int,
    temperature: float,
    dry_run: bool,
    retry_policy: RetryPolicy,
) -> dict[str, Any]:
    image_path = ROOT / case["image_path"]
    attempt_id = f"{provider}__{case['case_id']}__{prompt_variant}__run-{repeat_index:03d}"
    started = time.perf_counter()
    retry_attempts: list[dict[str, Any]] = []

    if dry_run:
        raw_prediction = expected_group["ground_truth"]
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
                "prompt_name": case["prompt_name"],
                "prompt_context_mode": prompt_context_mode,
                "prompt_variant": prompt_variant,
                "view_semantics_warning_enabled": view_semantics_warning_enabled,
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
        for attempt_number in range(1, retry_policy.max_attempts + 1):
            provider_result, raw_prediction, parse_error = execute_provider_attempt(
                provider=provider,
                client=client,
                case=case,
                prompt_text=prompt_text,
                prompt_context_mode=prompt_context_mode,
                prompt_variant=prompt_variant,
                view_semantics_warning_enabled=view_semantics_warning_enabled,
                image_path=image_path,
                temperature=temperature,
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

            sleep_seconds = 0.0
            if parse_error and retryable:
                sleep_seconds = calculate_retry_delay(attempt_number, retry_policy)

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

            if not parse_error:
                break
            if not retryable:
                break
            time.sleep(sleep_seconds)

    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    request_summary = provider_result.get("request_summary")
    if isinstance(request_summary, dict):
        request_summary["prompt_context_mode"] = prompt_context_mode
        request_summary["prompt_variant"] = prompt_variant
        request_summary["view_semantics_warning_enabled"] = view_semantics_warning_enabled
    if dry_run:
        comparison = compare_outputs(case["prompt_name"], raw_prediction, expected_group["ground_truth"])
    elif parse_error:
        comparison = compare_outputs(case["prompt_name"], {"_parse_error": parse_error}, expected_group["ground_truth"])
    else:
        comparison = compare_outputs(case["prompt_name"], raw_prediction, expected_group["ground_truth"])

    retry_count = max(0, len(retry_attempts) - 1)
    retried = retry_count > 0
    if parse_error:
        final_outcome = "failed_after_retries" if retried else "failed_without_retry"
    else:
        final_outcome = "success_after_retry" if retried else "success_first_try"

    attempt_record = {
        "attempt_id": attempt_id,
        "provider": provider,
        "model": client.model,
        "case_id": case["case_id"],
        "image_id": case["image_id"],
        "answer_key": case["answer_key"],
        "package_name": case["package_name"],
        "package_slug": case["package_slug"],
        "variant_name": case["variant_name"],
        "variant_slug": case["variant_slug"],
        "prompt_name": case["prompt_name"],
        "prompt_context_mode": prompt_context_mode,
        "prompt_variant": prompt_variant,
        "view_semantics_warning_enabled": view_semantics_warning_enabled,
        "rendered_prompt": prompt_text,
        "repeat_index": repeat_index,
        "retried": retried,
        "retry_count": retry_count,
        "retry_summary": {
            "retry_policy": retry_policy.to_dict(),
            "attempt_count": len(retry_attempts),
            "final_outcome": final_outcome,
            "retryable_error_count": sum(
                1 for item in retry_attempts if item.get("retryable")
            ),
        },
        "retry_attempts": retry_attempts,
        "latency_ms": latency_ms,
        "status_code": provider_result["status_code"],
        "response_text": provider_result["response_text"],
        "raw_prediction": raw_prediction,
        "parse_error": parse_error,
        "schema_valid": comparison["schema_valid"],
        "validation_errors": comparison["validation_errors"],
        "field_match_rate": comparison["field_match_rate"],
        "matched_field_count": comparison["matched_field_count"],
        "field_count": comparison["field_count"],
        "field_matches": comparison["field_matches"],
        "exact_match": comparison["exact_match"],
        "normalized_output": comparison["normalized_output"],
        "expected_output": expected_group["ground_truth"],
        "request_summary": provider_result["request_summary"],
        "raw_response_text": provider_result["raw_response_text"],
        "response_json": provider_result["response_json"],
    }
    attempt_record["height_nuisance_analysis"] = build_height_nuisance_analysis(
        case=case,
        expected_group=expected_group,
        comparison=comparison,
        raw_prediction=raw_prediction,
    )
    attempt_record["artifact_paths"] = write_response_artifacts(
        run_dir=run_dir,
        provider=provider,
        case_id=case["case_id"],
        prompt_variant=prompt_variant,
        repeat_index=repeat_index,
        response_text=provider_result["response_text"],
        raw_response_text=provider_result["raw_response_text"],
        response_json=provider_result["response_json"],
    )
    write_json(
        run_dir / "attempts" / provider / case["case_id"] / prompt_variant / f"run-{repeat_index:03d}.json",
        attempt_record,
    )
    return attempt_record


def sort_attempts(attempts: list[dict[str, Any]], providers: list[str]) -> list[dict[str, Any]]:
    provider_order = {provider: index for index, provider in enumerate(providers)}
    return sorted(
        attempts,
        key=lambda row: (
            provider_order.get(row["provider"], len(provider_order)),
            row["case_id"],
            row["prompt_variant"],
            row["repeat_index"],
        ),
    )


def run_benchmark(args: argparse.Namespace) -> Path:
    load_env_file(args.env_file)
    retry_policy = build_retry_policy_from_args(args)
    sdk_versions = get_installed_sdk_versions()
    prompt_variants = resolve_prompt_variants(args.prompt_variant)
    view_semantics_warning_text = (
        read_text(args.view_semantics_warning_path)
        if "view_semantics_warning" in prompt_variants
        else ""
    )
    view_semantics_warning_sha256 = hashlib.sha256(
        view_semantics_warning_text.encode("utf-8")
    ).hexdigest()
    cases_payload = read_json(args.cases)
    ground_truth_payload = read_json(args.ground_truth)
    answer_map = load_ground_truth_map(ground_truth_payload)
    all_cases = cases_payload.get("cases")
    if not isinstance(all_cases, list):
        raise BenchmarkError("Cases file must contain a `cases` array")
    selected_cases = filter_cases(all_cases, args)
    prompt_template_cache = build_prompt_template_cache(selected_cases)

    missing_ground_truth = sorted(
        {case["answer_key"] for case in selected_cases} - set(answer_map)
    )
    if missing_ground_truth:
        raise BenchmarkError(f"Missing ground truth for answer keys: {missing_ground_truth}")

    provider_runtime = resolve_provider_runtime(
        args,
        dry_run=args.dry_run,
        env_file=args.env_file,
    )
    models = provider_runtime.models
    batch_sizes = provider_runtime.batch_sizes

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"benchmark-{timestamp}"
    run_dir = args.output_root / sanitize_slug(run_name)
    if run_dir.exists():
        raise BenchmarkError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    run_config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "transport": "provider APIs",
        "sdk_versions": sdk_versions,
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
        "prompt_context_mode": args.prompt_context_mode,
        "prompt_variant": args.prompt_variant,
        "prompt_variants": prompt_variants,
        "view_semantics_warning_path": path_for_record(args.view_semantics_warning_path),
        "view_semantics_warning_sha256": view_semantics_warning_sha256,
        "view_semantics_warning_length": len(view_semantics_warning_text),
        "ground_truth_path": path_for_record(args.ground_truth),
        "cases_path": path_for_record(args.cases),
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
    work_queues: dict[str, queue.Queue[tuple[dict[str, Any], int, str] | None]] = {}
    worker_threads: list[threading.Thread] = []

    def provider_worker(provider: str, worker_index: int) -> None:
        client = clients[provider]
        task_queue = work_queues[provider]
        while True:
            item = task_queue.get()
            try:
                if item is None:
                    break

                case, repeat_index, prompt_variant = item
                view_semantics_warning_enabled = prompt_variant == "view_semantics_warning"
                expected_group = answer_map[case["answer_key"]]
                prompt_template = prompt_template_cache[case["prompt_name"]]
                prompt_text = render_prompt(
                    prompt_template,
                    case=case,
                    prompt_context_mode=args.prompt_context_mode,
                    prompt_variant=prompt_variant,
                    view_semantics_warning_text=view_semantics_warning_text,
                )
                attempt_record = process_attempt(
                    run_dir=run_dir,
                    provider=provider,
                    client=client,
                    case=case,
                    expected_group=expected_group,
                    prompt_text=prompt_text,
                    prompt_context_mode=args.prompt_context_mode,
                    prompt_variant=prompt_variant,
                    view_semantics_warning_enabled=view_semantics_warning_enabled,
                    repeat_index=repeat_index,
                    temperature=args.temperature,
                    dry_run=args.dry_run,
                    retry_policy=retry_policy,
                )
                with attempts_lock:
                    attempts.append(attempt_record)
                print(
                    f"[{provider}] {case['case_id']} {prompt_variant} run {repeat_index}/{args.repeats} "
                    f"status={attempt_record['status_code']} exact={attempt_record['exact_match']} "
                    f"schema_valid={attempt_record['schema_valid']} worker={worker_index}",
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
        provider_queue: queue.Queue[tuple[dict[str, Any], int, str] | None] = queue.Queue()
        work_queues[provider] = provider_queue
        for case in selected_cases:
            for repeat_index in range(1, args.repeats + 1):
                for prompt_variant in prompt_variants:
                    provider_queue.put((case, repeat_index, prompt_variant))
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
            "attempt_id": row["attempt_id"],
            "provider": row["provider"],
            "model": row["model"],
            "case_id": row["case_id"],
            "image_id": row["image_id"],
            "answer_key": row["answer_key"],
            "package_slug": row["package_slug"],
            "variant_name": row["variant_name"],
            "variant_slug": row["variant_slug"],
            "prompt_name": row["prompt_name"],
            "prompt_context_mode": row["prompt_context_mode"],
            "prompt_variant": row["prompt_variant"],
            "view_semantics_warning_enabled": row["view_semantics_warning_enabled"],
            "repeat_index": row["repeat_index"],
            "retried": row["retried"],
            "retry_count": row["retry_count"],
            "retry_final_outcome": row["retry_summary"]["final_outcome"],
            "latency_ms": row["latency_ms"],
            "status_code": row["status_code"],
            "schema_valid": row["schema_valid"],
            "exact_match": row["exact_match"],
            "field_match_rate": row["field_match_rate"],
            "parse_error": row["parse_error"] or "",
            "validation_errors": "; ".join(row["validation_errors"]),
        }
        for row in attempts
    ]
    case_summary_rows, provider_summary_rows = summarize_attempts(attempts)
    api_error_rows = build_api_error_rows(attempts)

    write_json(run_dir / "summary.json", {
        "config": run_config,
        "api_error_count": len(api_error_rows),
        "provider_summary": provider_summary_rows,
        "case_summary": case_summary_rows,
    })
    write_csv(run_dir / "attempts.csv", attempts_csv_rows)
    write_csv(run_dir / "case_summary.csv", case_summary_rows)
    write_csv(run_dir / "provider_summary.csv", provider_summary_rows)
    write_csv(run_dir / "api_errors.csv", api_error_rows)
    assert_no_api_errors(api_error_rows, allow_api_errors=args.allow_api_errors, run_dir=run_dir)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run multi-provider image extraction benchmarks for the package drawing dataset."
    )
    add_provider_arguments(parser)
    parser.add_argument("--repeats", type=int, default=1, help="How many times to run each provider on each case.")
    parser.add_argument("--case-id", action="append", help="Limit execution to one or more case IDs.")
    parser.add_argument("--answer-key", action="append", help="Limit execution to one or more answer keys.")
    parser.add_argument("--package-slug", action="append", help="Limit execution to one or more package slugs.")
    parser.add_argument("--prompt-name", action="append", help="Limit execution to one or more prompt names.")
    parser.add_argument("--max-cases", type=int, help="Limit the number of matched cases after filtering.")
    parser.add_argument("--delay-seconds", type=float, default=0.0, help="Optional delay between attempts.")
    parser.add_argument("--timeout-seconds", type=int, default=180, help="HTTP timeout per request.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature to send when supported.")
    parser.add_argument(
        "--prompt-context-mode",
        choices=SUPPORTED_PROMPT_CONTEXT_MODES,
        default="none",
        help="How much package metadata to inject into the prompt. Default: none.",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=SUPPORTED_PROMPT_VARIANTS,
        default="baseline",
        help=(
            "Prompt variant to run. `baseline` removes the view semantics warning; "
            "`view_semantics_warning` inserts it; `both` runs both variants. Default: baseline."
        ),
    )
    parser.add_argument(
        "--view-semantics-warning-path",
        type=Path,
        default=DEFAULT_VIEW_SEMANTICS_WARNING_PATH,
        help="Prompt fragment inserted when --prompt-variant is view_semantics_warning or both.",
    )
    parser.add_argument("--run-name", help="Optional custom output folder name under runs/.")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls and use ground truth as fake responses to validate the pipeline.")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH, help="Path to the .env file.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH, help="Path to task cases JSON.")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH_PATH, help="Path to ground truth JSON.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory where run outputs are written.")
    add_retry_arguments(parser)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    try:
        run_dir = run_benchmark(args)
    except (BenchmarkError, ProviderError, ApiErrorGuardError, ValueError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    print(f"Benchmark outputs written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
