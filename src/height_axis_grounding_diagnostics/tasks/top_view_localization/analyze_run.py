#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from height_axis_grounding_diagnostics.common.io_utils import parse_json_text, read_json, write_csv  # noqa: E402
from height_axis_grounding_diagnostics.common.retry import ApiErrorGuardError, raise_if_run_has_api_errors  # noqa: E402
from height_axis_grounding_diagnostics.tasks.top_view_localization.schema import (  # noqa: E402
    SLOT_ORDER,
    compare_top_view_outputs,
    validate_bbox,
    validate_top_view_output,
)


TASK_NAME = "top_view_localization"
DEFAULT_ANALYSIS_ROOT = ROOT / "analysis" / TASK_NAME
ATTEMPT_RECORD_RE = re.compile(r"^run-\d+\.json$")


class AnalysisError(Exception):
    pass


def canonical(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def refresh_attempt_from_response_text(attempt_path: Path, attempt: dict[str, Any]) -> dict[str, Any]:
    refreshed = dict(attempt)
    response_path = attempt_path.with_suffix(".response.txt")
    if response_path.exists():
        response_text = response_path.read_text(encoding="utf-8")
        refreshed["response_text"] = response_text
        refreshed["analysis_prediction_source"] = "response_text"
        raw_prediction, parse_error = (
            parse_json_text(response_text) if response_text else (None, "Empty response text")
        )
        refreshed["raw_prediction"] = raw_prediction
        refreshed["parse_error"] = parse_error

    expected_output = refreshed.get("expected_output")
    if not isinstance(expected_output, dict):
        return refreshed

    raw_prediction = refreshed.get("raw_prediction")
    parse_error = refreshed.get("parse_error")
    comparison = (
        compare_top_view_outputs({"_parse_error": parse_error}, expected_output)
        if parse_error
        else compare_top_view_outputs(raw_prediction, expected_output)
    )
    if parse_error:
        comparison["top_view_result_type"] = "parse_error"
    for key, value in comparison.items():
        refreshed[key] = value
    return refreshed


def load_attempts(run_dir: Path) -> list[dict[str, Any]]:
    attempt_files = sorted(
        path
        for path in run_dir.glob("attempts/*/*/run-*.json")
        if ATTEMPT_RECORD_RE.fullmatch(path.name)
    )
    if not attempt_files:
        raise AnalysisError(f"No attempt JSON files found under {run_dir}")

    attempts: list[dict[str, Any]] = []
    for path in attempt_files:
        attempt = read_json(path)
        if not isinstance(attempt, dict):
            raise AnalysisError(f"Attempt record must be a JSON object: {path}")
        attempts.append(refresh_attempt_from_response_text(path, attempt))
    return attempts


def build_attempt_rows(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        rows.append(
            {
                "attempt_id": attempt["attempt_id"],
                "provider": attempt["provider"],
                "model": attempt["model"],
                "case_id": attempt["case_id"],
                "image_id": attempt["image_id"],
                "package_slug": attempt["package_slug"],
                "variant_slug": attempt["variant_slug"],
                "repeat_index": attempt["repeat_index"],
                "schema_valid": bool(attempt.get("schema_valid")),
                "answer_schema_valid": bool(attempt.get("answer_schema_valid")),
                "layout_exact_match": bool(attempt.get("layout_exact_match")),
                "layout_slot_correct_count": attempt.get("layout_slot_correct_count", 0),
                "layout_slot_total_count": attempt.get("layout_slot_total_count", len(SLOT_ORDER)),
                "layout_slot_accuracy": attempt.get("layout_slot_accuracy", 0.0),
                "occupied_slot_precision": attempt.get("occupied_slot_precision", 0.0),
                "occupied_slot_recall": attempt.get("occupied_slot_recall", 0.0),
                "occupied_slot_f1": attempt.get("occupied_slot_f1", 0.0),
                "occupied_slot_matched_count": attempt.get("occupied_slot_matched_count", 0),
                "expected_occupied_slot_count": attempt.get("expected_occupied_slot_count", 0),
                "predicted_occupied_slot_count": attempt.get("predicted_occupied_slot_count", 0),
                "bbox_output_valid": bool(attempt.get("bbox_output_valid")),
                "expected_top_view_slot": attempt.get("expected_top_view_slot", ""),
                "predicted_top_view_slot": attempt.get("predicted_top_view_slot", ""),
                "top_view_slot_match": bool(attempt.get("top_view_slot_match")),
                "top_view_result_type": attempt.get("top_view_result_type", ""),
                "exact_match": bool(attempt.get("exact_match")),
                "parse_error": attempt.get("parse_error") or "",
                "validation_errors": "; ".join(attempt.get("validation_errors") or []),
            }
        )
    return rows


def build_layout_rows(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        expected_output = attempt.get("expected_output")
        normalized_output = attempt.get("normalized_output")
        expected_norm, _ = (
            validate_top_view_output(expected_output, context="expected", require_bbox=False)
            if isinstance(expected_output, dict)
            else (None, [])
        )
        predicted_norm = normalized_output if isinstance(normalized_output, dict) else None
        rows.append(
            {
                "attempt_id": attempt["attempt_id"],
                "provider": attempt["provider"],
                "model": attempt["model"],
                "case_id": attempt["case_id"],
                "image_id": attempt["image_id"],
                "package_slug": attempt["package_slug"],
                "variant_slug": attempt["variant_slug"],
                "repeat_index": attempt["repeat_index"],
                "expected_layout": canonical(expected_norm.get("layout") if expected_norm else None),
                "predicted_layout": canonical(predicted_norm.get("layout") if predicted_norm else None),
                "layout_exact_match": bool(attempt.get("layout_exact_match")),
                "occupied_slot_precision": attempt.get("occupied_slot_precision", 0.0),
                "occupied_slot_recall": attempt.get("occupied_slot_recall", 0.0),
                "occupied_slot_f1": attempt.get("occupied_slot_f1", 0.0),
            }
        )
    return rows


def build_top_view_rows(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        rows.append(
            {
                "attempt_id": attempt["attempt_id"],
                "provider": attempt["provider"],
                "model": attempt["model"],
                "case_id": attempt["case_id"],
                "image_id": attempt["image_id"],
                "package_slug": attempt["package_slug"],
                "variant_slug": attempt["variant_slug"],
                "repeat_index": attempt["repeat_index"],
                "expected_top_view_slot": attempt.get("expected_top_view_slot", ""),
                "predicted_top_view_slot": attempt.get("predicted_top_view_slot", ""),
                "matched": bool(attempt.get("top_view_slot_match")),
                "result_type": attempt.get("top_view_result_type", ""),
                "schema_valid": bool(attempt.get("schema_valid")),
                "answer_schema_valid": bool(attempt.get("answer_schema_valid")),
                "parse_error": attempt.get("parse_error") or "",
                "validation_errors": "; ".join(attempt.get("validation_errors") or []),
            }
        )
    return rows


def build_bbox_rows(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        raw_prediction = attempt.get("raw_prediction")
        views = raw_prediction.get("views") if isinstance(raw_prediction, dict) else None
        if not isinstance(views, list):
            rows.append(
                {
                    "attempt_id": attempt["attempt_id"],
                    "provider": attempt["provider"],
                    "model": attempt["model"],
                    "case_id": attempt["case_id"],
                    "image_id": attempt["image_id"],
                    "package_slug": attempt["package_slug"],
                    "variant_slug": attempt["variant_slug"],
                    "repeat_index": attempt["repeat_index"],
                    "view_index": "",
                    "slot": "",
                    "predicted_bbox": "",
                    "bbox_present": False,
                    "bbox_valid": False,
                    "bbox_errors": "prediction.views must be an array",
                }
            )
            continue

        for view_index, view in enumerate(views):
            bbox_errors: list[str] = []
            bbox = view.get("bounding_box_2d") if isinstance(view, dict) else None
            bbox_present = isinstance(view, dict) and "bounding_box_2d" in view
            bbox_valid = validate_bbox(
                bbox,
                f"prediction.views[{view_index}].bounding_box_2d",
                bbox_errors,
            ) is not None and not bbox_errors
            rows.append(
                {
                    "attempt_id": attempt["attempt_id"],
                    "provider": attempt["provider"],
                    "model": attempt["model"],
                    "case_id": attempt["case_id"],
                    "image_id": attempt["image_id"],
                    "package_slug": attempt["package_slug"],
                    "variant_slug": attempt["variant_slug"],
                    "repeat_index": attempt["repeat_index"],
                    "view_index": view_index,
                    "slot": view.get("slot") if isinstance(view, dict) else "",
                    "predicted_bbox": canonical(bbox) if bbox_present else "",
                    "bbox_present": bbox_present,
                    "bbox_valid": bbox_valid,
                    "bbox_errors": "; ".join(bbox_errors),
                }
            )
    return rows


def ratio(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def summarize_attempt_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    metric_names = [
        "schema_valid",
        "answer_schema_valid",
        "layout_exact_match",
        "layout_slot_accuracy",
        "occupied_slot_precision",
        "occupied_slot_recall",
        "occupied_slot_f1",
        "bbox_output_valid",
        "top_view_slot_match",
        "exact_match",
    ]
    count_names = [
        "layout_slot_correct_count",
        "layout_slot_total_count",
        "occupied_slot_matched_count",
        "expected_occupied_slot_count",
        "predicted_occupied_slot_count",
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for group_values, group_rows in sorted(grouped.items()):
        summary = {key: value for key, value in zip(group_keys, group_values)}
        total = len(group_rows)
        summary["attempt_count"] = total
        summary["case_count"] = len({row["case_id"] for row in group_rows})
        for metric in metric_names:
            summary[f"case_macro_{metric}"] = round(
                sum(float(row[metric]) for row in group_rows) / total,
                4,
            )
        summary["case_exact_match_count"] = sum(1 for row in group_rows if row["exact_match"])
        summary["case_schema_valid_count"] = sum(1 for row in group_rows if row["schema_valid"])
        summary["case_answer_schema_valid_count"] = sum(1 for row in group_rows if row["answer_schema_valid"])
        summary["case_bbox_output_valid_count"] = sum(1 for row in group_rows if row["bbox_output_valid"])
        summary["top_view_match_count"] = sum(1 for row in group_rows if row["top_view_slot_match"])
        for count_name in count_names:
            summary[count_name] = sum(int(row.get(count_name, 0) or 0) for row in group_rows)

        occupied_matched = summary["occupied_slot_matched_count"]
        occupied_expected = summary["expected_occupied_slot_count"]
        occupied_predicted = summary["predicted_occupied_slot_count"]
        summary["answer_micro_layout_slot_accuracy"] = ratio(
            summary["layout_slot_correct_count"],
            summary["layout_slot_total_count"],
        )
        summary["answer_micro_occupied_slot_precision"] = ratio(occupied_matched, occupied_predicted)
        summary["answer_micro_occupied_slot_recall"] = ratio(occupied_matched, occupied_expected)
        occupied_precision = summary["answer_micro_occupied_slot_precision"]
        occupied_recall = summary["answer_micro_occupied_slot_recall"]
        summary["answer_micro_occupied_slot_f1"] = (
            round(2 * occupied_precision * occupied_recall / (occupied_precision + occupied_recall), 4)
            if occupied_precision + occupied_recall
            else 0.0
        )
        summary["answer_micro_top_view_accuracy"] = ratio(summary["top_view_match_count"], total)
        summary_rows.append(summary)
    return summary_rows


def analyze_run(run_dir: Path, output_dir: Path | None, *, allow_api_errors: bool = False) -> Path:
    raise_if_run_has_api_errors(run_dir, allow_api_errors=allow_api_errors)
    attempts = load_attempts(run_dir)
    attempt_rows = build_attempt_rows(attempts)
    layout_rows = build_layout_rows(attempts)
    top_view_rows = build_top_view_rows(attempts)
    bbox_rows = build_bbox_rows(attempts)
    provider_summary = summarize_attempt_rows(attempt_rows, ["provider", "model"])
    case_summary = summarize_attempt_rows(
        attempt_rows,
        ["provider", "case_id", "image_id", "variant_slug"],
    )

    resolved_output_dir = output_dir.resolve() if output_dir else (DEFAULT_ANALYSIS_ROOT / run_dir.name).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(resolved_output_dir / "attempt_top_view_results.csv", attempt_rows)
    write_csv(resolved_output_dir / "provider_top_view_summary.csv", provider_summary)
    write_csv(resolved_output_dir / "case_top_view_summary.csv", case_summary)
    write_csv(resolved_output_dir / "layout_results.csv", layout_rows)
    write_csv(resolved_output_dir / "top_view_results.csv", top_view_rows)
    write_csv(resolved_output_dir / "bbox_results.csv", bbox_rows)
    return resolved_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate top-view localization analysis tables from a benchmark run."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-api-errors",
        action="store_true",
        help="Analyze even when the run still contains provider API errors after retries.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    try:
        output_dir = analyze_run(run_dir, args.output_dir, allow_api_errors=args.allow_api_errors)
    except (AnalysisError, ApiErrorGuardError) as exc:
        print(f"Analysis failed: {exc}")
        return 1
    print(f"Analysis outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
