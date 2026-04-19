#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from height_axis_grounding_diagnostics.common.retry import ApiErrorGuardError, raise_if_run_has_api_errors  # noqa: E402

DEFAULT_ANALYSIS_DIR = ROOT / "analysis" / "package_target_extraction"
DEFAULT_GROUND_TRUTH_PATH = ROOT / "data" / "tasks" / "package_target_extraction" / "ground_truth.json"
ATTEMPT_RECORD_RE = re.compile(r"^run-\d+\.json$")
ID_PATTERN = re.compile(r"^ID\d+$")


class AnalysisError(Exception):
    pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AnalysisError(f"Missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Invalid JSON in {path}: {exc}") from exc


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


def canonical_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


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


def compare_outputs(prompt_name: str, predicted: Any, expected: Any) -> dict[str, Any]:
    predicted_norm, predicted_errors = normalize_output(prompt_name, predicted)
    expected_norm, expected_errors = normalize_output(prompt_name, expected)
    if expected_errors:
        raise AnalysisError(f"Ground truth failed validation for {prompt_name}: {expected_errors}")

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


def get_variant_slug(attempt: dict[str, Any]) -> str:
    variant_slug = attempt.get("variant_slug")
    if isinstance(variant_slug, str) and variant_slug:
        return variant_slug
    image_id = attempt.get("image_id")
    if isinstance(image_id, str) and "__" in image_id:
        return image_id.split("__", 1)[1]
    return ""


def get_prompt_variant(attempt: dict[str, Any]) -> str:
    prompt_variant = attempt.get("prompt_variant")
    if not isinstance(prompt_variant, str) or not prompt_variant:
        raise AnalysisError(f"Attempt missing prompt_variant: {attempt.get('attempt_id')}")
    return prompt_variant


def load_ground_truth_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    answer_groups = payload.get("answer_groups")
    if not isinstance(answer_groups, list):
        raise AnalysisError(f"Ground truth file must contain an `answer_groups` array: {path}")
    mapping: dict[str, dict[str, Any]] = {}
    for group in answer_groups:
        if not isinstance(group, dict):
            raise AnalysisError(f"Ground truth answer_groups entries must be objects: {path}")
        answer_key = group.get("answer_key")
        if not isinstance(answer_key, str) or not answer_key:
            raise AnalysisError(f"Ground truth answer_group missing valid answer_key: {path}")
        mapping[answer_key] = group
    return mapping


def normalize_nuisance_dimensions(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def get_predicted_height_value(attempt: dict[str, Any]) -> Any:
    normalized_output = attempt.get("normalized_output")
    raw_prediction = attempt.get("raw_prediction")
    if isinstance(normalized_output, dict) and "overall_package_height" in normalized_output:
        return normalized_output["overall_package_height"]
    if isinstance(raw_prediction, dict) and "overall_package_height" in raw_prediction:
        return raw_prediction["overall_package_height"]
    return None


def compute_height_nuisance_analysis(
    attempt: dict[str, Any],
    ground_truth_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected_output = attempt.get("expected_output") or {}
    expected_target = expected_output.get("overall_package_height")
    predicted_value = get_predicted_height_value(attempt)

    if not get_variant_slug(attempt).startswith("rotated-"):
        return {
            "expected_target": expected_target,
            "predicted_value": predicted_value,
            "nuisance_dimensions": None,
            "matches_target": predicted_value == expected_target if predicted_value is not None else False,
            "matches_nuisance": None,
            "result": "not_applicable",
        }

    answer_key = attempt.get("answer_key")
    answer_group = ground_truth_map.get(answer_key)
    if answer_group is None:
        raise AnalysisError(f"Missing answer_key in ground truth: {answer_key}")

    evaluation_metadata = answer_group.get("evaluation_metadata")
    if not isinstance(evaluation_metadata, dict):
        raise AnalysisError(f"Ground truth answer group missing evaluation_metadata: {answer_key}")

    nuisance_dimensions = normalize_nuisance_dimensions(
        evaluation_metadata.get("nuisance_dimensions")
    )

    if nuisance_dimensions is None:
        return {
            "expected_target": expected_target,
            "predicted_value": predicted_value,
            "nuisance_dimensions": None,
            "matches_target": predicted_value == expected_target,
            "matches_nuisance": None,
            "result": "metadata_missing",
        }

    if predicted_value is None:
        return {
            "expected_target": expected_target,
            "predicted_value": None,
            "nuisance_dimensions": nuisance_dimensions,
            "matches_target": False,
            "matches_nuisance": False,
            "result": "no_prediction",
        }

    matches_target = predicted_value == expected_target
    matches_nuisance = predicted_value in nuisance_dimensions
    if matches_target:
        result = "target_match"
    elif matches_nuisance:
        result = "nuisance_match"
    else:
        result = "other_mismatch"

    return {
        "expected_target": expected_target,
        "predicted_value": predicted_value,
        "nuisance_dimensions": nuisance_dimensions,
        "matches_target": matches_target,
        "matches_nuisance": matches_nuisance,
        "result": result,
    }


def refresh_attempt_from_response_text(attempt_path: Path, attempt: dict[str, Any]) -> dict[str, Any]:
    response_path = attempt_path.with_suffix(".response.txt")
    if not response_path.exists():
        return attempt

    refreshed = dict(attempt)
    response_text = response_path.read_text(encoding="utf-8")
    refreshed["response_text"] = response_text
    refreshed["analysis_prediction_source"] = "response_text"

    prompt_name = refreshed.get("prompt_name")
    expected_output = refreshed.get("expected_output")
    if not isinstance(prompt_name, str) or not isinstance(expected_output, dict):
        return refreshed

    raw_prediction, parse_error = (
        parse_json_text(response_text) if response_text else (None, "Empty response text")
    )
    if parse_error:
        comparison = compare_outputs(prompt_name, {"_parse_error": parse_error}, expected_output)
    else:
        comparison = compare_outputs(prompt_name, raw_prediction, expected_output)

    refreshed["raw_prediction"] = raw_prediction
    refreshed["parse_error"] = parse_error
    refreshed["schema_valid"] = comparison["schema_valid"]
    refreshed["validation_errors"] = comparison["validation_errors"]
    refreshed["field_match_rate"] = comparison["field_match_rate"]
    refreshed["matched_field_count"] = comparison["matched_field_count"]
    refreshed["field_count"] = comparison["field_count"]
    refreshed["field_matches"] = comparison["field_matches"]
    refreshed["exact_match"] = comparison["exact_match"]
    refreshed["normalized_output"] = comparison["normalized_output"]
    return refreshed


def load_attempts(run_dir: Path) -> list[dict[str, Any]]:
    attempt_files = sorted(
        path
        for path in run_dir.glob("attempts/*/*/*/run-*.json")
        if ATTEMPT_RECORD_RE.fullmatch(path.name)
    )
    if not attempt_files:
        raise AnalysisError(
            f"No new-layout attempt JSON files found under {run_dir}/attempts/*/*/*/run-*.json"
        )
    attempts: list[dict[str, Any]] = []
    for path in attempt_files:
        attempt = read_json(path)
        if not isinstance(attempt, dict):
            raise AnalysisError(f"Attempt record must be a JSON object: {path}")
        attempts.append(refresh_attempt_from_response_text(path, attempt))
    return attempts


def classify_field_result(attempt: dict[str, Any], field_name: str, matched: bool) -> str:
    if matched:
        return "match"
    if attempt.get("parse_error"):
        return "parse_error"
    if not attempt.get("schema_valid"):
        return "schema_invalid"
    normalized_output = attempt.get("normalized_output")
    if isinstance(normalized_output, dict) and field_name in normalized_output:
        return "value_mismatch"
    return "missing_prediction"


def build_field_rows(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        expected_output = attempt.get("expected_output")
        if not isinstance(expected_output, dict):
            raise AnalysisError(f"Attempt missing expected_output object: {attempt.get('attempt_id')}")

        field_matches = attempt.get("field_matches") or {}
        normalized_output = attempt.get("normalized_output")
        raw_prediction = attempt.get("raw_prediction")

        for field_name, expected_value in expected_output.items():
            matched = bool(field_matches.get(field_name, False))
            predicted_value = None
            if isinstance(normalized_output, dict) and field_name in normalized_output:
                predicted_value = normalized_output[field_name]
            elif isinstance(raw_prediction, dict) and field_name in raw_prediction:
                predicted_value = raw_prediction[field_name]

            rows.append(
                {
                    "attempt_id": attempt["attempt_id"],
                    "provider": attempt["provider"],
                    "model": attempt["model"],
                    "case_id": attempt["case_id"],
                    "image_id": attempt["image_id"],
                    "answer_key": attempt["answer_key"],
                    "package_slug": attempt["package_slug"],
                    "variant_name": attempt["variant_name"],
                    "variant_slug": get_variant_slug(attempt),
                    "prompt_name": attempt["prompt_name"],
                    "prompt_variant": get_prompt_variant(attempt),
                    "repeat_index": attempt["repeat_index"],
                    "field_name": field_name,
                    "matched": matched,
                    "result_type": classify_field_result(attempt, field_name, matched),
                    "expected_value": canonical_value(expected_value),
                    "predicted_value": canonical_value(predicted_value),
                    "schema_valid": bool(attempt.get("schema_valid")),
                    "exact_match": bool(attempt.get("exact_match")),
                    "parse_error": attempt.get("parse_error") or "",
                    "validation_errors": "; ".join(attempt.get("validation_errors") or []),
                }
            )
    return rows


def build_height_nuisance_rows(
    attempts: list[dict[str, Any]],
    ground_truth_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        analysis = compute_height_nuisance_analysis(attempt, ground_truth_map)

        rows.append(
            {
                "attempt_id": attempt["attempt_id"],
                "provider": attempt["provider"],
                "model": attempt["model"],
                "case_id": attempt["case_id"],
                "image_id": attempt["image_id"],
                "answer_key": attempt["answer_key"],
                "package_slug": attempt["package_slug"],
                "variant_name": attempt["variant_name"],
                "variant_slug": get_variant_slug(attempt),
                "prompt_name": attempt["prompt_name"],
                "prompt_variant": get_prompt_variant(attempt),
                "repeat_index": attempt["repeat_index"],
                "expected_target": canonical_value(analysis.get("expected_target")),
                "predicted_value": canonical_value(analysis.get("predicted_value")),
                "nuisance_dimensions": canonical_value(analysis.get("nuisance_dimensions")),
                "matches_target": analysis.get("matches_target"),
                "matches_nuisance": analysis.get("matches_nuisance"),
                "result": analysis.get("result"),
                "schema_valid": bool(attempt.get("schema_valid")),
                "exact_match": bool(attempt.get("exact_match")),
                "parse_error": attempt.get("parse_error") or "",
                "validation_errors": "; ".join(attempt.get("validation_errors") or []),
            }
        )
    return rows


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    group_keys: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for group_values, group_rows in sorted(grouped.items()):
        summary = {key: value for key, value in zip(group_keys, group_values)}
        total = len(group_rows)
        match_count = sum(1 for row in group_rows if row["result_type"] == "match")
        value_mismatch_count = sum(1 for row in group_rows if row["result_type"] == "value_mismatch")
        parse_error_count = sum(1 for row in group_rows if row["result_type"] == "parse_error")
        schema_invalid_count = sum(1 for row in group_rows if row["result_type"] == "schema_invalid")
        missing_prediction_count = sum(1 for row in group_rows if row["result_type"] == "missing_prediction")
        summary_rows.append(
            {
                **summary,
                "attempt_count": total,
                "match_count": match_count,
                "match_rate": round(match_count / total, 4) if total else 0.0,
                "value_mismatch_count": value_mismatch_count,
                "parse_error_count": parse_error_count,
                "schema_invalid_count": schema_invalid_count,
                "missing_prediction_count": missing_prediction_count,
            }
        )
    return summary_rows


def summarize_height_nuisance_rows(
    rows: list[dict[str, Any]],
    *,
    group_keys: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for group_values, group_rows in sorted(grouped.items()):
        summary = {key: value for key, value in zip(group_keys, group_values)}
        applicable_rows = [
            row for row in group_rows if row["result"] not in {"not_applicable", "metadata_missing"}
        ]
        missed_rows = [row for row in applicable_rows if row["result"] != "target_match"]
        summary_rows.append(
            {
                **summary,
                "attempt_count": len(group_rows),
                "applicable_attempt_count": len(applicable_rows),
                "target_match_count": sum(1 for row in group_rows if row["result"] == "target_match"),
                "nuisance_match_count": sum(1 for row in group_rows if row["result"] == "nuisance_match"),
                "other_mismatch_count": sum(1 for row in group_rows if row["result"] == "other_mismatch"),
                "no_prediction_count": sum(1 for row in group_rows if row["result"] == "no_prediction"),
                "metadata_missing_count": sum(1 for row in group_rows if row["result"] == "metadata_missing"),
                "target_match_rate": round(
                    sum(1 for row in applicable_rows if row["result"] == "target_match") / len(applicable_rows),
                    4,
                ) if applicable_rows else 0.0,
                "nuisance_match_rate": round(
                    sum(1 for row in applicable_rows if row["result"] == "nuisance_match") / len(applicable_rows),
                    4,
                ) if applicable_rows else 0.0,
                "nuisance_match_rate_among_misses": round(
                    sum(1 for row in missed_rows if row["result"] == "nuisance_match") / len(missed_rows),
                    4,
                ) if missed_rows else 0.0,
            }
        )
    return summary_rows


def build_prompt_variant_delta_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_keys = ["provider", "model", "prompt_name", "field_name", "variant_slug"]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_keys)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for group_values, group_rows in sorted(grouped.items()):
        by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            by_variant[row["prompt_variant"]].append(row)

        baseline_rows = by_variant.get("baseline", [])
        warning_rows = by_variant.get("view_semantics_warning", [])
        if not baseline_rows and not warning_rows:
            continue

        baseline_match_count = sum(1 for row in baseline_rows if row["result_type"] == "match")
        warning_match_count = sum(1 for row in warning_rows if row["result_type"] == "match")
        baseline_attempt_count = len(baseline_rows)
        warning_attempt_count = len(warning_rows)
        baseline_match_rate = (
            baseline_match_count / baseline_attempt_count if baseline_attempt_count else None
        )
        warning_match_rate = (
            warning_match_count / warning_attempt_count if warning_attempt_count else None
        )
        delta = (
            warning_match_rate - baseline_match_rate
            if baseline_match_rate is not None and warning_match_rate is not None
            else None
        )

        summary_rows.append(
            {
                **{key: value for key, value in zip(group_keys, group_values)},
                "baseline_attempt_count": baseline_attempt_count,
                "baseline_match_count": baseline_match_count,
                "baseline_match_rate": round(baseline_match_rate, 4)
                if baseline_match_rate is not None
                else "",
                "view_semantics_warning_attempt_count": warning_attempt_count,
                "view_semantics_warning_match_count": warning_match_count,
                "view_semantics_warning_match_rate": round(warning_match_rate, 4)
                if warning_match_rate is not None
                else "",
                "match_rate_delta": round(delta, 4) if delta is not None else "",
            }
        )
    return summary_rows


def resolve_output_dir(run_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    return (DEFAULT_ANALYSIS_DIR / run_dir.name).resolve()


def analyze_run(
    run_dir: Path,
    output_dir: Path | None,
    ground_truth_path: Path,
    *,
    allow_api_errors: bool = False,
) -> Path:
    raise_if_run_has_api_errors(run_dir, allow_api_errors=allow_api_errors)
    attempts = load_attempts(run_dir)
    ground_truth_map = load_ground_truth_map(ground_truth_path)
    field_rows = build_field_rows(attempts)
    height_nuisance_rows = build_height_nuisance_rows(attempts, ground_truth_map)
    provider_field_summary = summarize_rows(
        field_rows,
        group_keys=["provider", "model", "prompt_variant", "prompt_name", "field_name"],
    )
    case_field_summary = summarize_rows(
        field_rows,
        group_keys=["provider", "case_id", "prompt_variant", "prompt_name", "field_name"],
    )
    provider_height_nuisance_summary = summarize_height_nuisance_rows(
        height_nuisance_rows,
        group_keys=["provider", "model", "prompt_variant", "variant_slug", "prompt_name"],
    )
    case_height_nuisance_summary = summarize_height_nuisance_rows(
        height_nuisance_rows,
        group_keys=["provider", "case_id", "prompt_variant", "variant_slug", "prompt_name"],
    )
    prompt_variant_delta_summary = build_prompt_variant_delta_summary(field_rows)

    resolved_output_dir = resolve_output_dir(run_dir, output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(resolved_output_dir / "attempt_field_results.csv", field_rows)
    write_csv(resolved_output_dir / "provider_field_summary.csv", provider_field_summary)
    write_csv(resolved_output_dir / "case_field_summary.csv", case_field_summary)
    write_csv(resolved_output_dir / "height_nuisance_results.csv", height_nuisance_rows)
    write_csv(
        resolved_output_dir / "provider_height_nuisance_summary.csv",
        provider_height_nuisance_summary,
    )
    write_csv(
        resolved_output_dir / "case_height_nuisance_summary.csv",
        case_height_nuisance_summary,
    )
    write_csv(resolved_output_dir / "prompt_variant_delta_summary.csv", prompt_variant_delta_summary)
    return resolved_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate field-level analysis tables from a benchmark run without modifying the run artifacts."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to a run directory under runs/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory. Default: analysis/<run-name>/",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_PATH,
        help="Ground truth file used to recompute nuisance analysis.",
    )
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
        output_dir = analyze_run(
            run_dir,
            args.output_dir,
            args.ground_truth.resolve(),
            allow_api_errors=args.allow_api_errors,
        )
    except (AnalysisError, ApiErrorGuardError) as exc:
        print(f"Analysis failed: {exc}")
        return 1
    print(f"Analysis outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
