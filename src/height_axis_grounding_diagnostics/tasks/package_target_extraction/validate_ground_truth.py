#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import sys
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CASES_PATH = ROOT / "data" / "tasks" / "package_target_extraction" / "cases.json"
DEFAULT_GROUND_TRUTH_PATH = ROOT / "data" / "tasks" / "package_target_extraction" / "ground_truth.json"
VALID_STATUSES = {"pending", "complete"}
ID_PATTERN = re.compile(r"^ID\d+$")


class ValidationError(Exception):
    pass


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in {path}: {exc}") from exc


def expect_keys(obj: dict, expected: set[str], context: str) -> None:
    actual = set(obj.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        problems = []
        if missing:
            problems.append(f"missing={missing}")
        if extra:
            problems.append(f"extra={extra}")
        raise ValidationError(f"{context} has unexpected keys: {', '.join(problems)}")


def validate_number_ground_truth(ground_truth: dict, context: str) -> None:
    expected_keys = {
        "body_long_side",
        "body_short_side",
        "maximum_terminal_to_terminal_span",
        "overall_package_height",
    }
    expect_keys(ground_truth, expected_keys, context)
    for key, value in ground_truth.items():
        if value is not None and not isinstance(value, (int, float)):
            raise ValidationError(f"{context}.{key} must be a number or null")


def validate_id_value(value: object, context: str) -> None:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValidationError(f"{context} must be a string like ID7")


def validate_id_ground_truth(ground_truth: dict, context: str) -> None:
    expected_keys = {
        "body_side_dimensions",
        "maximum_terminal_to_terminal_span",
        "overall_package_height",
    }
    expect_keys(ground_truth, expected_keys, context)

    body_dims = ground_truth["body_side_dimensions"]
    if body_dims is not None:
        if not isinstance(body_dims, list) or len(body_dims) != 2:
            raise ValidationError(f"{context}.body_side_dimensions must be null or a 2-item array")
        for index, item in enumerate(body_dims):
            validate_id_value(item, f"{context}.body_side_dimensions[{index}]")

    for key in ("maximum_terminal_to_terminal_span", "overall_package_height"):
        value = ground_truth[key]
        if value is not None:
            validate_id_value(value, f"{context}.{key}")


def validate_number_or_number_list(value: object, context: str) -> None:
    if isinstance(value, bool):
        raise ValidationError(f"{context} must be null, a number, or an array of numbers")
    if isinstance(value, (int, float)):
        return
    if not isinstance(value, list):
        raise ValidationError(f"{context} must be null, a number, or an array of numbers")
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValidationError(f"{context}[{index}] must be a number")


def validate_id_or_id_list(value: object, context: str) -> None:
    if isinstance(value, str):
        validate_id_value(value, context)
        return
    if not isinstance(value, list):
        raise ValidationError(f"{context} must be null, an ID string, or an array of ID strings")
    for index, item in enumerate(value):
        validate_id_value(item, f"{context}[{index}]")


def validate_evaluation_metadata(
    evaluation_metadata: dict,
    *,
    prompt_name: str,
    rotated_image_ids: list[str],
    context: str,
) -> None:
    expected_keys = {"nuisance_dimensions"}
    expect_keys(evaluation_metadata, expected_keys, context)

    nuisance_dimensions = evaluation_metadata["nuisance_dimensions"]
    nuisance_context = f"{context}.nuisance_dimensions"
    if nuisance_dimensions is None:
        return
    if not rotated_image_ids:
        raise ValidationError(f"{nuisance_context} must be null when no rotated images exist")
    if prompt_name == "extract_number":
        validate_number_or_number_list(nuisance_dimensions, nuisance_context)
    elif prompt_name == "extract_id":
        validate_id_or_id_list(nuisance_dimensions, nuisance_context)
    else:
        raise ValidationError(f"{context} has unsupported prompt_name: {prompt_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate package target extraction ground truth."
    )
    parser.add_argument(
        "ground_truth",
        nargs="?",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_PATH,
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    ground_truth_path = args.ground_truth.resolve()
    cases_path = args.cases.resolve()
    cases_payload = load_json(cases_path)
    ground_truth = load_json(ground_truth_path)

    cases = cases_payload.get("cases")
    if not isinstance(cases, list):
        raise ValidationError("Cases file must contain a `cases` array")

    cases_by_case_id = {case["case_id"]: case for case in cases}
    cases_by_image_id = {case["image_id"]: case for case in cases}
    cases_by_answer_key: dict[str, list[dict]] = {}
    for case in cases:
        cases_by_answer_key.setdefault(case["answer_key"], []).append(case)

    answer_groups = ground_truth.get("answer_groups")

    if not isinstance(answer_groups, list):
        raise ValidationError("`answer_groups` must be a JSON array")
    if len(answer_groups) != len(cases_by_answer_key):
        raise ValidationError(
            f"Expected {len(cases_by_answer_key)} answer_groups, found {len(answer_groups)}"
        )

    seen_answer_keys: set[str] = set()
    covered_case_ids: set[str] = set()
    for index, answer_group in enumerate(answer_groups):
        context = f"answer_groups[{index}]"
        expected_keys = {
            "answer_key",
            "package_name",
            "package_slug",
            "prompt_name",
            "applies_to_case_ids",
            "applies_to_image_ids",
            "applies_to_variants",
            "annotation_status",
            "ground_truth",
            "evaluation_metadata",
            "notes",
        }
        if not isinstance(answer_group, dict):
            raise ValidationError(f"{context} must be an object")
        expect_keys(answer_group, expected_keys, context)

        answer_key = answer_group["answer_key"]
        if not isinstance(answer_key, str):
            raise ValidationError(f"{context}.answer_key must be a string")
        if answer_key in seen_answer_keys:
            raise ValidationError(f"Duplicate answer_key: {answer_key}")
        seen_answer_keys.add(answer_key)

        case_group = cases_by_answer_key.get(answer_key)
        if case_group is None:
            raise ValidationError(f"{context}.answer_key is not present in cases: {answer_key}")
        case_group = sorted(case_group, key=lambda item: item["image_id"])
        case_reference = case_group[0]

        for key in ("package_name", "package_slug", "prompt_name"):
            if answer_group[key] != case_reference[key]:
                raise ValidationError(
                    f"{context}.{key} does not match cases for answer_key={answer_key}"
                )

        expected_case_ids = [case["case_id"] for case in case_group]
        expected_image_ids = [case["image_id"] for case in case_group]
        expected_variants = [case["variant_name"] for case in case_group]
        actual_case_ids = answer_group["applies_to_case_ids"]
        actual_image_ids = answer_group["applies_to_image_ids"]
        actual_variants = answer_group["applies_to_variants"]

        if actual_case_ids != expected_case_ids:
            raise ValidationError(
                f"{context}.applies_to_case_ids does not match cases for answer_key={answer_key}"
            )
        if actual_image_ids != expected_image_ids:
            raise ValidationError(
                f"{context}.applies_to_image_ids does not match cases for answer_key={answer_key}"
            )
        if actual_variants != expected_variants:
            raise ValidationError(
                f"{context}.applies_to_variants does not match cases for answer_key={answer_key}"
            )

        rotated_image_ids = [
            case["image_id"]
            for case in case_group
            if str(case["variant_slug"]).startswith("rotated-")
        ]

        for case_id in actual_case_ids:
            if case_id in covered_case_ids:
                raise ValidationError(f"Case covered more than once: {case_id}")
            if case_id not in cases_by_case_id:
                raise ValidationError(f"{context}.applies_to_case_ids contains unknown case_id: {case_id}")
            covered_case_ids.add(case_id)

        for image_id in actual_image_ids:
            if image_id not in cases_by_image_id:
                raise ValidationError(f"{context}.applies_to_image_ids contains unknown image_id: {image_id}")

        status = answer_group["annotation_status"]
        if status not in VALID_STATUSES:
            raise ValidationError(f"{context}.annotation_status must be one of {sorted(VALID_STATUSES)}")

        if not isinstance(answer_group["notes"], str):
            raise ValidationError(f"{context}.notes must be a string")

        ground_truth_obj = answer_group["ground_truth"]
        if not isinstance(ground_truth_obj, dict):
            raise ValidationError(f"{context}.ground_truth must be an object")

        evaluation_metadata = answer_group["evaluation_metadata"]
        if not isinstance(evaluation_metadata, dict):
            raise ValidationError(f"{context}.evaluation_metadata must be an object")
        validate_evaluation_metadata(
            evaluation_metadata,
            prompt_name=answer_group["prompt_name"],
            rotated_image_ids=rotated_image_ids,
            context=f"{context}.evaluation_metadata",
        )

        if answer_group["prompt_name"] == "extract_number":
            validate_number_ground_truth(ground_truth_obj, f"{context}.ground_truth")
        elif answer_group["prompt_name"] == "extract_id":
            validate_id_ground_truth(ground_truth_obj, f"{context}.ground_truth")
        else:
            raise ValidationError(f"{context}.prompt_name is unsupported: {answer_group['prompt_name']}")

    if covered_case_ids != set(cases_by_case_id):
        missing = sorted(set(cases_by_case_id) - covered_case_ids)
        raise ValidationError(f"Some cases are not covered by any answer_group: {missing}")

    print(
        f"Validation passed for {len(answer_groups)} answer_groups covering "
        f"{len(covered_case_ids)} cases: {ground_truth_path.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
