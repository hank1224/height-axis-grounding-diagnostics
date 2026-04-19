from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any


TASK_NAME = "pure_ocr_extraction"
VALID_STATUSES = {"pending", "complete"}
VALID_ORIENTATIONS = {"horizontal", "vertical"}
SLOT_ORDER = ("upper_left", "upper_right", "lower_left", "lower_right")
SLOT_INDEX = {slot: index for index, slot in enumerate(SLOT_ORDER)}
OUTPUT_KEYS = {"layout", "views"}
VIEW_KEYS = {"slot", "bounding_box_2d", "dimensions"}
GROUND_TRUTH_VIEW_KEYS = {"slot", "dimensions"}
DIMENSION_KEYS = {
    "value",
    "orientation",
    "belongs_to_slot",
}


class OcrSchemaError(Exception):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def expect_keys(obj: dict[str, Any], expected: set[str], context: str, errors: list[str]) -> None:
    actual = set(obj.keys())
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        errors.append(f"{context} missing keys: {missing}")
    if extra:
        errors.append(f"{context} unexpected keys: {extra}")


def parse_slot(value: Any, context: str, errors: list[str]) -> str | None:
    if value not in SLOT_INDEX:
        errors.append(f"{context} must be one of {list(SLOT_ORDER)}")
        return None
    return value


def validate_layout(value: Any, context: str, errors: list[str]) -> dict[str, int] | None:
    if not isinstance(value, dict):
        errors.append(f"{context} must be an object")
        return None
    expect_keys(value, set(SLOT_ORDER), context, errors)
    if set(value) != set(SLOT_ORDER):
        return None

    normalized_layout: dict[str, int] = {}
    for slot in SLOT_ORDER:
        cell = value.get(slot)
        if isinstance(cell, bool) or cell not in {0, 1}:
            errors.append(f"{context}.{slot} must be 0 or 1")
        else:
            normalized_layout[slot] = cell
    occupied_count = sum(normalized_layout.values())
    if occupied_count != 3:
        errors.append(f"{context} must contain exactly three occupied slots")
    return normalized_layout if not errors else None


def occupied_slots_from_layout(layout: dict[str, int]) -> set[str]:
    occupied: set[str] = set()
    for slot, cell in layout.items():
        if cell == 1:
            occupied.add(slot)
    return occupied


def validate_bbox(value: Any, context: str, errors: list[str]) -> list[int | float] | None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        errors.append(f"{context} must be [ymin, xmin, ymax, xmax] with numeric values")
        return None

    ymin, xmin, ymax, xmax = value
    if any(item < 0 or item > 1000 for item in value):
        errors.append(f"{context} values must be normalized coordinates from 0 to 1000")
    if ymin >= ymax:
        errors.append(f"{context} must have ymin < ymax")
    if xmin >= xmax:
        errors.append(f"{context} must have xmin < xmax")
    return value


def strip_bboxes(output: dict[str, Any]) -> dict[str, Any]:
    stripped_views = []
    for view in output.get("views", []):
        stripped_views.append(
            {
                key: value
                for key, value in view.items()
                if key != "bounding_box_2d"
            }
        )
    return {
        "layout": output.get("layout"),
        "views": stripped_views,
    }


def validate_bbox_outputs(data: Any) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return False, ["prediction must be a JSON object"]
    views = data.get("views")
    if not isinstance(views, list):
        return False, ["prediction.views must be an array"]
    for view_index, view in enumerate(views):
        context = f"prediction.views[{view_index}].bounding_box_2d"
        if not isinstance(view, dict) or "bounding_box_2d" not in view:
            errors.append(f"{context} is required")
            continue
        validate_bbox(view.get("bounding_box_2d"), context, errors)
    return not errors, errors


def validate_dimension(
    value: Any,
    *,
    view_slot: str,
    context: str,
    errors: list[str],
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        errors.append(f"{context} must be an object")
        return None
    expect_keys(value, DIMENSION_KEYS, context, errors)

    dimension_value = value.get("value")
    if not isinstance(dimension_value, str) or not dimension_value:
        errors.append(f"{context}.value must be a non-empty string")

    orientation = value.get("orientation")
    if orientation not in VALID_ORIENTATIONS:
        errors.append(f"{context}.orientation must be one of {sorted(VALID_ORIENTATIONS)}")

    belongs_to_slot = parse_slot(value.get("belongs_to_slot"), f"{context}.belongs_to_slot", errors)
    if belongs_to_slot is not None and belongs_to_slot != view_slot:
        errors.append(f"{context}.belongs_to_slot must match the parent view slot")

    if errors:
        return None
    return {
        "value": dimension_value,
        "orientation": orientation,
        "belongs_to_slot": view_slot,
    }


def validate_ocr_output(
    data: Any,
    *,
    context: str = "output",
    require_bbox: bool = True,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return None, [f"{context} must be a JSON object"]

    expect_keys(data, OUTPUT_KEYS, context, errors)
    layout = validate_layout(data.get("layout"), f"{context}.layout", errors)
    views = data.get("views")
    if not isinstance(views, list):
        errors.append(f"{context}.views must be an array")
        return None, errors
    if layout is None:
        return None, errors

    occupied_slots = occupied_slots_from_layout(layout)
    normalized_views: list[dict[str, Any]] = []
    seen_slots: set[str] = set()

    for view_index, view in enumerate(views):
        view_context = f"{context}.views[{view_index}]"
        if not isinstance(view, dict):
            errors.append(f"{view_context} must be an object")
            continue
        if require_bbox:
            expect_keys(view, VIEW_KEYS, view_context, errors)
        else:
            expect_keys(view, GROUND_TRUTH_VIEW_KEYS, view_context, errors)

        slot = parse_slot(view.get("slot"), f"{view_context}.slot", errors)
        if slot is None:
            continue
        if slot not in occupied_slots:
            errors.append(f"{view_context}.slot is not marked occupied in layout")
        if slot in seen_slots:
            errors.append(f"{view_context}.slot appears more than once")
        seen_slots.add(slot)

        if require_bbox:
            validate_bbox(view.get("bounding_box_2d"), f"{view_context}.bounding_box_2d", errors)
        dimensions = view.get("dimensions")
        if not isinstance(dimensions, list):
            errors.append(f"{view_context}.dimensions must be an array")
            continue

        normalized_dimensions = []
        for dimension_index, dimension in enumerate(dimensions):
            normalized_dimension = validate_dimension(
                dimension,
                view_slot=slot,
                context=f"{view_context}.dimensions[{dimension_index}]",
                errors=errors,
            )
            if normalized_dimension is not None:
                normalized_dimensions.append(normalized_dimension)

        normalized_view = {
            "slot": slot,
            "dimensions": normalized_dimensions,
        }
        normalized_views.append(normalized_view)

    missing_views = occupied_slots - seen_slots
    extra_views = seen_slots - occupied_slots
    if missing_views:
        errors.append(f"{context}.views missing occupied slots: {sorted(missing_views, key=SLOT_INDEX.get)}")
    if extra_views:
        errors.append(f"{context}.views contains empty slots: {sorted(extra_views, key=SLOT_INDEX.get)}")
    if len(views) != len(occupied_slots):
        errors.append(f"{context}.views must contain exactly one object per occupied slot")

    if errors:
        return None, errors

    normalized_views.sort(key=lambda item: SLOT_INDEX[item["slot"]])
    for view in normalized_views:
        view["dimensions"].sort(
            key=lambda item: (
                item["value"],
                item["orientation"],
                item["belongs_to_slot"],
            )
        )
    return {
        "layout": layout,
        "views": normalized_views,
    }, []


def load_answer_map(ground_truth_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    answer_groups = ground_truth_payload.get("answer_groups")
    if not isinstance(answer_groups, list):
        raise OcrSchemaError("Ground truth file must contain an `answer_groups` array")
    mapping: dict[str, dict[str, Any]] = {}
    for index, group in enumerate(answer_groups):
        if not isinstance(group, dict):
            raise OcrSchemaError(f"answer_groups[{index}] must be an object")
        answer_key = group.get("answer_key")
        if not isinstance(answer_key, str) or not answer_key:
            raise OcrSchemaError(f"answer_groups[{index}].answer_key must be a non-empty string")
        if answer_key in mapping:
            raise OcrSchemaError(f"Duplicate answer_key: {answer_key}")
        mapping[answer_key] = group
    return mapping


def validate_ground_truth_payload(
    *,
    cases_payload: dict[str, Any],
    ground_truth_payload: dict[str, Any],
    require_complete: bool = False,
) -> list[str]:
    errors: list[str] = []
    cases = cases_payload.get("cases")
    answer_groups = ground_truth_payload.get("answer_groups")
    if not isinstance(cases, list):
        return ["cases file must contain a `cases` array"]
    if not isinstance(answer_groups, list):
        return ["ground truth file must contain an `answer_groups` array"]

    cases_by_answer_key: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            errors.append(f"cases[{index}] must be an object")
            continue
        answer_key = case.get("answer_key")
        if not isinstance(answer_key, str) or not answer_key:
            errors.append(f"cases[{index}].answer_key must be a non-empty string")
            continue
        cases_by_answer_key[answer_key] = case

    seen_answer_keys: set[str] = set()
    for index, group in enumerate(answer_groups):
        context = f"answer_groups[{index}]"
        if not isinstance(group, dict):
            errors.append(f"{context} must be an object")
            continue
        expected_keys = {
            "answer_key",
            "case_id",
            "image_id",
            "package_name",
            "package_slug",
            "variant_name",
            "variant_slug",
            "prompt_path",
            "image_path",
            "annotation_status",
            "ground_truth",
            "notes",
        }
        expect_keys(group, expected_keys, context, errors)

        answer_key = group.get("answer_key")
        if not isinstance(answer_key, str) or not answer_key:
            errors.append(f"{context}.answer_key must be a non-empty string")
            continue
        if answer_key in seen_answer_keys:
            errors.append(f"Duplicate answer_key: {answer_key}")
        seen_answer_keys.add(answer_key)

        case = cases_by_answer_key.get(answer_key)
        if case is None:
            errors.append(f"{context}.answer_key is not present in cases: {answer_key}")
            continue
        for key in ("case_id", "image_id", "package_name", "package_slug", "variant_name", "variant_slug", "prompt_path", "image_path"):
            if group.get(key) != case.get(key):
                errors.append(f"{context}.{key} does not match cases for answer_key={answer_key}")

        status = group.get("annotation_status")
        if status not in VALID_STATUSES:
            errors.append(f"{context}.annotation_status must be one of {sorted(VALID_STATUSES)}")
        elif require_complete and status != "complete":
            errors.append(f"{context}.annotation_status must be complete for benchmarking")

        notes = group.get("notes")
        if not isinstance(notes, str):
            errors.append(f"{context}.notes must be a string")

        ground_truth = group.get("ground_truth")
        if status == "pending" and ground_truth is None and not require_complete:
            continue
        if ground_truth is None:
            reason = "for benchmarking" if require_complete else "when annotation_status is complete"
            errors.append(f"{context}.ground_truth must be filled {reason}")
            continue
        _, output_errors = validate_ocr_output(
            ground_truth,
            context=f"{context}.ground_truth",
            require_bbox=False,
        )
        errors.extend(output_errors)

    missing = sorted(set(cases_by_answer_key) - seen_answer_keys)
    if missing:
        errors.append(f"Ground truth is missing answer groups for cases: {missing}")
    extra = sorted(seen_answer_keys - set(cases_by_answer_key))
    if extra:
        errors.append(f"Ground truth contains answer groups not present in cases: {extra}")

    return errors


def flatten_dimensions(output: dict[str, Any]) -> list[dict[str, Any]]:
    dimensions: list[dict[str, Any]] = []
    for view in output.get("views", []):
        for dimension in view.get("dimensions", []):
            dimensions.append(dimension)
    return dimensions


def counter_f1(expected_items: list[Any], predicted_items: list[Any]) -> dict[str, Any]:
    expected_counter = Counter(expected_items)
    predicted_counter = Counter(predicted_items)
    matched_count = sum((expected_counter & predicted_counter).values())
    expected_count = sum(expected_counter.values())
    predicted_count = sum(predicted_counter.values())
    precision = matched_count / predicted_count if predicted_count else (1.0 if expected_count == 0 else 0.0)
    recall = matched_count / expected_count if expected_count else (1.0 if predicted_count == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "matched_count": matched_count,
        "expected_count": expected_count,
        "predicted_count": predicted_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def view_map(output: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {view["slot"]: view for view in output.get("views", [])}


def compare_ocr_outputs(predicted: Any, expected: Any) -> dict[str, Any]:
    expected_norm, expected_errors = validate_ocr_output(
        expected,
        context="expected",
        require_bbox=False,
    )
    if expected_norm is None:
        raise OcrSchemaError(f"Ground truth failed validation: {expected_errors}")

    bbox_output_valid, bbox_errors = validate_bbox_outputs(predicted)
    full_output_norm, full_output_errors = validate_ocr_output(
        predicted,
        context="prediction",
        require_bbox=True,
    )
    predicted_for_ocr = strip_bboxes(predicted) if isinstance(predicted, dict) else predicted
    predicted_norm, predicted_errors = validate_ocr_output(
        predicted_for_ocr,
        context="prediction",
        require_bbox=False,
    )
    schema_valid = full_output_norm is not None
    validation_errors = full_output_errors if full_output_errors else predicted_errors
    if predicted_norm is None:
        expected_dims = flatten_dimensions(expected_norm)
        expected_slots = occupied_slots_from_layout(expected_norm["layout"])
        return {
            "schema_valid": schema_valid,
            "ocr_schema_valid": False,
            "validation_errors": validation_errors,
            "normalized_output": None,
            "layout_exact_match": False,
            "layout_slot_correct_count": 0,
            "layout_slot_total_count": len(SLOT_ORDER),
            "layout_slot_accuracy": 0.0,
            "occupied_slot_precision": 0.0,
            "occupied_slot_recall": 0.0,
            "occupied_slot_f1": 0.0,
            "occupied_slot_matched_count": 0,
            "expected_occupied_slot_count": len(expected_slots),
            "predicted_occupied_slot_count": 0,
            "bbox_output_valid": bbox_output_valid,
            "bbox_validation_errors": bbox_errors,
            "dimension_value_precision": 0.0,
            "dimension_value_recall": 0.0,
            "dimension_value_f1": 0.0,
            "dimension_value_matched_count": 0,
            "dimension_value_slot_matched_count": 0,
            "dimension_assignment_accuracy": 0.0,
            "orientation_accuracy": 0.0,
            "exact_match": False,
            "expected_dimension_count": len(expected_dims),
            "predicted_dimension_count": 0,
            "matched_dimension_count": 0,
            "dimension_full_matched_count": 0,
        }

    expected_slots = occupied_slots_from_layout(expected_norm["layout"])
    predicted_slots = occupied_slots_from_layout(predicted_norm["layout"])
    slot_scores = counter_f1(list(expected_slots), list(predicted_slots))

    expected_dims = flatten_dimensions(expected_norm)
    predicted_dims = flatten_dimensions(predicted_norm)
    expected_value_items = [item["value"] for item in expected_dims]
    predicted_value_items = [item["value"] for item in predicted_dims]
    value_scores = counter_f1(expected_value_items, predicted_value_items)

    expected_value_slot_items = [
        (item["value"], item["belongs_to_slot"]) for item in expected_dims
    ]
    predicted_value_slot_items = [
        (item["value"], item["belongs_to_slot"]) for item in predicted_dims
    ]
    value_slot_scores = counter_f1(expected_value_slot_items, predicted_value_slot_items)

    expected_full_items = [
        (item["value"], item["belongs_to_slot"], item["orientation"]) for item in expected_dims
    ]
    predicted_full_items = [
        (item["value"], item["belongs_to_slot"], item["orientation"]) for item in predicted_dims
    ]
    full_scores = counter_f1(expected_full_items, predicted_full_items)

    layout_exact_match = predicted_norm["layout"] == expected_norm["layout"]
    layout_slot_correct_count = sum(
        1
        for slot in SLOT_ORDER
        if predicted_norm["layout"].get(slot) == expected_norm["layout"].get(slot)
    )
    layout_slot_accuracy = layout_slot_correct_count / len(SLOT_ORDER)
    # Bbox is only a model-output validity signal. GT never contains bbox, and
    # predicted bbox is stripped before answer matching.
    exact_match = canonical_json(strip_bboxes(predicted_norm)) == canonical_json(strip_bboxes(expected_norm))
    assignment_accuracy = (
        value_slot_scores["matched_count"] / value_scores["matched_count"]
        if value_scores["matched_count"]
        else (1.0 if not expected_dims and not predicted_dims else 0.0)
    )
    orientation_accuracy = (
        full_scores["matched_count"] / value_slot_scores["matched_count"]
        if value_slot_scores["matched_count"]
        else (1.0 if not expected_dims and not predicted_dims else 0.0)
    )

    return {
        "schema_valid": schema_valid,
        "ocr_schema_valid": True,
        "validation_errors": validation_errors,
        "normalized_output": predicted_norm,
        "layout_exact_match": layout_exact_match,
        "layout_slot_correct_count": layout_slot_correct_count,
        "layout_slot_total_count": len(SLOT_ORDER),
        "layout_slot_accuracy": round(layout_slot_accuracy, 4),
        "occupied_slot_precision": slot_scores["precision"],
        "occupied_slot_recall": slot_scores["recall"],
        "occupied_slot_f1": slot_scores["f1"],
        "occupied_slot_matched_count": slot_scores["matched_count"],
        "expected_occupied_slot_count": slot_scores["expected_count"],
        "predicted_occupied_slot_count": slot_scores["predicted_count"],
        "bbox_output_valid": bbox_output_valid,
        "bbox_validation_errors": bbox_errors,
        "dimension_value_precision": value_scores["precision"],
        "dimension_value_recall": value_scores["recall"],
        "dimension_value_f1": value_scores["f1"],
        "dimension_value_matched_count": value_scores["matched_count"],
        "dimension_value_slot_matched_count": value_slot_scores["matched_count"],
        "dimension_assignment_accuracy": round(assignment_accuracy, 4),
        "orientation_accuracy": round(orientation_accuracy, 4),
        "exact_match": exact_match,
        "expected_dimension_count": len(expected_dims),
        "predicted_dimension_count": len(predicted_dims),
        "matched_dimension_count": full_scores["matched_count"],
        "dimension_full_matched_count": full_scores["matched_count"],
    }


def build_dimension_match_rows(attempt: dict[str, Any]) -> list[dict[str, Any]]:
    expected_output = attempt.get("expected_output")
    normalized_output = attempt.get("normalized_output")
    if not isinstance(expected_output, dict):
        return []
    expected_norm, _ = validate_ocr_output(expected_output, context="expected", require_bbox=False)
    predicted_norm = normalized_output if isinstance(normalized_output, dict) else None
    expected_dims = flatten_dimensions(expected_norm) if expected_norm else []
    predicted_dims = flatten_dimensions(predicted_norm) if predicted_norm else []

    predicted_by_value: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for dimension in predicted_dims:
        predicted_by_value[dimension["value"]].append(dimension)

    rows: list[dict[str, Any]] = []
    for expected_dimension in expected_dims:
        candidates = predicted_by_value.get(expected_dimension["value"], [])
        selected = None
        selected_index = None
        for index, candidate in enumerate(candidates):
            if (
                candidate["belongs_to_slot"] == expected_dimension["belongs_to_slot"]
                and candidate["orientation"] == expected_dimension["orientation"]
            ):
                selected = candidate
                selected_index = index
                break
        if selected is None and candidates:
            selected = candidates[0]
            selected_index = 0
        if selected_index is not None:
            candidates.pop(selected_index)

        if selected is None:
            match_level = "missing"
        elif selected["belongs_to_slot"] != expected_dimension["belongs_to_slot"]:
            match_level = "value_only"
        elif selected["orientation"] != expected_dimension["orientation"]:
            match_level = "value_and_slot"
        else:
            match_level = "full_match"

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
                "expected_value": expected_dimension["value"],
                "predicted_value": selected["value"] if selected else "",
                "expected_slot": expected_dimension["belongs_to_slot"],
                "predicted_slot": selected["belongs_to_slot"] if selected else "",
                "expected_orientation": expected_dimension["orientation"],
                "predicted_orientation": selected["orientation"] if selected else "",
                "match_level": match_level,
            }
        )

    for extra_candidates in predicted_by_value.values():
        for predicted_dimension in extra_candidates:
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
                    "expected_value": "",
                    "predicted_value": predicted_dimension["value"],
                    "expected_slot": "",
                    "predicted_slot": predicted_dimension["belongs_to_slot"],
                    "expected_orientation": "",
                    "predicted_orientation": predicted_dimension["orientation"],
                    "match_level": "extra_prediction",
                }
            )
    return rows
