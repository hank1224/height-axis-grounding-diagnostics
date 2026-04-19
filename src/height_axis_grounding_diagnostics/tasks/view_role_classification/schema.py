from __future__ import annotations

import json
from typing import Any


TASK_NAME = "view_role_classification"
VALID_STATUSES = {"pending", "complete"}
SLOT_ORDER = ("upper_left", "upper_right", "lower_left", "lower_right")
SLOT_INDEX = {slot: index for index, slot in enumerate(SLOT_ORDER)}
VIEW_ROLE_ORDER = ("top_view", "side_view", "end_view")
VIEW_ROLE_INDEX = {role: index for index, role in enumerate(VIEW_ROLE_ORDER)}
OUTPUT_KEYS = {"layout", "views"}
VIEW_KEYS = {"slot", "bounding_box_2d", "view_role"}
GROUND_TRUTH_VIEW_KEYS = {"slot", "view_role"}


class ViewRoleSchemaError(Exception):
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


def parse_view_role(value: Any, context: str, errors: list[str]) -> str | None:
    if value not in VIEW_ROLE_INDEX:
        errors.append(f"{context} must be one of {list(VIEW_ROLE_ORDER)}")
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
    if sum(normalized_layout.values()) != 3:
        errors.append(f"{context} must contain exactly three occupied slots")
    return normalized_layout if not errors else None


def occupied_slots_from_layout(layout: dict[str, int]) -> set[str]:
    return {slot for slot, cell in layout.items() if cell == 1}


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
        if isinstance(view, dict):
            stripped_views.append({key: value for key, value in view.items() if key != "bounding_box_2d"})
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


def validate_view_role_output(
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
    seen_roles: set[str] = set()

    for view_index, view in enumerate(views):
        view_context = f"{context}.views[{view_index}]"
        if not isinstance(view, dict):
            errors.append(f"{view_context} must be an object")
            continue
        expect_keys(view, VIEW_KEYS if require_bbox else GROUND_TRUTH_VIEW_KEYS, view_context, errors)

        slot = parse_slot(view.get("slot"), f"{view_context}.slot", errors)
        view_role = parse_view_role(view.get("view_role"), f"{view_context}.view_role", errors)
        if slot is None or view_role is None:
            continue
        if slot not in occupied_slots:
            errors.append(f"{view_context}.slot is not marked occupied in layout")
        if slot in seen_slots:
            errors.append(f"{view_context}.slot appears more than once")
        seen_slots.add(slot)
        if view_role in seen_roles:
            errors.append(f"{view_context}.view_role appears more than once")
        seen_roles.add(view_role)

        if require_bbox:
            validate_bbox(view.get("bounding_box_2d"), f"{view_context}.bounding_box_2d", errors)
        normalized_views.append({"slot": slot, "view_role": view_role})

    missing_views = occupied_slots - seen_slots
    extra_views = seen_slots - occupied_slots
    if missing_views:
        errors.append(f"{context}.views missing occupied slots: {sorted(missing_views, key=SLOT_INDEX.get)}")
    if extra_views:
        errors.append(f"{context}.views contains empty slots: {sorted(extra_views, key=SLOT_INDEX.get)}")
    if len(views) != len(occupied_slots):
        errors.append(f"{context}.views must contain exactly one object per occupied slot")

    missing_roles = set(VIEW_ROLE_ORDER) - seen_roles
    extra_roles = seen_roles - set(VIEW_ROLE_ORDER)
    if missing_roles:
        errors.append(f"{context}.views missing view roles: {sorted(missing_roles, key=VIEW_ROLE_INDEX.get)}")
    if extra_roles:
        errors.append(f"{context}.views contains invalid extra roles: {sorted(extra_roles)}")

    if errors:
        return None, errors

    normalized_views.sort(key=lambda item: SLOT_INDEX[item["slot"]])
    return {
        "layout": layout,
        "views": normalized_views,
    }, []


def load_answer_map(ground_truth_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    answer_groups = ground_truth_payload.get("answer_groups")
    if not isinstance(answer_groups, list):
        raise ViewRoleSchemaError("Ground truth file must contain an `answer_groups` array")
    mapping: dict[str, dict[str, Any]] = {}
    for index, group in enumerate(answer_groups):
        if not isinstance(group, dict):
            raise ViewRoleSchemaError(f"answer_groups[{index}] must be an object")
        answer_key = group.get("answer_key")
        if not isinstance(answer_key, str) or not answer_key:
            raise ViewRoleSchemaError(f"answer_groups[{index}].answer_key must be a non-empty string")
        if answer_key in mapping:
            raise ViewRoleSchemaError(f"Duplicate answer_key: {answer_key}")
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
        for key in (
            "case_id",
            "image_id",
            "package_name",
            "package_slug",
            "variant_name",
            "variant_slug",
            "prompt_path",
            "image_path",
        ):
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
        _, output_errors = validate_view_role_output(
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


def counter_f1(expected_items: list[Any], predicted_items: list[Any]) -> dict[str, Any]:
    expected = set(expected_items)
    predicted = set(predicted_items)
    matched_count = len(expected & predicted)
    expected_count = len(expected)
    predicted_count = len(predicted)
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


def view_role_map(output: dict[str, Any]) -> dict[str, str]:
    return {view["slot"]: view["view_role"] for view in output.get("views", [])}


def slot_for_role(output: dict[str, Any], role: str) -> str | None:
    for view in output.get("views", []):
        if view.get("view_role") == role:
            return view.get("slot")
    return None


def get_raw_role_items(data: Any) -> list[tuple[Any, Any]]:
    if not isinstance(data, dict):
        return []
    views = data.get("views")
    if not isinstance(views, list):
        return []
    items = []
    for view in views:
        if isinstance(view, dict):
            items.append((view.get("slot"), view.get("view_role")))
    return items


def compare_view_role_outputs(predicted: Any, expected: Any) -> dict[str, Any]:
    expected_norm, expected_errors = validate_view_role_output(
        expected,
        context="expected",
        require_bbox=False,
    )
    if expected_norm is None:
        raise ViewRoleSchemaError(f"Ground truth failed validation: {expected_errors}")

    bbox_output_valid, bbox_errors = validate_bbox_outputs(predicted)
    full_output_norm, full_output_errors = validate_view_role_output(
        predicted,
        context="prediction",
        require_bbox=True,
    )
    predicted_for_answer = strip_bboxes(predicted) if isinstance(predicted, dict) else predicted
    predicted_norm, predicted_errors = validate_view_role_output(
        predicted_for_answer,
        context="prediction",
        require_bbox=False,
    )
    schema_valid = full_output_norm is not None
    answer_schema_valid = predicted_norm is not None
    validation_errors = full_output_errors if full_output_errors else predicted_errors

    expected_slots = occupied_slots_from_layout(expected_norm["layout"])
    predicted_slots = (
        occupied_slots_from_layout(predicted_norm["layout"])
        if predicted_norm is not None
        else set()
    )
    slot_scores = counter_f1(list(expected_slots), list(predicted_slots))
    expected_role_items = [(view["slot"], view["view_role"]) for view in expected_norm["views"]]

    if predicted_norm is None:
        predicted_role_items = get_raw_role_items(predicted)
        role_scores = counter_f1(expected_role_items, predicted_role_items)
        return {
            "schema_valid": schema_valid,
            "answer_schema_valid": False,
            "validation_errors": validation_errors,
            "normalized_output": None,
            "layout_exact_match": False,
            "layout_slot_correct_count": 0,
            "layout_slot_total_count": len(SLOT_ORDER),
            "layout_slot_accuracy": 0.0,
            "occupied_slot_precision": slot_scores["precision"],
            "occupied_slot_recall": slot_scores["recall"],
            "occupied_slot_f1": slot_scores["f1"],
            "occupied_slot_matched_count": slot_scores["matched_count"],
            "expected_occupied_slot_count": slot_scores["expected_count"],
            "predicted_occupied_slot_count": slot_scores["predicted_count"],
            "bbox_output_valid": bbox_output_valid,
            "bbox_validation_errors": bbox_errors,
            "role_assignment_correct_count": role_scores["matched_count"],
            "role_assignment_total_count": len(expected_role_items),
            "role_assignment_accuracy": 0.0,
            "role_assignment_precision": role_scores["precision"],
            "role_assignment_recall": role_scores["recall"],
            "role_assignment_f1": role_scores["f1"],
            "view_role_exact_match": False,
            "expected_top_view_slot": slot_for_role(expected_norm, "top_view"),
            "predicted_top_view_slot": None,
            "top_view_slot_match": False,
            "expected_side_view_slot": slot_for_role(expected_norm, "side_view"),
            "predicted_side_view_slot": None,
            "side_view_slot_match": False,
            "expected_end_view_slot": slot_for_role(expected_norm, "end_view"),
            "predicted_end_view_slot": None,
            "end_view_slot_match": False,
            "exact_match": False,
        }

    predicted_role_items = [(view["slot"], view["view_role"]) for view in predicted_norm["views"]]
    role_scores = counter_f1(expected_role_items, predicted_role_items)
    layout_exact_match = predicted_norm["layout"] == expected_norm["layout"]
    layout_slot_correct_count = sum(
        1 for slot in SLOT_ORDER if predicted_norm["layout"].get(slot) == expected_norm["layout"].get(slot)
    )
    layout_slot_accuracy = layout_slot_correct_count / len(SLOT_ORDER)
    role_assignment_total = len(expected_role_items)
    role_assignment_correct = role_scores["matched_count"]
    role_assignment_accuracy = role_assignment_correct / role_assignment_total if role_assignment_total else 1.0
    view_role_exact_match = role_assignment_correct == role_assignment_total
    exact_match = canonical_json(strip_bboxes(predicted_norm)) == canonical_json(strip_bboxes(expected_norm))

    expected_top_view_slot = slot_for_role(expected_norm, "top_view")
    predicted_top_view_slot = slot_for_role(predicted_norm, "top_view")
    expected_side_view_slot = slot_for_role(expected_norm, "side_view")
    predicted_side_view_slot = slot_for_role(predicted_norm, "side_view")
    expected_end_view_slot = slot_for_role(expected_norm, "end_view")
    predicted_end_view_slot = slot_for_role(predicted_norm, "end_view")
    return {
        "schema_valid": schema_valid,
        "answer_schema_valid": answer_schema_valid,
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
        "role_assignment_correct_count": role_assignment_correct,
        "role_assignment_total_count": role_assignment_total,
        "role_assignment_accuracy": round(role_assignment_accuracy, 4),
        "role_assignment_precision": role_scores["precision"],
        "role_assignment_recall": role_scores["recall"],
        "role_assignment_f1": role_scores["f1"],
        "view_role_exact_match": view_role_exact_match,
        "expected_top_view_slot": expected_top_view_slot,
        "predicted_top_view_slot": predicted_top_view_slot,
        "top_view_slot_match": expected_top_view_slot == predicted_top_view_slot,
        "expected_side_view_slot": expected_side_view_slot,
        "predicted_side_view_slot": predicted_side_view_slot,
        "side_view_slot_match": expected_side_view_slot == predicted_side_view_slot,
        "expected_end_view_slot": expected_end_view_slot,
        "predicted_end_view_slot": predicted_end_view_slot,
        "end_view_slot_match": expected_end_view_slot == predicted_end_view_slot,
        "exact_match": exact_match,
    }
