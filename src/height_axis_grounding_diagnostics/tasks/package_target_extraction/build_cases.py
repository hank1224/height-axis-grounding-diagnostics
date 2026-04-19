#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from height_axis_grounding_diagnostics.common.io_utils import read_json, write_json  # noqa: E402


TASK_NAME = "package_target_extraction"
DEFAULT_IMAGE_MANIFEST = ROOT / "data" / "package_drawings" / "image_manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "tasks" / TASK_NAME

PROMPT_SPECS = {
    "extract_number": {
        "prompt_path": "data/tasks/package_target_extraction/prompts/extract_number.md",
        "variant_slugs": {"canonical-values", "rotated-values"},
        "answer_schema": {
            "body_long_side": None,
            "body_short_side": None,
            "maximum_terminal_to_terminal_span": None,
            "overall_package_height": None,
        },
    },
    "extract_id": {
        "prompt_path": "data/tasks/package_target_extraction/prompts/extract_id.md",
        "variant_slugs": {"canonical-ids", "rotated-ids", "rotated-ids-seating-plane-anchor"},
        "answer_schema": {
            "body_side_dimensions": None,
            "maximum_terminal_to_terminal_span": None,
            "overall_package_height": None,
        },
    },
}


class BuildCasesError(Exception):
    pass


def prompt_name_for_variant(variant_slug: str) -> str:
    for prompt_name, spec in PROMPT_SPECS.items():
        if variant_slug in spec["variant_slugs"]:
            return prompt_name
    raise BuildCasesError(f"Unsupported variant_slug for package target extraction: {variant_slug}")


def load_images(image_manifest_path: Path) -> list[dict[str, Any]]:
    payload = read_json(image_manifest_path)
    images = payload.get("images")
    if not isinstance(images, list):
        raise BuildCasesError(f"Image manifest must contain an `images` array: {image_manifest_path}")
    return images


def build_cases(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for image in images:
        variant_slug = image["variant_slug"]
        prompt_name = prompt_name_for_variant(variant_slug)
        prompt_path = PROMPT_SPECS[prompt_name]["prompt_path"]
        answer_key = f"{image['package_slug']}__{prompt_name}"
        cases.append(
            {
                "case_id": f"{image['image_id']}__{TASK_NAME}",
                "image_id": image["image_id"],
                "answer_key": answer_key,
                "task_name": TASK_NAME,
                "package_name": image["package_name"],
                "package_slug": image["package_slug"],
                "kicad_model_name": image["kicad_model_name"],
                "shape_class": image.get("shape_class", ""),
                "variant_name": image["variant_name"],
                "variant_slug": variant_slug,
                "prompt_name": prompt_name,
                "prompt_path": prompt_path,
                "source_image_path": image["source_image_path"],
                "image_path": image["image_path"],
            }
        )
    cases.sort(key=lambda item: item["case_id"])
    return cases


def make_cases_payload(cases: list[dict[str, Any]], image_manifest_path: Path) -> dict[str, Any]:
    return {
        "task_name": TASK_NAME,
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_image_manifest": image_manifest_path.relative_to(ROOT).as_posix(),
        "case_count": len(cases),
        "cases": cases,
    }


def make_ground_truth_template(cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_cases: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped_cases.setdefault(case["answer_key"], []).append(case)

    answer_groups = []
    for answer_key in sorted(grouped_cases):
        group_cases = sorted(grouped_cases[answer_key], key=lambda item: item["image_id"])
        prompt_name = group_cases[0]["prompt_name"]
        rotated_image_ids = [
            case["image_id"]
            for case in group_cases
            if str(case["variant_slug"]).startswith("rotated-")
        ]
        answer_groups.append(
            {
                "answer_key": answer_key,
                "package_name": group_cases[0]["package_name"],
                "package_slug": group_cases[0]["package_slug"],
                "prompt_name": prompt_name,
                "applies_to_case_ids": [case["case_id"] for case in group_cases],
                "applies_to_image_ids": [case["image_id"] for case in group_cases],
                "applies_to_variants": [case["variant_name"] for case in group_cases],
                "annotation_status": "pending",
                "ground_truth": json.loads(json.dumps(PROMPT_SPECS[prompt_name]["answer_schema"])),
                "evaluation_metadata": {
                    "nuisance_dimensions": None if rotated_image_ids else None,
                },
                "notes": "",
            }
        )

    return {
        "task_name": TASK_NAME,
        "schema_version": 1,
        "instructions": [
            "Each entry represents one shared answer for the same package under the same prompt type.",
            "Fill only `ground_truth`, `evaluation_metadata`, `annotation_status`, and `notes`.",
            "For `extract_number`, use JSON numbers without units.",
            "For `extract_id`, use JSON strings like `ID7` and arrays like [`ID7`, `ID8`].",
            "Keep `null` when a value is truly not derivable from the drawing.",
            "Use `evaluation_metadata.nuisance_dimensions` only for rotated variants.",
            "Use `annotation_status: complete` when that shared answer is fully reviewed.",
        ],
        "answer_groups": answer_groups,
    }


def build(args: argparse.Namespace) -> None:
    images = load_images(args.image_manifest)
    cases = build_cases(images)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases_path = args.output_dir / "cases.json"
    template_path = args.output_dir / "ground_truth.template.json"
    write_json(cases_path, make_cases_payload(cases, args.image_manifest))

    if template_path.exists() and not args.force_template:
        print(f"Kept existing template: {template_path.relative_to(ROOT)}")
    else:
        write_json(template_path, make_ground_truth_template(cases))
        print(f"Wrote template: {template_path.relative_to(ROOT)}")

    print(f"Wrote {len(cases)} package target cases: {cases_path.relative_to(ROOT)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build cases and a ground truth template for package target extraction."
    )
    parser.add_argument("--image-manifest", type=Path, default=DEFAULT_IMAGE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force-template", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        build(args)
    except (BuildCasesError, OSError, ValueError) as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
