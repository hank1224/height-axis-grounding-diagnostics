#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from height_axis_grounding_diagnostics.common.io_utils import read_json, write_json  # noqa: E402


TASK_NAME = "view_role_classification"
DEFAULT_SOURCE_MANIFEST = ROOT / "data" / "package_drawings" / "image_manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "tasks" / TASK_NAME
DEFAULT_PROMPT_PATH = "data/tasks/view_role_classification/prompt.md"
DEFAULT_VARIANT_SLUGS = (
    "canonical-values",
    "canonical-ids",
    "rotated-values",
    "rotated-ids",
    "rotated-ids-seating-plane-anchor",
)


class BuildCasesError(Exception):
    pass


def load_source_images(manifest_path: Path) -> list[dict[str, Any]]:
    payload = read_json(manifest_path)
    images = payload.get("images")
    if not isinstance(images, list):
        raise BuildCasesError(f"Image manifest must contain an `images` array: {manifest_path}")
    return images


def build_cases(
    images: list[dict[str, Any]],
    *,
    variant_slugs: set[str],
    prompt_path: str,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for image in images:
        variant_slug = image.get("variant_slug")
        if variant_slug not in variant_slugs:
            continue

        case_id = f"{image['image_id']}__{TASK_NAME}"
        cases.append(
            {
                "case_id": case_id,
                "image_id": image["image_id"],
                "answer_key": case_id,
                "task_name": TASK_NAME,
                "package_name": image["package_name"],
                "package_slug": image["package_slug"],
                "kicad_model_name": image["kicad_model_name"],
                "shape_class": image.get("shape_class", ""),
                "variant_name": image["variant_name"],
                "variant_slug": variant_slug,
                "prompt_path": prompt_path,
                "source_image_path": image["source_image_path"],
                "image_path": image["image_path"],
            }
        )

    cases.sort(key=lambda item: item["case_id"])
    if not cases:
        raise BuildCasesError("No images matched the requested variant slugs")
    return cases


def make_cases_payload(cases: list[dict[str, Any]], manifest_path: Path) -> dict[str, Any]:
    return {
        "task_name": TASK_NAME,
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_image_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "case_count": len(cases),
        "cases": cases,
    }


def make_ground_truth_template(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_name": TASK_NAME,
        "schema_version": 1,
        "instructions": [
            "Each answer_group is image-level and applies to exactly one case.",
            "Fill `ground_truth` with the layout and view-role answer schema, excluding `bounding_box_2d`.",
            "Do not include `bounding_box_2d` in ground_truth.",
            "Model outputs must include `bounding_box_2d`; predicted bbox is checked only for presence and logical normalized 0-1000 coordinates.",
            "Use only logical slot enum values: upper_left, upper_right, lower_left, lower_right.",
            "Exactly three slots must be occupied and one slot must be empty.",
            "Assign exactly one view to each role: top_view, side_view, and end_view.",
            "Each ground_truth view must contain `slot` and `view_role`.",
            "Keep `ground_truth` as null while annotation_status is pending.",
            "Set annotation_status to complete after the layout and view-role answer is fully reviewed.",
        ],
        "answer_groups": [
            {
                "answer_key": case["answer_key"],
                "case_id": case["case_id"],
                "image_id": case["image_id"],
                "package_name": case["package_name"],
                "package_slug": case["package_slug"],
                "variant_name": case["variant_name"],
                "variant_slug": case["variant_slug"],
                "prompt_path": case["prompt_path"],
                "image_path": case["image_path"],
                "annotation_status": "pending",
                "ground_truth": None,
                "notes": "",
            }
            for case in cases
        ],
    }


def build(args: argparse.Namespace) -> None:
    images = load_source_images(args.image_manifest)
    variant_slugs = set(args.variant_slug or DEFAULT_VARIANT_SLUGS)
    cases = build_cases(
        images,
        variant_slugs=variant_slugs,
        prompt_path=args.prompt_path,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases_path = args.output_dir / "cases.json"
    template_path = args.output_dir / "ground_truth.template.json"
    write_json(cases_path, make_cases_payload(cases, args.image_manifest))

    if template_path.exists() and not args.force_template:
        print(f"Kept existing template: {template_path.relative_to(ROOT)}")
    else:
        write_json(template_path, make_ground_truth_template(cases))
        print(f"Wrote template: {template_path.relative_to(ROOT)}")

    print(f"Wrote {len(cases)} view role classification cases: {cases_path.relative_to(ROOT)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build image-level cases and a ground truth template for view role classification."
    )
    parser.add_argument("--image-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt-path", default=DEFAULT_PROMPT_PATH)
    parser.add_argument(
        "--variant-slug",
        action="append",
        default=None,
        help="Variant slug to include. Defaults to all five package drawing variants.",
    )
    parser.add_argument(
        "--force-template",
        action="store_true",
        help="Overwrite ground_truth.template.json if it already exists.",
    )
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
