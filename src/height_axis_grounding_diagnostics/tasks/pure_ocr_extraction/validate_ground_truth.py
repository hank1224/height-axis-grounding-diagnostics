#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from height_axis_grounding_diagnostics.common.io_utils import read_json  # noqa: E402
from height_axis_grounding_diagnostics.tasks.pure_ocr_extraction.schema import validate_ground_truth_payload  # noqa: E402


DEFAULT_CASES_PATH = ROOT / "data" / "tasks" / "pure_ocr_extraction" / "cases.json"
DEFAULT_GROUND_TRUTH_PATH = ROOT / "data" / "tasks" / "pure_ocr_extraction" / "ground_truth.template.json"


def path_for_display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate pure OCR image-level ground truth."
    )
    parser.add_argument(
        "ground_truth",
        nargs="?",
        type=Path,
        default=DEFAULT_GROUND_TRUTH_PATH,
        help="Ground truth JSON to validate.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Pure OCR cases JSON.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require every answer group to contain valid ground_truth.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    cases_payload = read_json(args.cases)
    ground_truth_payload = read_json(args.ground_truth)
    errors = validate_ground_truth_payload(
        cases_payload=cases_payload,
        ground_truth_payload=ground_truth_payload,
        require_complete=args.require_complete,
    )
    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    answer_count = len(ground_truth_payload.get("answer_groups", []))
    mode = "complete" if args.require_complete else "template"
    print(
        f"Validation passed for {answer_count} pure OCR answer groups "
        f"({mode} mode): {path_for_display(args.ground_truth)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
