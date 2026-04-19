# PCB Mounting Face Axis Mapping

## Purpose

This document covers the benchmark task implemented as `pcb_face_axis_mapping`.

The task measures whether a model can:

- detect the occupied drawing views
- keep the layout consistent
- assign the signed drawing-space axis that points from the package body toward the PCB mounting or connection face

Allowed axis labels:

- `+X`
- `-X`
- `+Y`
- `-Y`
- `+Z`
- `-Z`

Predictions may return `null` for `views[].pcb_mounting_face_axis` when the axis cannot be determined from the image. Ground truth must still use one of the six signed axis labels above.

## Dataset Scope

- 15 package types
- 5 image variants per package
- 75 cases total

Included variants:

- `canonical-values`
- `canonical-ids`
- `rotated-values`
- `rotated-ids`
- `rotated-ids-seating-plane-anchor`

## Ground Truth

Task assets live under `data/tasks/pcb_face_axis_mapping/`.

Important files:

- `cases.json`
- `ground_truth.json`
- `ground_truth.template.json`
- `prompt.md`

Expected answer shape:

- `layout`
- `views`
- `views[].slot`
- `views[].bounding_box_2d`
- `views[].pcb_mounting_face_axis`

Ground truth excludes `bounding_box_2d`.

Drawing-space axis convention:

- right on the page is `+X`
- left on the page is `-X`
- up on the page is `+Y`
- down on the page is `-Y`
- out of the page toward the viewer is `+Z`
- into the page is `-Z`

## Workflow

Build cases:

```bash
./.venv/bin/python scripts/pcb_face_axis_build_cases.py
```

Validate ground truth:

```bash
./.venv/bin/python scripts/pcb_face_axis_validate_ground_truth.py \
  data/tasks/pcb_face_axis_mapping/ground_truth.json \
  --require-complete
```

Run the benchmark:

```bash
./.venv/bin/python scripts/pcb_face_axis_run_benchmark.py \
  --providers openai gemini anthropic \
  --repeats 1
```

Analyze a run:

```bash
./.venv/bin/python scripts/pcb_face_axis_analyze_run.py \
  runs/pcb_face_axis_mapping/<run-name>
```

## Primary Metrics

- `schema_valid`
- `answer_schema_valid`
- `layout_exact_match`
- `layout_slot_accuracy`
- `occupied_slot_f1`
- `bbox_output_valid`
- `axis_accuracy`
- `axis_precision`
- `axis_recall`
- `axis_f1`
- `axis_exact_match`
- `exact_match`

## Artifacts

Run artifacts:

```text
runs/pcb_face_axis_mapping/<run-name>/
```

Analysis artifacts:

```text
analysis/pcb_face_axis_mapping/<run-name>/
  attempt_axis_results.csv
  provider_axis_summary.csv
  variant_axis_summary.csv
  shape_class_axis_summary.csv
  case_axis_summary.csv
  layout_results.csv
  pcb_mounting_face_axis_results.csv
  bbox_results.csv
```

## Notes

- Predicted `null` axis values are schema-valid abstentions. They are scored as missing/no-answer for axis matching and do not count as correct for `axis_exact_match` or `exact_match`.
- `variant_axis_summary.csv` compares canonical, rotated, and seating-plane-anchor conditions.
- `shape_class_axis_summary.csv` compares `sot_like_smd`, `two_terminal_diode_smd`, and `tabbed_power_smd`.
- This file is implementation-only. Curated claim and result documents live under `spec/experiment_claims/` and `experiment_results/`.
