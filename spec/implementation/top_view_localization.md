# Top View Localization

## Purpose

`top_view_localization` measures whether a model can:

- detect the drawing layout
- identify the occupied slots
- return the logical slot that corresponds to the package top view

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

Task assets live under `data/tasks/top_view_localization/`.

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
- `top_view_slot`

Ground truth excludes `bounding_box_2d`.

## Workflow

Build cases:

```bash
./.venv/bin/python scripts/top_view_build_cases.py
```

Validate ground truth:

```bash
./.venv/bin/python scripts/top_view_validate_ground_truth.py \
  data/tasks/top_view_localization/ground_truth.json \
  --require-complete
```

Run the benchmark:

```bash
./.venv/bin/python scripts/top_view_run_benchmark.py \
  --providers openai gemini anthropic \
  --repeats 1
```

Analyze a run:

```bash
./.venv/bin/python scripts/top_view_analyze_run.py \
  runs/top_view_localization/<run-name>
```

## Primary Metrics

- `schema_valid`
- `answer_schema_valid`
- `layout_exact_match`
- `layout_slot_accuracy`
- `occupied_slot_f1`
- `bbox_output_valid`
- `top_view_slot_match`
- `exact_match`

## Artifacts

Run artifacts:

```text
runs/top_view_localization/<run-name>/
```

Analysis artifacts:

```text
analysis/top_view_localization/<run-name>/
  attempt_top_view_results.csv
  provider_top_view_summary.csv
  case_top_view_summary.csv
  layout_results.csv
  top_view_results.csv
  bbox_results.csv
```

## Notes

- The analyzer reparses `run-001.response.txt` if an override file exists.
- `schema_valid` checks bbox; `answer_schema_valid` evaluates the answer after bbox removal.
- This file is implementation-only. Curated claim and result documents live under `spec/experiment_claims/` and `experiment_results/`.
