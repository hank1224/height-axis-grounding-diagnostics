# View Role Classification

## Purpose

`view_role_classification` measures whether a model can assign the three occupied drawing views to:

- `top_view`
- `side_view`
- `end_view`

The task still evaluates layout detection and slot occupancy, but the main answer surface is role assignment across views.

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

Task assets live under `data/tasks/view_role_classification/`.

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
- `views[].view_role`

Ground truth excludes `bounding_box_2d`.

## Workflow

Build cases:

```bash
./.venv/bin/python scripts/view_role_build_cases.py
```

Validate ground truth:

```bash
./.venv/bin/python scripts/view_role_validate_ground_truth.py \
  data/tasks/view_role_classification/ground_truth.json \
  --require-complete
```

Run the benchmark:

```bash
./.venv/bin/python scripts/view_role_run_benchmark.py \
  --providers openai gemini anthropic \
  --repeats 1
```

Analyze a run:

```bash
./.venv/bin/python scripts/view_role_analyze_run.py \
  runs/view_role_classification/<run-name>
```

## Primary Metrics

- `schema_valid`
- `answer_schema_valid`
- `layout_exact_match`
- `layout_slot_accuracy`
- `occupied_slot_f1`
- `bbox_output_valid`
- `role_assignment_accuracy`
- `role_assignment_f1`
- `view_role_exact_match`
- `top_view_slot_match`
- `side_view_slot_match`
- `end_view_slot_match`
- `exact_match`

## Artifacts

Run artifacts:

```text
runs/view_role_classification/<run-name>/
```

Analysis artifacts:

```text
analysis/view_role_classification/<run-name>/
  attempt_view_role_results.csv
  provider_view_role_summary.csv
  case_view_role_summary.csv
  layout_results.csv
  view_role_results.csv
  bbox_results.csv
```

## Notes

- Full `schema_valid` includes bbox checks and one-role-each validation.
- `view_role_exact_match` means the full `(slot, view_role)` assignment matches GT.
- This file is implementation-only. Curated claim and result documents live under `spec/experiment_claims/` and `experiment_results/`.
