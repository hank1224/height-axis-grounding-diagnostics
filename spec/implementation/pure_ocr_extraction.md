# Pure OCR Extraction

## Purpose

`pure_ocr_extraction` isolates OCR and layout understanding from package-level target-dimension inference.

The task asks the model to extract:

- drawing layout occupancy
- detected views and their slots
- horizontal and vertical dimension values
- dimension-to-slot assignment
- dimension orientation

## Dataset Scope

- 15 package types
- 2 numeric variants per package
- 30 cases total

Included variants:

- `canonical-values`
- `rotated-values`

ID-only drawings are excluded because the task evaluates numeric-dimension extraction.

## Ground Truth

Task assets live under `data/tasks/pure_ocr_extraction/`.

Important files:

- `cases.json`
- `ground_truth.json`
- `ground_truth.template.json`
- `prompt.md`

Ground truth intentionally excludes `bounding_box_2d`.

Predicted bbox is checked only for output validity:

- coordinates must be normalized to `0-1000`
- format must be `[ymin, xmin, ymax, xmax]`
- bbox is never matched against GT coordinates

## Workflow

Build cases:

```bash
./.venv/bin/python scripts/pure_ocr_build_cases.py
```

Validate ground truth:

```bash
./.venv/bin/python scripts/pure_ocr_validate_ground_truth.py \
  data/tasks/pure_ocr_extraction/ground_truth.json \
  --require-complete
```

Run the benchmark:

```bash
./.venv/bin/python scripts/pure_ocr_run_benchmark.py \
  --providers openai gemini anthropic \
  --ground-truth data/tasks/pure_ocr_extraction/ground_truth.json \
  --repeats 1
```

Analyze a run:

```bash
./.venv/bin/python scripts/pure_ocr_analyze_run.py \
  runs/pure_ocr_extraction/<run-name>
```

## Primary Metrics

- `layout_slot_accuracy`
- `occupied_slot_precision`, `occupied_slot_recall`, `occupied_slot_f1`
- `dimension_value_precision`, `dimension_value_recall`, `dimension_value_f1`
- `answer_micro_dimension_value_accuracy`
- `dimension_assignment_accuracy`
- `orientation_accuracy`
- `bbox_output_valid`
- `exact_match`

## Artifacts

Run artifacts:

```text
runs/pure_ocr_extraction/<run-name>/
```

Analysis artifacts:

```text
analysis/pure_ocr_extraction/<run-name>/
  attempt_ocr_results.csv
  provider_ocr_summary.csv
  case_ocr_summary.csv
  dimension_match_details.csv
  layout_results.csv
  bbox_results.csv
```

## Notes

- The analyzer can recompute metrics from `run-001.response.txt` when that override file exists.
- `schema_valid` includes bbox validation; `answer_schema_valid` strips bbox first.
- This file is implementation-only. Curated claim and result documents live under `spec/experiment_claims/` and `experiment_results/`.
