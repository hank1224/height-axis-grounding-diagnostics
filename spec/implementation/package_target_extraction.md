# Package Target Extraction

## Purpose

`package_target_extraction` measures whether a model can extract package-level target dimensions from one package drawing.

The task supports two answer modes:

- `extract_number`: return numeric dimensions
- `extract_id`: return dimension-line IDs

The task also supports prompt variants:

- `baseline`
- `view_semantics_warning`
- `both`

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

Task assets live under `data/tasks/package_target_extraction/`.

Important files:

- `cases.json`
- `ground_truth.json`
- `ground_truth.template.json`
- `prompts/extract_number.md`
- `prompts/extract_id.md`
- `prompts/view_semantics_warning.md`

`extract_number` expects:

- `body_long_side`
- `body_short_side`
- `maximum_terminal_to_terminal_span`
- `overall_package_height`

Each value must be a JSON number or `null`.

`extract_id` expects:

- `body_side_dimensions`
- `maximum_terminal_to_terminal_span`
- `overall_package_height`

`body_side_dimensions` is `null` or a two-item ID array. The two IDs are sorted before matching.

Ground truth may also include `evaluation_metadata.nuisance_dimensions` for height-specific nuisance analysis.

## Workflow

Build cases:

```bash
./.venv/bin/python scripts/package_target_build_cases.py
```

Validate ground truth:

```bash
./.venv/bin/python scripts/package_target_validate_ground_truth.py
```

Run the benchmark:

```bash
./.venv/bin/python scripts/package_target_run_benchmark.py \
  --providers openai gemini anthropic \
  --repeats 1 \
  --prompt-variant both
```

Analyze a run:

```bash
./.venv/bin/python scripts/package_target_analyze_run.py \
  runs/package_target_extraction/<run-name>
```

## Primary Metrics

- `field_match_rate`: fraction of expected fields that match after normalization
- `exact_match`: all expected fields matched within one attempt
- `schema_valid`: response parsed and matched the task schema
- `height_nuisance_results`: height-specific classification into `target_match`, `nuisance_match`, `other_mismatch`, and `no_prediction`

Prompt-variant comparison is written as `prompt_variant_delta_summary.csv`.

## Artifacts

Run artifacts:

```text
runs/package_target_extraction/<run-name>/
```

Analysis artifacts:

```text
analysis/package_target_extraction/<run-name>/
  attempt_field_results.csv
  provider_field_summary.csv
  case_field_summary.csv
  height_nuisance_results.csv
  provider_height_nuisance_summary.csv
  case_height_nuisance_summary.csv
  prompt_variant_delta_summary.csv
```

## Notes

- The analyzer can reparse `run-001.response.txt` when it exists beside `run-001.json`.
- The benchmark does not rely on provider-native structured output enforcement.
- This file is implementation-only. Curated claim and result documents live under `spec/experiment_claims/` and `experiment_results/`.
