# Height-Axis Grounding Diagnostics

Height-Axis Grounding Diagnostics studies how vision-language models interpret electronic package drawings under controlled diagnostic conditions. It separates task implementation specs, analyzer outputs, and a citation-ready single campaign result report so the benchmark can support both engineering work and thesis writing.

## At a Glance

- 15 package types
- 5 image variants per package
- 75 images for `package_target_extraction`, `top_view_localization`, `view_role_classification`, and `pcb_face_axis_mapping`
- 30 numeric-only images for `pure_ocr_extraction`
- 5 benchmark tasks


## Start Here

- If you want the overall documentation structure, start at [spec/README.md](spec/README.md).
- If you want the citation-ready single campaign result report, start at [experiment_results/2026-04-16_full_suite/README.md](experiment_results/2026-04-16_full_suite/README.md).
- If you want to understand which task answers which research question, start at [spec/experiment_claims/README.md](spec/experiment_claims/README.md).
- If you want implementation details for a task, start at [spec/README.md](spec/README.md) and then jump to the relevant file under `spec/implementation/`.

## Benchmark Tasks

- `package_target_extraction`: package-level target dimension extraction with value and ID prompts
- `pure_ocr_extraction`: OCR, layout, slot assignment, and orientation extraction without package-level inference
- `top_view_localization`: top-view slot detection
- `view_role_classification`: `top_view`, `side_view`, `end_view` slot classification
- `pcb_face_axis_mapping`: PCB mounting-face signed drawing-axis mapping

## Providers

The benchmark runners support `openai`, `gemini`, `anthropic`, and local `ollama`.
When `--providers` is omitted, all four providers are selected. For hosted providers,
copy `.env.example` to `.env` and fill in API keys as needed.

For Ollama, start the local service with `ollama serve`, pull a vision-capable model,
and set `OLLAMA_MODEL` or pass `--ollama-model`. The default Ollama API base URL is
`http://localhost:11434/api`, and the default Ollama concurrency is `1`.

## Repository Map

- `data/`: source exports, canonical image dataset, task assets, prompts, and ground truth
- `runs/`: raw benchmark runs
- `analysis/`: analyzer outputs derived from `runs/`
- `spec/`: implementation specs plus task-to-question capability maps
- `experiment_results/`: single campaign result report for direct citation; raw evidence remains in `runs/` and `analysis/`

## License

Except where otherwise noted, this repository is licensed under Creative
Commons Attribution-ShareAlike 4.0 International with the KiCad libraries
exception: `CC-BY-SA-4.0 WITH KiCad-libraries-exception`.

See [LICENSE.md](LICENSE.md). Third-party material remains subject to its
original license unless explicitly covered by this repository's license notice.

## Recommended Reading Paths

- Project overview and doc boundaries: [spec/README.md](spec/README.md)
- Task capability maps: [spec/experiment_claims/README.md](spec/experiment_claims/README.md)
- Task implementation specs:
  [package_target_extraction](spec/implementation/package_target_extraction.md),
  [pure_ocr_extraction](spec/implementation/pure_ocr_extraction.md),
  [top_view_localization](spec/implementation/top_view_localization.md),
  [view_role_classification](spec/implementation/view_role_classification.md),
  [pcb_mounting_face_axis_mapping](spec/implementation/pcb_mounting_face_axis_mapping.md)
