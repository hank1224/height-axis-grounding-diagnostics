# Spec Index

This folder contains two documentation layers with different responsibilities.

## 1. Implementation Specs

These files are the authoritative engineering specs for the five benchmark tasks.

- [implementation/package_target_extraction.md](implementation/package_target_extraction.md)
- [implementation/pure_ocr_extraction.md](implementation/pure_ocr_extraction.md)
- [implementation/top_view_localization.md](implementation/top_view_localization.md)
- [implementation/view_role_classification.md](implementation/view_role_classification.md)
- [implementation/pcb_mounting_face_axis_mapping.md](implementation/pcb_mounting_face_axis_mapping.md)

Implementation specs describe:

- task purpose
- dataset scope
- ground-truth schema and annotation rules
- build / validate / run / analyze workflow
- metric definitions and output artifacts

They do not contain benchmark conclusions.

## 2. Task-to-Question Capability Maps

These files record which research questions each task can answer, what scope is required, and which result section should be consulted in the campaign report.

- [experiment_claims/README.md](experiment_claims/README.md)
- [experiment_claims/package_target_extraction.md](experiment_claims/package_target_extraction.md)
- [experiment_claims/pure_ocr_extraction.md](experiment_claims/pure_ocr_extraction.md)
- [experiment_claims/top_view_localization.md](experiment_claims/top_view_localization.md)
- [experiment_claims/view_role_classification.md](experiment_claims/view_role_classification.md)
- [experiment_claims/pcb_face_axis_mapping.md](experiment_claims/pcb_face_axis_mapping.md)

Capability maps do not contain run-specific verdicts, result numbers, or citation-ready conclusions. The authoritative results live in the single campaign report at [../experiment_results/2026-04-16_full_suite/README.md](../experiment_results/2026-04-16_full_suite/README.md).
