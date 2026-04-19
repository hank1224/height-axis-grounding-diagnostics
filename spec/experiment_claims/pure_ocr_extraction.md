# pure_ocr_extraction 可驗證問題

本文件是 task-to-question capability map。它只描述 `pure_ocr_extraction` 能回答哪些研究問題，以及閱讀結果時要看哪些 metrics，不記錄任何 run-specific 結論。

## 08 Pure OCR Capability

- `question_id`: `08_pure_ocr_capability`
- `question_title`: `Pure OCR Capability`
- `can_validate`: 如果把任務限制在 OCR、layout 與 dimension value extraction，而不要求 package-level 推理，模型的基礎辨識能力落在哪個水準。
- `required_scope`: `provider_rollup=all_cases`
- `result_entry`: [../../experiment_results/2026-04-16_full_suite/README.md#sanity-checks](../../experiment_results/2026-04-16_full_suite/README.md#sanity-checks)
- `result_metrics`: `answer_micro_dimension_value_accuracy` and `answer_micro_layout_slot_accuracy`
