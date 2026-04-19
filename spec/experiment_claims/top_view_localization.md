# top_view_localization 可驗證問題

本文件是 task-to-question capability map。它只描述 `top_view_localization` 能回答哪些研究問題，以及閱讀結果時要看哪些 metrics，不記錄任何 run-specific 結論。

## 09 Top View Localization

- `question_id`: `09_top_view_localization`
- `question_title`: `Top View Localization`
- `can_validate`: 模型是否能穩定在電子封裝圖面中找出 `top_view`，以及這個能力與整體 layout slot 理解是否一致。
- `required_scope`: `provider_rollup=all_cases`
- `result_entry`: [../../experiment_results/2026-04-16_full_suite/README.md#sanity-checks](../../experiment_results/2026-04-16_full_suite/README.md#sanity-checks)
- `result_metrics`: `answer_micro_top_view_accuracy` and `answer_micro_layout_slot_accuracy`
