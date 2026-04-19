# view_role_classification 可驗證問題

本文件是 task-to-question capability map。它只描述 `view_role_classification` 能回答哪些研究問題，以及閱讀結果時要看哪些 metrics，不記錄任何 run-specific 結論。

## 10 View Role Classification

- `question_id`: `10_view_role_classification`
- `question_title`: `View Role Classification`
- `can_validate`: 當任務從單一 `top_view` 定位升級成三視圖角色分類時，`top_view`、`side_view`、`end_view` 三種角色的辨識能力是否仍然一致。
- `required_scope`: `provider_rollup=all_cases`、逐一比較 `top_view` / `side_view` / `end_view`
- `result_entry`: [../../experiment_results/2026-04-16_full_suite/README.md#sanity-checks](../../experiment_results/2026-04-16_full_suite/README.md#sanity-checks)
- `result_metrics`: `answer_micro_top_view_accuracy`, `answer_micro_side_view_accuracy`, and `answer_micro_end_view_accuracy`
