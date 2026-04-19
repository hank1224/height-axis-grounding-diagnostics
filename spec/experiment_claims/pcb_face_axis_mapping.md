# pcb_face_axis_mapping 可驗證問題

本文件是 task-to-question capability map。它只描述 `pcb_face_axis_mapping` 能回答哪些研究問題，以及閱讀結果時要看哪些 metrics，不記錄任何 run-specific 結論。

## 11 PCB Mounting Face Axis Mapping

- `question_id`: `11_pcb_mounting_face_axis_mapping`
- `question_title`: `PCB Mounting Face Axis Mapping`
- `can_validate`: 模型是否能把三張視圖穩定映射回 PCB mounting face 的 signed drawing-space axis，以及這個能力在逐軸正確率與整案例 exact match 兩個尺度上如何表現。
- `required_scope`: `provider_rollup=all_cases`
- `result_entry`: [../../experiment_results/2026-04-16_full_suite/README.md#pcb-mounting-face-axis-mapping](../../experiment_results/2026-04-16_full_suite/README.md#pcb-mounting-face-axis-mapping)
- `result_metrics`: `answer_micro_axis_accuracy` and `case_macro_axis_exact_match`
