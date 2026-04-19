# package_target_extraction 可驗證問題

本文件是 task-to-question capability map。它只描述 `package_target_extraction` 能回答哪些研究問題、需要什麼 scope，以及要看哪個 campaign report section，不記錄任何 run-specific 結論。

## 01 Semantic Aligned Dimension Extraction

- `question_id`: `01_semantic_aligned_dimension_extraction`
- `question_title`: `Semantic Aligned Dimension Extraction`
- `can_validate`: 在語意對齊的 canonical 條件下，三家 provider 的所有尺寸欄位整體準確率如何。
- `required_scope`: `variants=canonical-values|canonical-ids`、all available dimension fields、`prompt_variant=baseline`
- `result_entry`: [../../experiment_results/2026-04-16_full_suite/README.md#package-dimension-extraction](../../experiment_results/2026-04-16_full_suite/README.md#package-dimension-extraction)
- `result_metrics`: `overall_dimension_match_rate` and variant-level `dimension_match_rate`

## 02 Rotated Overall Dimension Accuracy

- `question_id`: `rotated_overall_dimension_accuracy`
- `question_title`: `Rotated Overall Dimension Accuracy`
- `can_validate`: 在語意不對齊的 rotated 條件下，三家 provider 的所有尺寸欄位整體準確率如何。
- `required_scope`: `variants=rotated-values|rotated-ids`、all available dimension fields、`prompt_variant=baseline`
- `result_entry`: [../../experiment_results/2026-04-16_full_suite/README.md#package-dimension-extraction](../../experiment_results/2026-04-16_full_suite/README.md#package-dimension-extraction)
- `result_metrics`: `overall_dimension_match_rate` and variant-level `dimension_match_rate`

## 12 Rotated Damage Profile

- `question_id`: `rotated_damage_profile`
- `question_title`: `Rotated Damage Profile`
- `can_validate`: rotated 條件造成整體準確率下降時，主要受傷的是哪些尺寸欄位。
- `required_scope`: `comparison=canonical vs rotated`、same provider / prompt family / field、`prompt_variant=baseline`
- `result_entry`: [../../experiment_results/2026-04-16_full_suite/README.md#package-dimension-extraction](../../experiment_results/2026-04-16_full_suite/README.md#package-dimension-extraction)
- `result_metrics`: `match_count_delta` by prompt family and field

## 03 Numeric Self Correction

- `question_id`: `03_numeric_self_correction`
- `question_title`: `Numeric Self Correction`
- `can_validate`: 在 rotated 條件下，numeric cue 是否會改變 `overall_package_height` 的整體 field-level match；height nuisance subset 可作為附加觀察。
- `required_scope`: `comparison=rotated-values vs rotated-ids`、`field=overall_package_height`、`prompt_variant=baseline`
- `result_entry`: [../../experiment_results/2026-04-16_full_suite/README.md#package-dimension-extraction](../../experiment_results/2026-04-16_full_suite/README.md#package-dimension-extraction)
- `result_metrics`: primary `match_rate` rows on `overall_package_height`; secondary `target_match_rate` rows on applicable rotated height nuisance cases

## 04 ID Only Dimension Extraction

- `question_id`: `04_id_only_dimension_extraction`
- `question_title`: `ID Only Dimension Extraction`
- `can_validate`: 當圖面只保留 dimension-line IDs、移除 numeric cue 時，`overall_package_height` 的 field-level 提取是否下降。
- `required_scope`: `comparison=rotated-values vs rotated-ids`、`field=overall_package_height`、`prompt_variant=baseline`
- `result_entry`: [../../experiment_results/2026-04-16_full_suite/README.md#package-dimension-extraction](../../experiment_results/2026-04-16_full_suite/README.md#package-dimension-extraction)
- `result_metrics`: `match_rate` rows on `overall_package_height`

## 05 Prompt Fix Effect

- `question_id`: `05_prompt_fix_effect`
- `question_title`: `Prompt Fix Effect`
- `can_validate`: `view_semantics_warning` 是否改變 rotated 高度欄位的 field-match，以及這個改變是否跨 variant 一致。
- `required_scope`: `comparison=baseline vs view_semantics_warning`、`variants=rotated-values|rotated-ids`、`field=overall_package_height`
- `result_entry`: [../../experiment_results/2026-04-16_full_suite/README.md#package-dimension-extraction](../../experiment_results/2026-04-16_full_suite/README.md#package-dimension-extraction)
- `result_metrics`: `match_rate` rows on `overall_package_height`

## 06 Seating Plane Anchor Effect

- `question_id`: `06_seating_plane_anchor_effect`
- `question_title`: `Seating Plane Anchor Effect`
- `can_validate`: 在 rotated ID 條件下，`SEATING PLANE` anchor 是否改變模型對目標高度的命中率。
- `required_scope`: `comparison=rotated-ids vs rotated-ids-seating-plane-anchor`、`prompt_variant=baseline`、只看 applicable rotated height nuisance cases
- `result_entry`: [../../experiment_results/2026-04-16_full_suite/README.md#package-dimension-extraction](../../experiment_results/2026-04-16_full_suite/README.md#package-dimension-extraction)
- `result_metrics`: `target_match_rate` on applicable rotated height nuisance cases

## 07 Error Awareness and Null Behavior

- `question_id`: `07_error_awareness_and_null_behavior`
- `question_title`: `Error Awareness and Null Behavior`
- `can_validate`: 在 rotated 高度易混淆案例中，模型是否傾向用 `no_prediction` abstain，還是更常直接給出答案。
- `required_scope`: `prompt_variant=baseline`、`variants=rotated-values|rotated-ids|rotated-ids-seating-plane-anchor`、只看 applicable rotated height nuisance cases
- `result_entry`: [../../experiment_results/2026-04-16_full_suite/README.md#package-dimension-extraction](../../experiment_results/2026-04-16_full_suite/README.md#package-dimension-extraction)
- `result_metrics`: `no_prediction_rate` and `target_match_rate` on applicable rotated height nuisance cases
