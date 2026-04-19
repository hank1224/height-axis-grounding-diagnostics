# Package Drawings Image Dataset

This directory contains the canonical image dataset generated from the Notion export under:

```text
data/raw/notion_exports/package_drawings/
```

It contains 15 package types with 5 image variants each, for 75 images total.

## 命名規則

- 每個封裝使用固定 slug，例如 `sot-23`、`smb-do-214aa`。
- 每張圖都改成語意化檔名：
  - `canonical-values.png`
  - `rotated-values.png`
  - `canonical-ids.png`
  - `rotated-ids.png`
  - `rotated-ids-seating-plane-anchor.png`
- 每張圖片另有穩定的 `image_id`，格式為 `{package_slug}__{variant_slug}`。
- 每張圖片保留 package-level `shape_class` metadata，來自來源表的 `外型分類` 欄位並正規化成 snake_case ID。

## 主要檔案

- `image_manifest.json`: machine-readable image list.
- `image_manifest.csv`: spreadsheet-friendly image list.
- `images/`: 重新命名後的正式資料集圖片。

## Rebuild

```bash
./.venv/bin/python scripts/import_notion_package_drawings.py
```

Task cases are generated separately under `data/tasks/`.
