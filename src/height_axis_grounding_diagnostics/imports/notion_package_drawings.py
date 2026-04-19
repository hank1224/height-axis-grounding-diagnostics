#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[3]
RAW_EXPORT_DIR = ROOT / "data" / "raw" / "notion_exports" / "package_drawings"
SOURCE_CSV = RAW_EXPORT_DIR / "export.csv"
SOURCE_FILES_DIR = RAW_EXPORT_DIR / "files"
DATASET_DIR = ROOT / "data" / "package_drawings"
IMAGES_DIR = DATASET_DIR / "images"
MANIFEST_JSON = DATASET_DIR / "image_manifest.json"
MANIFEST_CSV = DATASET_DIR / "image_manifest.csv"
README_PATH = DATASET_DIR / "README.md"


VARIANT_SPECS = [
    {
        "column": "Canonical + Values",
        "variant_slug": "canonical-values",
    },
    {
        "column": "Rotated + Values",
        "variant_slug": "rotated-values",
    },
    {
        "column": "Canonical + IDs",
        "variant_slug": "canonical-ids",
    },
    {
        "column": "Rotated + IDs",
        "variant_slug": "rotated-ids",
    },
    {
        "column": "Rotated + IDs + seating-plane anchor",
        "variant_slug": "rotated-ids-seating-plane-anchor",
    },
]

SHAPE_CLASS_SLUGS = {
    "SOT-like small multi-terminal SMD": "sot_like_smd",
    "Small SMD diode / two-terminal": "two_terminal_diode_smd",
    "Tabbed power SMD / TO-SOT": "tabbed_power_smd",
}


def slugify(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace("&", "and")
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized)
    return normalized.strip("-")


def shape_class_slug(value: str) -> str:
    try:
        return SHAPE_CLASS_SLUGS[value.strip()]
    except KeyError as exc:
        raise ValueError(f"Unsupported shape class: {value!r}") from exc


def ensure_clean_dataset_dir() -> None:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def load_rows() -> list[dict[str, str]]:
    with SOURCE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def copy_images_and_build_manifest(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    images: list[dict[str, object]] = []

    for row in rows:
        package_name = row["封裝名"]
        package_slug = slugify(package_name)
        package_dir = IMAGES_DIR / package_slug
        package_dir.mkdir(parents=True, exist_ok=True)

        for variant in VARIANT_SPECS:
            source_rel = unquote(row[variant["column"]])
            source_parts = Path(source_rel).parts
            source_path = SOURCE_FILES_DIR / source_rel
            if not source_path.exists() and len(source_parts) > 1:
                source_rel = Path(*source_parts[1:]).as_posix()
                source_path = SOURCE_FILES_DIR / source_rel
            destination_path = package_dir / f"{variant['variant_slug']}.png"
            shutil.copy2(source_path, destination_path)

            image_id = f"{package_slug}__{variant['variant_slug']}"
            images.append(
                {
                    "image_id": image_id,
                    "package_name": package_name,
                    "package_slug": package_slug,
                    "kicad_model_name": row["KiCad 模型名"],
                    "shape_class": shape_class_slug(row["外型分類"]),
                    "variant_name": variant["column"],
                    "variant_slug": variant["variant_slug"],
                    "source_image_path": source_path.relative_to(ROOT).as_posix(),
                    "image_path": destination_path.relative_to(ROOT).as_posix(),
                }
            )

    images.sort(key=lambda item: item["image_id"])
    return images


def write_manifest_json(images: list[dict[str, object]]) -> None:
    payload = {
        "dataset_name": "package_drawings",
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": SOURCE_CSV.relative_to(ROOT).as_posix(),
        "image_count": len(images),
        "images": images,
    }
    MANIFEST_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_manifest_csv(images: list[dict[str, object]]) -> None:
    fieldnames = [
        "image_id",
        "package_name",
        "package_slug",
        "kicad_model_name",
        "shape_class",
        "variant_name",
        "variant_slug",
        "source_image_path",
        "image_path",
    ]
    with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(images)


def write_readme() -> None:
    README_PATH.write_text(
        """# Package Drawings Image Dataset

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
""",
        encoding="utf-8",
    )


def main() -> None:
    ensure_clean_dataset_dir()
    rows = load_rows()
    images = copy_images_and_build_manifest(rows)
    write_manifest_json(images)
    write_manifest_csv(images)
    write_readme()
    print(f"Generated image dataset with {len(images)} images in {DATASET_DIR}")


if __name__ == "__main__":
    main()
