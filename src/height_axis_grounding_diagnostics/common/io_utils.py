from __future__ import annotations

import base64
import csv
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


class JsonParseError(Exception):
    pass


def load_env_file(env_path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not env_path.exists():
        return env

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        env[key] = value
        os.environ.setdefault(key, value)
    return env


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise JsonParseError(f"Missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise JsonParseError(f"Invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_json_candidate(text: str) -> str:
    cleaned = text.strip()
    fenced_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    start_positions = [idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx != -1]
    if not start_positions:
        return cleaned
    start = min(start_positions)

    end_positions = [idx for idx in (cleaned.rfind("}"), cleaned.rfind("]")) if idx != -1]
    if not end_positions:
        return cleaned[start:]
    end = max(end_positions)
    if end >= start:
        return cleaned[start : end + 1]
    return cleaned


def parse_json_text(text: str) -> tuple[Any | None, str | None]:
    candidate = extract_json_candidate(text)
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as exc:
        return None, f"JSON parse failed: {exc}"


def detect_mime_type(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    return mime_type or "application/octet-stream"


def encode_image_to_base64(image_path: Path) -> tuple[str, str]:
    mime_type = detect_mime_type(image_path)
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return mime_type, image_b64


def sanitize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bytes):
        return {
            "__type__": "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, bytearray):
        return {
            "__type__": "bytearray",
            "base64": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, set):
        return [sanitize_for_json(item) for item in sorted(value, key=repr)]
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return value.isoformat()
        except TypeError:
            pass
    return repr(value)


def dump_sdk_response(response: Any) -> tuple[str, Any]:
    if hasattr(response, "model_dump"):
        payload = response.model_dump()
        safe_payload = sanitize_for_json(payload)
        return json.dumps(safe_payload, ensure_ascii=False, indent=2), safe_payload
    if hasattr(response, "to_json_dict"):
        payload = response.to_json_dict()
        safe_payload = sanitize_for_json(payload)
        return json.dumps(safe_payload, ensure_ascii=False, indent=2), safe_payload
    if isinstance(response, dict):
        safe_payload = sanitize_for_json(response)
        return json.dumps(safe_payload, ensure_ascii=False, indent=2), safe_payload
    return str(response), None
