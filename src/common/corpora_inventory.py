from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_CORPORA_FILENAME = "corpora.json"


def default_corpora_path(project_root: Path) -> Path:
    return project_root / "data" / "processed" / DEFAULT_CORPORA_FILENAME


def load_corpora_inventory(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "corpora": {}}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as err:
        raise SystemExit(
            f"Invalid JSON in corpora inventory '{path}': {err}. Fix the file or recreate it via ingestion."
        )
    except OSError as err:
        raise SystemExit(f"Could not read corpora inventory '{path}': {err}")

    if data is None:
        return {"version": 1, "corpora": {}}
    if not isinstance(data, dict):
        raise SystemExit(
            f"Invalid format in corpora inventory '{path}': expected a top-level JSON object"
        )

    if "version" not in data:
        data["version"] = 1
    if "corpora" not in data or not isinstance(data.get("corpora"), dict):
        data["corpora"] = {}

    return data


def save_corpora_inventory(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    try:
        tmp_path.write_text(payload + "\n", encoding="utf-8")
        os.replace(tmp_path, path)
    except OSError as err:
        raise SystemExit(f"Could not write corpora inventory '{path}': {err}")


def upsert_corpus_inventory(
    data: dict,
    corpus_id: str,
    display_name: str | None = None,
    enabled: bool = True,
    extra: dict | None = None,
) -> dict:
    if not isinstance(data, dict):
        data = {"version": 1, "corpora": {}}

    if "version" not in data:
        data["version"] = 1

    corpora = data.get("corpora")
    if not isinstance(corpora, dict):
        corpora = {}
        data["corpora"] = corpora

    key = str(corpus_id or "").strip()
    if not key:
        return data

    entry = corpora.get(key)
    if not isinstance(entry, dict):
        entry = {}

    entry["display_name"] = (display_name or "").strip() or key.upper()
    entry["enabled"] = bool(enabled)

    if isinstance(extra, dict) and extra:
        for k, v in extra.items():
            entry[k] = v

    corpora[key] = entry
    return data
