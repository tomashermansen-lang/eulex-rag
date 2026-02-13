from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_REGISTRY_FILENAME = "corpus_registry.json"


def normalize_corpus_id(corpus_id: str) -> str:
    return " ".join((corpus_id or "").casefold().strip().split()).replace("-", "_")


def normalize_alias(a: str) -> str:
    return " ".join((a or "").casefold().strip().split())


def default_registry_path(project_root: Path) -> Path:
    return project_root / "data" / "processed" / DEFAULT_REGISTRY_FILENAME


def load_registry(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as err:
        raise SystemExit(
            f"Invalid JSON in corpus registry '{path}': {err}. Fix the file or recreate it via ingestion."
        )
    except OSError as err:
        raise SystemExit(f"Could not read corpus registry '{path}': {err}")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid format in corpus registry '{path}': expected a top-level JSON object")
    return data


def save_registry(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    try:
        tmp_path.write_text(payload + "\n", encoding="utf-8")
        os.replace(tmp_path, path)
    except OSError as err:
        raise SystemExit(f"Could not write corpus registry '{path}': {err}")


def derive_aliases(corpus_id: str, display_name: str) -> list[str]:
    candidates: list[str] = []

    dn = normalize_alias(display_name)
    if len(dn) >= 3:
        candidates.append(dn)

    cid_norm = normalize_corpus_id(corpus_id)
    cid_as_alias = normalize_alias(cid_norm)
    if len(cid_as_alias) >= 3:
        candidates.append(cid_as_alias)

    cid_spaces = normalize_alias(cid_norm.replace("_", " "))
    if len(cid_spaces) >= 3:
        candidates.append(cid_spaces)

    cid_hyphen = normalize_alias(cid_norm.replace("_", "-"))
    if len(cid_hyphen) >= 3:
        candidates.append(cid_hyphen)

    seen: set[str] = set()
    out: list[str] = []
    for a in candidates:
        a = normalize_alias(a)
        if len(a) < 3:
            continue
        if a in seen:
            continue
        seen.add(a)
        out.append(a)
    return out


def upsert_corpus(registry: dict, corpus_id: str, display_name: str, aliases: list[str]) -> dict:
    key = normalize_corpus_id(corpus_id)
    entry = registry.get(key)
    if not isinstance(entry, dict):
        entry = {}

    display = (display_name or "").strip() or key

    existing_aliases = entry.get("aliases")
    merged: list[str] = []
    if isinstance(existing_aliases, list):
        for a in existing_aliases:
            if isinstance(a, str):
                merged.append(normalize_alias(a))

    for a in aliases or []:
        if isinstance(a, str):
            merged.append(normalize_alias(a))

    merged_unique: list[str] = []
    seen: set[str] = set()
    for a in merged:
        if len(a) < 3:
            continue
        if a in seen:
            continue
        seen.add(a)
        merged_unique.append(a)

    registry[key] = {
        "display_name": display,
        "aliases": merged_unique,
    }
    return registry
