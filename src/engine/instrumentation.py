import os
import re
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from .constants import _TRUTHY_ENV_VALUES
from .helpers import _truthy_env
from .types import ClaimIntent
from .planning import UserProfile

def is_debug_corpus_enabled() -> bool:
    return _truthy_env("RAG_DEBUG_CORPUS")


def is_intent_log_enabled() -> bool:
    return _truthy_env("RAG_INTENT_LOG_ENABLED")


def is_intent_log_only_general() -> bool:
    return _truthy_env("RAG_INTENT_LOG_ONLY_GENERAL")


def is_debug_dump_run_meta_enabled() -> bool:
    return _truthy_env("RAG_DEBUG_DUMP_RUN_META")


def is_engineering_json_mode_enabled() -> bool:
    return _truthy_env("ENGINEERING_JSON_MODE")


def is_legal_json_mode_enabled() -> bool:
    return _truthy_env("LEGAL_JSON_MODE")


def is_debug_enabled() -> bool:
    """Check if RAG_DEBUG is enabled."""
    return _truthy_env("RAG_DEBUG")


def _debug_dump_run_meta(*, stage: str, run_meta: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    """Write deterministic run_meta snapshots when explicitly enabled.

    Controlled by env:
    - RAG_DEBUG_DUMP_RUN_META=1
    - RAG_DEBUG_DUMP_RUN_META_PATH=runs/debug/.../file.json
    """

    try:
        enabled = str(os.getenv("RAG_DEBUG_DUMP_RUN_META", "") or "").strip().lower() in _TRUTHY_ENV_VALUES
        out_path = str(os.getenv("RAG_DEBUG_DUMP_RUN_META_PATH", "") or "").strip()
        if not enabled or not out_path:
            return

        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        def _jsonable(obj: Any) -> Any:
            try:
                json.dumps(obj)
                return obj
            except Exception:  # noqa: BLE001
                if isinstance(obj, dict):
                    return {str(k): _jsonable(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_jsonable(x) for x in obj]
                return str(obj)

        base = {"case_label": str(os.getenv("RAG_DEBUG_DUMP_LABEL", "") or ""), "snapshots": []}
        if p.exists():
            try:
                base = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                base = {"case_label": str(os.getenv("RAG_DEBUG_DUMP_LABEL", "") or ""), "snapshots": []}

        snaps = base.get("snapshots")
        if not isinstance(snaps, list):
            snaps = []
            base["snapshots"] = snaps

        snaps.append(
            {
                "stage": stage,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_meta": _jsonable(run_meta),
                "extra": _jsonable(extra or {}),
            }
        )
        p.write_text(json.dumps(base, indent=2), encoding="utf-8")

    except Exception:  # noqa: BLE001
        pass


def log_intent_event(
    *,
    question: str,
    intent: ClaimIntent,
    profile: UserProfile | str,
    corpus_id: str | None = None,
    project_root: Path | None = None,
) -> None:
    """Log an intent classification event to JSONL for offline analysis.

    Controlled by env:
    - RAG_INTENT_LOG_ENABLED=1
    - RAG_INTENT_LOG_ONLY_GENERAL=1 (optional: only log GENERAL intents)
    - RAG_INTENT_LOG_MAX_CHARS=500 (optional: truncate question)
    - RAG_INTENT_LOG_PATH=path/to/file.jsonl (optional: override log path)
    """
    from . import policy as policy_engine  # late import to avoid circular

    try:
        enabled = is_intent_log_enabled()
        if not enabled:
            return

        only_general = is_intent_log_only_general()
        if only_general and str(getattr(intent, "value", intent) or "") != ClaimIntent.GENERAL.value:
            return

        max_chars_raw = str(os.getenv("RAG_INTENT_LOG_MAX_CHARS", "") or "").strip()
        max_chars = 500
        if max_chars_raw:
            try:
                max_chars = int(max_chars_raw)
            except Exception:  # noqa: BLE001
                max_chars = 500
        max_chars = max(0, min(max_chars, 5000))

        q_raw = str(question or "")

        # Normalize for hash (pre-redaction) to enable offline de-duplication.
        q_norm = re.sub(r"\s+", " ", q_raw.strip().lower())
        q_hash = hashlib.sha256(q_norm.encode("utf-8", errors="ignore")).hexdigest()

        # Basic redaction: emails and phone-like long digit sequences.
        q_redacted = q_raw
        q_redacted = re.sub(
            r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            "[REDACTED_EMAIL]",
            q_redacted,
        )
        # Phone-like: sequences containing >=8 digits allowing separators.
        q_redacted = re.sub(
            r"(?x)(?<!\w)(?:\+?\d[\d\s().-]{6,}\d)(?!\w)",
            lambda m: "[REDACTED_PHONE]" if len(re.sub(r"\D", "", m.group(0))) >= 8 else m.group(0),
            q_redacted,
        )

        if max_chars and len(q_redacted) > max_chars:
            q_redacted = q_redacted[:max_chars]

        signals = policy_engine._intent_match_signals(q_raw)

        profile_str = None
        try:
            if isinstance(profile, str):
                profile_str = profile.strip().upper() or None
            else:
                profile_str = str(getattr(profile, "value", None) or getattr(profile, "name", None) or "").strip().upper() or None
        except Exception:  # noqa: BLE001
            profile_str = None
        profile_str = profile_str or "UNKNOWN"

        corpus_id_str = str(corpus_id or "").strip() or None

        rerank = {
            "hybrid": True,  # 4-factor hybrid rerank always enabled
        }

        log_path_env = str(os.getenv("RAG_INTENT_LOG_PATH", "") or "").strip()
        if log_path_env:
            log_path = Path(log_path_env).expanduser()
        else:
            try:
                project_root_path = Path(str(project_root)).resolve() if project_root else Path.cwd().resolve()
            except Exception:  # noqa: BLE001
                project_root_path = Path.cwd()
            log_path = project_root_path / "data" / "intent_logs" / "intent_events.jsonl"

        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            return

        payload = {
            "ts_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "corpus_id": corpus_id_str,
            "profile": profile_str,
            "intent": str(getattr(intent, "value", intent) or ""),
            "rerank": rerank,
            "question": q_redacted,
            "question_sha256": q_hash,
            "signals": {
                "enforcement": bool(signals.get("enforcement")),
                "requirements": bool(signals.get("requirements")),
                "classification": bool(signals.get("classification")),
                "scope": bool(signals.get("scope")),
            },
        }

        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception:  # noqa: BLE001
            return
    except Exception:  # noqa: BLE001
        return


def collect_corpus_debug_telemetry(
    answer_text: str,
    references_structured_all: list[dict[str, Any]],
    contract_min_citations: int | None,
    run_meta: dict[str, Any],
    total_retrieved: int | None = None,
    citable_count: int | None = None,
    strip_references_fn: Any | None = None,
    count_normative_fn: Any | None = None,
) -> None:
    """Collect and emit corpus debug telemetry.

    This function consolidates the debug telemetry collection that was
    previously inline in rag.py answer_structured() (lines 2315-2402).

    Args:
        answer_text: The final answer text.
        references_structured_all: All structured references.
        contract_min_citations: Minimum citations required by contract.
        run_meta: The run metadata dict to update (mutated in-place).
        total_retrieved: Total number of retrieved chunks.
        citable_count: Number of citable chunks.
        strip_references_fn: Function to strip trailing references section.
        count_normative_fn: Function to count normative sentences.
    """
    from . import helpers as _helpers

    _strip = strip_references_fn or _helpers._strip_trailing_references_section
    _count_normative = count_normative_fn or _helpers._count_normative_sentences

    allowed_idxs = sorted(
        {
            int(r.get("idx"))
            for r in list(references_structured_all or [])
            if isinstance(r, dict) and str(r.get("idx") or "").strip().isdigit()
        }
    )

    txt_for_citations = _strip(str(answer_text or ""))
    citation_matches_raw = [m.group(0) for m in re.finditer(r"\[(\d{1,3})\]", txt_for_citations)]
    cited_idxs = sorted({int(m.group(1)) for m in re.finditer(r"\[(\d{1,3})\]", txt_for_citations)})
    valid_cited = sorted(set(cited_idxs) & set(allowed_idxs))

    normative_count = _count_normative(txt_for_citations)
    has_utilstraekkelig = bool(re.search(r"(?i)\bUTILSTRÆKKELIG_EVIDENS\b", txt_for_citations))

    missing_ref = str(answer_text or "").strip() == "MISSING_REF"
    gate_reason = None
    if missing_ref:
        req = (run_meta.get("missing_ref_reason") or {}).get("required_support")
        if str(req or "").strip().lower() == "min_citations_not_met":
            gate_reason = "MIN_CITATIONS_NOT_MET"
        elif req:
            gate_reason = str(req).upper()
        else:
            try:
                min_cit_dbg = int(contract_min_citations) if contract_min_citations is not None else 0
            except Exception:  # noqa: BLE001
                min_cit_dbg = 0
            if min_cit_dbg > 0 and len(allowed_idxs) >= min_cit_dbg and len(valid_cited) < min_cit_dbg:
                gate_reason = "MIN_CITATIONS_NOT_MET"
            else:
                gate_reason = "MISSING_REF"

    try:
        total_retrieved_debug = int(total_retrieved) if total_retrieved is not None else None
    except Exception:  # noqa: BLE001
        total_retrieved_debug = None
    try:
        citable_debug = int(citable_count) if citable_count is not None else None
    except Exception:  # noqa: BLE001
        citable_debug = None

    run_meta.setdefault("corpus_debug", {})
    run_meta["corpus_debug"].update(
        {
            "retrieved_refs": total_retrieved_debug,
            "citable_refs": citable_debug,
            "allowed_idxs": allowed_idxs,
            "parsed_citations_raw": citation_matches_raw,
            "cited_idxs": cited_idxs,
            "valid_cited": valid_cited,
            "normative_count": int(normative_count),
            "has_utilstraekkelig_evidens": bool(has_utilstraekkelig),
            "final_gate_reason": gate_reason,
            "answer_missing_ref": bool(missing_ref),
        }
    )

    print(
        "[corpus_debug] "
        + json.dumps(
            {
                "selected_corpus": (run_meta.get("corpus_debug") or {}).get("selected_corpus_norm"),
                "profile": (run_meta.get("corpus_debug") or {}).get("profile"),
                "contract_check": (run_meta.get("corpus_debug") or {}).get("contract_check"),
                "min_citations": (run_meta.get("corpus_debug") or {}).get("expected_min_citations"),
                "corpus_candidates": (run_meta.get("corpus_debug") or {}).get("corpus_candidates"),
                "retrieved_refs": total_retrieved_debug,
                "citable_refs": citable_debug,
                "allowed_idxs": allowed_idxs,
                "parsed_citations_raw": citation_matches_raw,
                "cited_idxs": cited_idxs,
                "valid_cited": valid_cited,
                "normative_count": int(normative_count),
                "has_utilstraekkelig_evidens": bool(has_utilstraekkelig),
                "gate_reason": gate_reason,
                "missing_ref": bool(missing_ref),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
