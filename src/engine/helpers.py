"""Re-export facade for backward compatibility.

All functions that lived here have been moved to focused modules:
- metadata_helpers.py — metadata normalization + anchor extraction
- query_helpers.py — question parsing + ref extraction + intent heuristics
- text_transforms.py — answer text normalization + modal translation
- payload_builders.py — response payload assembly
- constants.py — _truthy_env
- rag_config.py — iso_utc_now, best_effort_git_commit_short, collection_name_best_effort

This file re-exports every public name so existing callers continue to work.
New code should import from the specific module directly.
"""

# --- constants ---
from .constants import _truthy_env  # noqa: F401

# --- metadata_helpers ---
from .metadata_helpers import (  # noqa: F401
    normalize_anchor,
    normalize_annex_for_chroma,
    get_meta_value,
    normalize_metadata,
    normalize_anchor_list,
    _derive_structural_fields_from_location_id,
    _extract_raw_anchors_from_chunk,
    anchors_from_metadata,
    anchors_present_from_hits,
    _extract_anchor_mentions_from_answer,
)

# --- query_helpers ---
from .query_helpers import (  # noqa: F401
    classify_query_intent,
    _extract_article_ref,
    _extract_article_refs,
    _looks_like_multi_part_question,
    _extract_annex_refs,
    _extract_recital_ref,
    _looks_like_recital_quote_question,
    _extract_chapter_ref,
    _extract_section_ref,
    _roman_to_int,
    _ref_to_int,
    _looks_like_structure_question,
    _looks_like_substantive_question,
    _looks_like_chapter_overview_question,
    _looks_like_chapter_summary_question,
)

# --- text_transforms ---
from .text_transforms import (  # noqa: F401
    _count_normative_sentences,
    _normalize_modals_to_danish,
    _strip_trailing_references_section,
    _normalize_abstain_text,
)

# --- payload_builders ---
from .payload_builders import (  # noqa: F401
    compute_required_anchor_idxs,
    build_answer_response_payload,
    build_retrieval_state_dict,
    build_hybrid_rerank_dict,
)

# --- rag_config ---
from .rag_config import (  # noqa: F401
    iso_utc_now,
    best_effort_git_commit_short,
    collection_name_best_effort,
)
