from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from ..common.corpus_registry import default_registry_path, load_registry, normalize_alias, normalize_corpus_id


@dataclass(frozen=True)
class CorpusInfo:
    corpus_key: str  # normalized via normalize_corpus_id (hyphen -> underscore)
    display_name: str
    aliases: tuple[str, ...]


def _word_boundary_regex(term: str) -> re.Pattern[str]:
    # Use Unicode-aware \w boundaries to avoid matching inside other words.
    # Term is already normalized (casefolded, collapsed whitespace).
    esc = re.escape(term)
    return re.compile(rf"(?<!\w){esc}(?!\w)")


@dataclass(frozen=True)
class CorpusResolver:
    by_key: dict[str, CorpusInfo]
    _alias_patterns: tuple[tuple[str, re.Pattern[str]], ...]

    @classmethod
    def from_project_root(cls, project_root: Path) -> "CorpusResolver":
        reg_path = default_registry_path(project_root)
        raw = load_registry(reg_path)

        by_key: dict[str, CorpusInfo] = {}
        patterns: list[tuple[str, re.Pattern[str]]] = []

        for corpus_id_raw, entry in sorted(raw.items(), key=lambda t: str(t[0])):
            if not isinstance(corpus_id_raw, str) or not isinstance(entry, dict):
                continue
            key = normalize_corpus_id(corpus_id_raw)
            display = str(entry.get("display_name") or key).strip() or key

            aliases_in = entry.get("aliases")
            aliases: list[str] = []
            if isinstance(aliases_in, list):
                for a in aliases_in:
                    if isinstance(a, str) and a.strip():
                        aliases.append(normalize_alias(a))

            # Also match display name as an alias.
            dn = normalize_alias(display)
            if dn and dn not in aliases:
                aliases.append(dn)

            # Keep deterministic order; ignore very short aliases.
            cleaned: list[str] = []
            seen: set[str] = set()
            for a in aliases:
                a2 = normalize_alias(a)
                if len(a2) < 3:
                    continue
                if a2 in seen:
                    continue
                seen.add(a2)
                cleaned.append(a2)

            info = CorpusInfo(corpus_key=key, display_name=display, aliases=tuple(cleaned))
            by_key[key] = info
            for a in info.aliases:
                patterns.append((key, _word_boundary_regex(a)))

        # Prefer longer aliases first to reduce accidental matches.
        patterns.sort(key=lambda t: (-len(t[1].pattern), t[0]))
        return cls(by_key=by_key, _alias_patterns=tuple(patterns))

    def display_name_for(self, corpus_id: str | None) -> str | None:
        key = normalize_corpus_id(str(corpus_id or ""))
        info = self.by_key.get(key)
        if info is None:
            return None
        return info.display_name

    def mentioned_corpus_keys(self, text: str | None) -> list[str]:
        """Return corpus keys mentioned in text based on registry aliases.

        Conservative, deterministic matching:
        - casefold + whitespace-collapsed matching
        - word-boundary regexes for each alias
        """

        q = normalize_alias(str(text or ""))
        if not q:
            return []
        found: set[str] = set()
        for key, pat in self._alias_patterns:
            if key in found:
                continue
            if pat.search(q):
                found.add(key)
        return sorted(found)

    def any_alias_in(self, text: str | None, *, keys: Iterable[str] | None = None) -> bool:
        q = normalize_alias(str(text or ""))
        if not q:
            return False
        allowed = None
        if keys is not None:
            allowed = {normalize_corpus_id(k) for k in keys}
        for key, pat in self._alias_patterns:
            if allowed is not None and key not in allowed:
                continue
            if pat.search(q):
                return True
        return False

    def get_all_corpus_ids(self) -> list[str]:
        """Return all registered corpus IDs."""
        return list(self.by_key.keys())


@lru_cache(maxsize=8)
def load_resolver_for_project_root(project_root: str) -> CorpusResolver:
    return CorpusResolver.from_project_root(Path(project_root))
