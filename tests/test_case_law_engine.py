"""Tests for case_law_engine — CLI orchestrator for case law ingestion.

Covers: argument parsing, corpus resolution, document type detection,
HTML download, doc version, single-case processing, enrichment,
dry-run table, summary report, and full integration scenarios.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from src.common.config_loader import (
    CaseLawSettings,
    Settings,
)
from src.ingestion.cellar_models import CaseMetadata


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

CORPUS_CELEX_MAPPING = {"32016R0679": "gdpr", "32024R1689": "ai_act"}
CELEX_DOC_TYPE_MAPPING = {"CJ": "judgment", "CC": "ag_opinion", "CO": "order"}

SAMPLE_HTML = """<html><body>
<div id="document1">
<p class="sti-art">Grounds</p>
<p>1. The applicant seeks annulment of the decision.</p>
<p>2. The Court considers the merits.</p>
<p class="sti-art">Operative part</p>
<p>The action is dismissed.</p>
</div>
</body></html>"""


def _make_settings(
    *,
    enabled: bool = True,
    enrichment_enabled: bool = False,
    **overrides: Any,
) -> Settings:
    """Create Settings with case_law config for testing."""
    cl_kwargs: dict[str, Any] = {
        "enabled": enabled,
        "sparql_endpoint": "https://publications.europa.eu/webapi/rdf/sparql",
        "request_timeout_secs": 30,
        "max_retries": 3,
        "request_delay_secs": 1.0,
        "user_agent": "TestAgent/1.0",
        "max_response_bytes": 10_485_760,
    }
    cl_kwargs.update(overrides)

    # Build embedding_enrichment section in raw settings
    return Settings(case_law=CaseLawSettings(**cl_kwargs))


def _make_case(
    *,
    ecli: str = "ECLI:EU:C:2020:790",
    case_number: str = "C-311/18",
    celex_id: str = "62020CJ0790",
    date: str = "2020-07-16",
    court: str = "C",
    title: str = "Schrems II",
) -> CaseMetadata:
    return CaseMetadata(
        ecli=ecli,
        case_number=case_number,
        celex_id=celex_id,
        date=date,
        court=court,
        title=title,
    )


def _make_mock_response(content: str = SAMPLE_HTML, status_code: int = 200):
    """Create a mock requests.Response with iter_content support."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    # iter_content returns chunks; for simplicity, one chunk
    resp.iter_content.return_value = iter([content.encode("utf-8")])
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# C1.1: parse_args()
# ─────────────────────────────────────────────────────────────────────────────


class TestParseArgs:
    """T1.1.1–T1.1.9: CLI argument parsing."""

    def test_corpus_flag(self):
        from src.ingestion.case_law_engine import parse_args

        args = parse_args(["--corpus", "gdpr"])
        assert args.corpus == "gdpr"

    def test_dry_run_flag(self):
        from src.ingestion.case_law_engine import parse_args

        args = parse_args(["--corpus", "gdpr", "--dry-run"])
        assert args.dry_run is True

    def test_out_dir_flag(self):
        from src.ingestion.case_law_engine import parse_args

        args = parse_args(["--corpus", "gdpr", "--out-dir", "/tmp/out"])
        assert args.out_dir == "/tmp/out"

    def test_language_flag(self):
        from src.ingestion.case_law_engine import parse_args

        args = parse_args(["--corpus", "gdpr", "--language", "da"])
        assert args.language == "da"

    def test_log_level_flag(self):
        from src.ingestion.case_law_engine import parse_args

        args = parse_args(["--corpus", "gdpr", "--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

    def test_limit_flag(self):
        from src.ingestion.case_law_engine import parse_args

        args = parse_args(["--corpus", "gdpr", "--limit", "5"])
        assert args.limit == 5

    def test_missing_corpus_exits(self):
        from src.ingestion.case_law_engine import parse_args

        with pytest.raises(SystemExit):
            parse_args([])

    def test_empty_corpus_exits(self, monkeypatch):
        from src.ingestion.case_law_engine import parse_args

        with pytest.raises(SystemExit):
            parse_args(["--corpus", ""])

    def test_limit_zero_means_no_limit(self):
        from src.ingestion.case_law_engine import parse_args

        args = parse_args(["--corpus", "gdpr", "--limit", "0"])
        assert args.limit == 0


# ─────────────────────────────────────────────────────────────────────────────
# C1.2: resolve_corpus_to_celex()
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveCorpusToCelex:
    """T1.2.1–T1.2.4: Corpus-to-CELEX resolution."""

    def test_resolves_gdpr(self):
        from src.ingestion.case_law_engine import resolve_corpus_to_celex

        assert resolve_corpus_to_celex("gdpr", CORPUS_CELEX_MAPPING) == "32016R0679"

    def test_case_insensitive(self):
        from src.ingestion.case_law_engine import resolve_corpus_to_celex

        assert resolve_corpus_to_celex("GDPR", CORPUS_CELEX_MAPPING) == "32016R0679"

    def test_underscore_corpus(self):
        from src.ingestion.case_law_engine import resolve_corpus_to_celex

        assert resolve_corpus_to_celex("ai_act", CORPUS_CELEX_MAPPING) == "32024R1689"

    def test_unknown_corpus_exits(self):
        from src.ingestion.case_law_engine import resolve_corpus_to_celex

        with pytest.raises(SystemExit, match="Unknown corpus"):
            resolve_corpus_to_celex("unknown", CORPUS_CELEX_MAPPING)


# ─────────────────────────────────────────────────────────────────────────────
# C1.3: detect_document_type()
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectDocumentType:
    """T1.3.1–T1.3.5: Document type detection from CELEX."""

    def test_cj_is_judgment(self):
        from src.ingestion.case_law_engine import detect_document_type

        result = detect_document_type("62020CJ0790", CELEX_DOC_TYPE_MAPPING)
        assert result.value == "judgment"

    def test_cc_is_ag_opinion(self):
        from src.ingestion.case_law_engine import detect_document_type

        result = detect_document_type("62023CC0197", CELEX_DOC_TYPE_MAPPING)
        assert result.value == "ag_opinion"

    def test_co_is_order(self):
        from src.ingestion.case_law_engine import detect_document_type

        result = detect_document_type("62023CO0050", CELEX_DOC_TYPE_MAPPING)
        assert result.value == "order"

    def test_unrecognized_defaults_to_judgment(self, caplog):
        from src.ingestion.case_law_engine import detect_document_type

        with caplog.at_level(logging.WARNING):
            result = detect_document_type("62020XX0790", CELEX_DOC_TYPE_MAPPING)
        assert result.value == "judgment"
        assert "Unrecognized" in caplog.text or "unrecognized" in caplog.text.lower()

    def test_ambiguous_first_match_wins(self):
        from src.ingestion.case_law_engine import detect_document_type

        result = detect_document_type("62020CJCC0790", CELEX_DOC_TYPE_MAPPING)
        assert result.value == "judgment"  # CJ matches first


# ─────────────────────────────────────────────────────────────────────────────
# C1.4: download_case_html()
# ─────────────────────────────────────────────────────────────────────────────


class TestDownloadCaseHtml:
    """T1.4.1–T1.4.6: HTML download with rate limiting."""

    def test_returns_html_content(self, monkeypatch):
        from src.ingestion.case_law_engine import download_case_html

        mock_resp = _make_mock_response("<html>test</html>")
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.requests.get",
            MagicMock(return_value=mock_resp),
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.wait_for_rate_limit", lambda: None
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.validate_eurlex_url", lambda url: True
        )

        result = download_case_html(
            "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62020CJ0790",
            timeout_secs=30,
            max_response_bytes=10_485_760,
            user_agent="TestAgent/1.0",
        )
        assert result == "<html>test</html>"

    def test_calls_rate_limiter(self, monkeypatch):
        from src.ingestion.case_law_engine import download_case_html

        rate_limit_called = []
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.wait_for_rate_limit",
            lambda: rate_limit_called.append(True),
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.validate_eurlex_url", lambda url: True
        )
        mock_resp = _make_mock_response("<html/>")
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.requests.get",
            MagicMock(return_value=mock_resp),
        )

        download_case_html(
            "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62020CJ0790",
            timeout_secs=30,
            max_response_bytes=10_485_760,
            user_agent="TestAgent/1.0",
        )
        assert len(rate_limit_called) == 1

    def test_uses_timeout(self, monkeypatch):
        from src.ingestion.case_law_engine import download_case_html

        monkeypatch.setattr(
            "src.ingestion.case_law_engine.wait_for_rate_limit", lambda: None
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.validate_eurlex_url", lambda url: True
        )
        mock_get = MagicMock(return_value=_make_mock_response("<html/>"))
        monkeypatch.setattr("src.ingestion.case_law_engine.requests.get", mock_get)

        download_case_html(
            "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62020CJ0790",
            timeout_secs=42,
            max_response_bytes=10_485_760,
            user_agent="TestAgent/1.0",
        )
        _, kwargs = mock_get.call_args
        assert kwargs["timeout"] == 42

    def test_http_error_raises(self, monkeypatch):
        import requests

        from src.ingestion.case_law_engine import download_case_html

        monkeypatch.setattr(
            "src.ingestion.case_law_engine.wait_for_rate_limit", lambda: None
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.validate_eurlex_url", lambda url: True
        )
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.RequestException("500 error")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.requests.get",
            MagicMock(return_value=mock_resp),
        )

        with pytest.raises(requests.RequestException):
            download_case_html(
                "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62020CJ0790",
                timeout_secs=30,
                max_response_bytes=10_485_760,
                user_agent="TestAgent/1.0",
            )

    def test_oversized_response_raises(self, monkeypatch):
        from src.ingestion.case_law_engine import download_case_html

        monkeypatch.setattr(
            "src.ingestion.case_law_engine.wait_for_rate_limit", lambda: None
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.validate_eurlex_url", lambda url: True
        )
        # Response with 100 bytes, but limit is 50
        mock_resp = _make_mock_response("x" * 100)
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.requests.get",
            MagicMock(return_value=mock_resp),
        )

        with pytest.raises(ValueError, match="exceeds"):
            download_case_html(
                "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62020CJ0790",
                timeout_secs=30,
                max_response_bytes=50,
                user_agent="TestAgent/1.0",
            )

    def test_non_eurlex_url_rejected(self, monkeypatch):
        from src.ingestion.eurlex_listing import EurLexSecurityError

        from src.ingestion.case_law_engine import download_case_html

        monkeypatch.setattr(
            "src.ingestion.case_law_engine.wait_for_rate_limit", lambda: None
        )
        # Don't mock validate_eurlex_url — let it actually validate
        with pytest.raises(EurLexSecurityError):
            download_case_html(
                "https://evil.com/malware",
                timeout_secs=30,
                max_response_bytes=10_485_760,
                user_agent="TestAgent/1.0",
            )


# ─────────────────────────────────────────────────────────────────────────────
# C1.5: compute_doc_version()
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeDocVersion:
    """T1.5.1–T1.5.3: Content-based doc version hash."""

    def test_returns_sha256(self):
        from src.ingestion.case_law_engine import compute_doc_version

        expected = hashlib.sha256("hello".encode("utf-8")).hexdigest()
        assert compute_doc_version("hello") == expected

    def test_same_content_same_hash(self):
        from src.ingestion.case_law_engine import compute_doc_version

        assert compute_doc_version("test") == compute_doc_version("test")

    def test_different_content_different_hash(self):
        from src.ingestion.case_law_engine import compute_doc_version

        assert compute_doc_version("a") != compute_doc_version("b")


# ─────────────────────────────────────────────────────────────────────────────
# C1.8: print_dry_run_table()
# ─────────────────────────────────────────────────────────────────────────────


class TestPrintDryRunTable:
    """T1.8.1–T1.8.3: Dry run table output."""

    def test_prints_table_with_columns(self, capsys):
        from src.ingestion.case_law_engine import print_dry_run_table

        cases = [
            _make_case(),
            _make_case(case_number="C-252/21", ecli="ECLI:EU:C:2023:537"),
        ]
        print_dry_run_table(cases)
        out = capsys.readouterr().out
        assert "C-311/18" in out
        assert "C-252/21" in out
        assert "ECLI:EU:C:2020:790" in out
        assert "Schrems II" in out

    def test_prints_total_count(self, capsys):
        from src.ingestion.case_law_engine import print_dry_run_table

        cases = [_make_case(), _make_case(), _make_case()]
        print_dry_run_table(cases)
        out = capsys.readouterr().out
        assert "3" in out

    def test_empty_list(self, capsys):
        from src.ingestion.case_law_engine import print_dry_run_table

        print_dry_run_table([])
        out = capsys.readouterr().out
        assert "0" in out


# ─────────────────────────────────────────────────────────────────────────────
# C1.9: print_summary()
# ─────────────────────────────────────────────────────────────────────────────


class TestPrintSummary:
    """T1.9.1–T1.9.3: Summary report output."""

    def test_prints_all_stats(self, capsys):
        from src.ingestion.case_law_engine import IngestionSummary, print_summary

        summary = IngestionSummary(
            cases_discovered=5,
            cases_processed=3,
            cases_skipped=2,
            total_chunks=45,
            enriched_chunks=40,
            enrichment_enabled=True,
            output_dir="/tmp/out",
            elapsed_secs=12.5,
        )
        print_summary(summary)
        out = capsys.readouterr().out
        assert "5" in out  # discovered
        assert "3" in out  # processed
        assert "2" in out  # skipped
        assert "45" in out  # chunks
        assert "88" in out  # enrichment rate ~88.9%
        assert "/tmp/out" in out
        assert "12" in out  # elapsed time

    def test_zero_chunks_no_division_error(self, capsys):
        from src.ingestion.case_law_engine import IngestionSummary, print_summary

        summary = IngestionSummary(
            cases_discovered=1,
            cases_processed=1,
            cases_skipped=0,
            total_chunks=0,
            enriched_chunks=0,
            enrichment_enabled=True,
            output_dir="/tmp/out",
            elapsed_secs=1.0,
        )
        print_summary(summary)  # Should not raise
        out = capsys.readouterr().out
        assert "0" in out

    def test_enrichment_disabled_shows_na(self, capsys):
        from src.ingestion.case_law_engine import IngestionSummary, print_summary

        summary = IngestionSummary(
            cases_discovered=1,
            cases_processed=1,
            total_chunks=10,
            enrichment_enabled=False,
            output_dir="/tmp/out",
            elapsed_secs=1.0,
        )
        print_summary(summary)
        out = capsys.readouterr().out
        assert "N/A" in out


# ─────────────────────────────────────────────────────────────────────────────
# C1.6: process_single_case()
# C1.7: enrich_case_law_rows()
# C1.10: run_ingestion() — integration tests
# ─────────────────────────────────────────────────────────────────────────────


class TestProcessSingleCase:
    """T1.6.1–T1.6.8: Single-case pipeline orchestration."""

    def _setup_mocks(
        self,
        monkeypatch,
        tmp_path,
        *,
        url="https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62020CJ0790",
        html=SAMPLE_HTML,
        enrichment_enabled=False,
    ):
        """Set up standard mocks for process_single_case tests."""
        settings = _make_settings(enabled=True)
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.load_settings", lambda: settings
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.wait_for_rate_limit", lambda: None
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.fetch_case_document_url",
            lambda celex_id: url,
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.validate_eurlex_url", lambda u: True
        )
        mock_resp = _make_mock_response(html)
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.requests.get",
            MagicMock(return_value=mock_resp),
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.is_enrichment_enabled",
            lambda: enrichment_enabled,
        )
        if enrichment_enabled:
            from src.ingestion.ingestion_generation import EnrichmentResult

            monkeypatch.setattr(
                "src.ingestion.case_law_engine.generate_enrichment",
                lambda *a, **kw: EnrichmentResult(
                    contextual_description="test desc",
                    search_terms=["term1"],
                    roles=["interpretation"],
                ),
            )

    def test_success_returns_case_result(self, monkeypatch, tmp_path):
        from src.ingestion.case_law_engine import process_single_case

        self._setup_mocks(monkeypatch, tmp_path)
        case = _make_case()
        result = process_single_case(
            case,

            corpus_id="gdpr",
            out_dir=tmp_path,
            language="en",
            corpus_celex_mapping=CORPUS_CELEX_MAPPING,
            celex_document_type_mapping=CELEX_DOC_TYPE_MAPPING,
            timeout_secs=30,
            max_response_bytes=10_485_760,
            user_agent="TestAgent/1.0",
        )
        assert result is not None
        assert result.celex_id == case.celex_id
        assert result.ecli == case.ecli
        assert result.chunk_count > 0

    def test_no_url_returns_none(self, monkeypatch, tmp_path, caplog):
        from src.ingestion.case_law_engine import process_single_case

        self._setup_mocks(monkeypatch, tmp_path, url=None)
        # Override to return None
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.fetch_case_document_url",
            lambda celex_id: None,
        )
        case = _make_case()
        with caplog.at_level(logging.WARNING):
            result = process_single_case(
                case,
    
                corpus_id="gdpr",
                out_dir=tmp_path,
                language="en",
                corpus_celex_mapping=CORPUS_CELEX_MAPPING,
                celex_document_type_mapping=CELEX_DOC_TYPE_MAPPING,
                timeout_secs=30,
                max_response_bytes=10_485_760,
                user_agent="TestAgent/1.0",
            )
        assert result is None
        assert "no_url" in caplog.text.lower() or "No HTML URL" in caplog.text

    def test_download_fails_returns_none(self, monkeypatch, tmp_path, caplog):
        import requests

        from src.ingestion.case_law_engine import process_single_case

        self._setup_mocks(monkeypatch, tmp_path)
        # Override HTTP to fail
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.RequestException("fail")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.requests.get",
            MagicMock(return_value=mock_resp),
        )
        case = _make_case()
        with caplog.at_level(logging.WARNING):
            result = process_single_case(
                case,
    
                corpus_id="gdpr",
                out_dir=tmp_path,
                language="en",
                corpus_celex_mapping=CORPUS_CELEX_MAPPING,
                celex_document_type_mapping=CELEX_DOC_TYPE_MAPPING,
                timeout_secs=30,
                max_response_bytes=10_485_760,
                user_agent="TestAgent/1.0",
            )
        assert result is None

    def test_empty_title_uses_case_number(self, monkeypatch, tmp_path):
        from src.ingestion.case_law_engine import process_single_case

        self._setup_mocks(monkeypatch, tmp_path)
        case = _make_case(title="")
        result = process_single_case(
            case,
            corpus_id="gdpr",
            out_dir=tmp_path,
            language="en",
            corpus_celex_mapping=CORPUS_CELEX_MAPPING,
            celex_document_type_mapping=CELEX_DOC_TYPE_MAPPING,
            timeout_secs=30,
            max_response_bytes=10_485_760,
            user_agent="TestAgent/1.0",
        )
        assert result is not None

        # Verify fallback: case_name in metadata equals case_number
        jsonl_path = tmp_path / "gdpr" / f"{case.celex_id}_chunks.jsonl"
        row = json.loads(jsonl_path.read_text().strip().split("\n")[0])
        assert row["metadata"]["case_name"] == case.case_number

    def test_writes_jsonl(self, monkeypatch, tmp_path):
        from src.ingestion.case_law_engine import process_single_case

        self._setup_mocks(monkeypatch, tmp_path)
        case = _make_case()
        process_single_case(
            case,

            corpus_id="gdpr",
            out_dir=tmp_path,
            language="en",
            corpus_celex_mapping=CORPUS_CELEX_MAPPING,
            celex_document_type_mapping=CELEX_DOC_TYPE_MAPPING,
            timeout_secs=30,
            max_response_bytes=10_485_760,
            user_agent="TestAgent/1.0",
        )
        jsonl_path = tmp_path / "gdpr" / f"{case.celex_id}_chunks.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) > 0
        row = json.loads(lines[0])
        assert "text" in row
        assert "metadata" in row

    def test_enrichment_applied_when_enabled(self, monkeypatch, tmp_path):
        from src.ingestion.case_law_engine import process_single_case

        self._setup_mocks(monkeypatch, tmp_path, enrichment_enabled=True)
        case = _make_case()
        result = process_single_case(
            case,
            corpus_id="gdpr",
            out_dir=tmp_path,
            language="en",
            corpus_celex_mapping=CORPUS_CELEX_MAPPING,
            celex_document_type_mapping=CELEX_DOC_TYPE_MAPPING,
            timeout_secs=30,
            max_response_bytes=10_485_760,
            user_agent="TestAgent/1.0",
        )
        assert result is not None
        assert result.chunk_count > 0

        # Verify enrichment metadata actually written to JSONL
        jsonl_path = tmp_path / "gdpr" / f"{case.celex_id}_chunks.jsonl"
        assert jsonl_path.exists()
        row = json.loads(jsonl_path.read_text().strip().split("\n")[0])
        meta = row["metadata"]
        assert "enrichment_terms" in meta
        assert "contextual_description" in meta

    def test_enrichment_skipped_when_disabled(self, monkeypatch, tmp_path):
        from src.ingestion.case_law_engine import process_single_case

        self._setup_mocks(monkeypatch, tmp_path, enrichment_enabled=False)
        case = _make_case()
        result = process_single_case(
            case,

            corpus_id="gdpr",
            out_dir=tmp_path,
            language="en",
            corpus_celex_mapping=CORPUS_CELEX_MAPPING,
            celex_document_type_mapping=CELEX_DOC_TYPE_MAPPING,
            timeout_secs=30,
            max_response_bytes=10_485_760,
            user_agent="TestAgent/1.0",
        )
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# C1.7: enrich_case_law_rows()
# ─────────────────────────────────────────────────────────────────────────────


class TestEnrichCaseLawRows:
    """T1.7.1–T1.7.3: Enrichment of case law chunk rows."""

    def test_calls_generate_enrichment_with_correct_args(self, monkeypatch):
        """T1.7.1: generate_enrichment called with correct metadata args."""
        from src.ingestion.case_law_engine import enrich_case_law_rows
        from src.ingestion.ingestion_generation import EnrichmentResult

        monkeypatch.setattr(
            "src.ingestion.case_law_engine.is_enrichment_enabled", lambda: True
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.get_max_concurrent", lambda: 1
        )

        call_args: list[dict] = []

        def spy_enrichment(text, *, article_title, corpus_id, metadata):
            call_args.append({
                "text": text,
                "article_title": article_title,
                "corpus_id": corpus_id,
                "metadata": metadata,
            })
            return EnrichmentResult(
                contextual_description="desc",
                search_terms=["t1"],
                roles=["interpretation"],
            )

        monkeypatch.setattr(
            "src.ingestion.case_law_engine.generate_enrichment", spy_enrichment
        )

        rows = [
            {
                "text": "chunk text",
                "metadata": {
                    "case_name": "Test Case",
                    "case_number": "C-100/20",
                },
            }
        ]
        enrich_case_law_rows(rows, corpus_id="gdpr")

        assert len(call_args) == 1
        assert call_args[0]["text"] == "chunk text"
        assert call_args[0]["article_title"] == "Test Case (C-100/20)"
        assert call_args[0]["corpus_id"] == "gdpr"

    def test_returns_rows_with_enrichment_metadata(self, monkeypatch):
        """T1.7.2: Enriched rows contain enrichment_terms and contextual_description."""
        from src.ingestion.case_law_engine import enrich_case_law_rows
        from src.ingestion.ingestion_generation import EnrichmentResult

        monkeypatch.setattr(
            "src.ingestion.case_law_engine.is_enrichment_enabled", lambda: True
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.get_max_concurrent", lambda: 1
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.generate_enrichment",
            lambda *a, **kw: EnrichmentResult(
                contextual_description="enriched desc",
                search_terms=["term1", "term2"],
                roles=["application"],
            ),
        )

        rows = [{"text": "chunk", "metadata": {"case_name": "X", "case_number": "C-1/20"}}]
        result, enriched_count = enrich_case_law_rows(rows, corpus_id="gdpr")

        assert enriched_count == 1
        assert "enrichment_terms" in result[0]["metadata"]
        assert "contextual_description" in result[0]["metadata"]
        assert result[0]["metadata"]["enrichment_terms"] == ["term1", "term2"]
        assert result[0]["metadata"]["contextual_description"] == "enriched desc"

    def test_returns_rows_unchanged_when_enrichment_disabled(self, monkeypatch):
        """T1.7.3: Rows returned unchanged when enrichment is disabled."""
        from src.ingestion.case_law_engine import enrich_case_law_rows

        monkeypatch.setattr(
            "src.ingestion.case_law_engine.is_enrichment_enabled", lambda: False
        )

        rows = [{"text": "chunk", "metadata": {"case_name": "X"}}]
        result, enriched_count = enrich_case_law_rows(rows, corpus_id="gdpr")

        assert enriched_count == 0
        assert result == rows
        assert "enrichment_terms" not in result[0]["metadata"]


# ─────────────────────────────────────────────────────────────────────────────
# C1.10: run_ingestion() — Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunIngestion:
    """T1.10.1–T1.10.15: End-to-end integration tests."""

    def _setup_full_mocks(
        self,
        monkeypatch,
        tmp_path,
        *,
        cases: list[CaseMetadata] | None = None,
        enabled: bool = True,
        enrichment_enabled: bool = False,
        download_fail_indices: set[int] | None = None,
    ):
        """Set up all mocks for run_ingestion() integration tests."""
        if cases is None:
            cases = [_make_case()]

        settings = _make_settings(enabled=enabled)
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.load_settings", lambda: settings
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.wait_for_rate_limit", lambda: None
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.validate_eurlex_url", lambda u: True
        )

        from src.common.config_loader import clear_config_cache, CaseLawParserSettings

        clear_config_cache()

        # Mock get_case_law_settings to return proper parser settings
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.get_case_law_settings",
            lambda: CaseLawParserSettings(
                section_heading_patterns={},
                article_reference_patterns=(),
                corpus_celex_mapping=CORPUS_CELEX_MAPPING,
                celex_document_type_mapping=CELEX_DOC_TYPE_MAPPING,
            ),
        )

        monkeypatch.setattr(
            "src.ingestion.case_law_engine.fetch_cases_for_legislation",
            lambda celex_id: cases,
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.fetch_case_document_url",
            lambda celex_id: (
                f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex_id}"
            ),
        )

        download_fail_indices = download_fail_indices or set()
        call_count = {"n": 0}

        def mock_get(url, **kwargs):
            import requests

            idx = call_count["n"]
            call_count["n"] += 1
            if idx in download_fail_indices:
                resp = MagicMock()
                resp.raise_for_status.side_effect = requests.RequestException("fail")
                resp.__enter__ = MagicMock(return_value=resp)
                resp.__exit__ = MagicMock(return_value=False)
                return resp
            return _make_mock_response(SAMPLE_HTML)

        monkeypatch.setattr("src.ingestion.case_law_engine.requests.get", mock_get)

        monkeypatch.setattr(
            "src.ingestion.case_law_engine.is_enrichment_enabled",
            lambda: enrichment_enabled,
        )
        if enrichment_enabled:
            from src.ingestion.ingestion_generation import EnrichmentResult

            monkeypatch.setattr(
                "src.ingestion.case_law_engine.generate_enrichment",
                lambda *a, **kw: EnrichmentResult(
                    contextual_description="test",
                    search_terms=["term"],
                    roles=["interpretation"],
                ),
            )

        return settings

    def test_as1_dry_run(self, monkeypatch, tmp_path, capsys):
        """AS-1: Dry run prints table, no files written."""
        cases = [
            _make_case(case_number="C-311/18"),
            _make_case(
                case_number="C-252/21",
                celex_id="62021CJ0252",
                ecli="ECLI:EU:C:2023:537",
            ),
            _make_case(
                case_number="C-175/20",
                celex_id="62020CJ0175",
                ecli="ECLI:EU:C:2022:124",
            ),
        ]
        self._setup_full_mocks(monkeypatch, tmp_path, cases=cases)
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "gdpr", "--dry-run", "--out-dir", str(tmp_path)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        run_ingestion()
        out = capsys.readouterr().out
        assert "C-311/18" in out
        assert "C-252/21" in out
        assert "C-175/20" in out
        # No JSONL files
        assert not list(tmp_path.glob("**/*.jsonl"))

    def test_as2_full_pipeline(self, monkeypatch, tmp_path, capsys):
        """AS-2: Full pipeline produces JSONL with correct metadata."""
        self._setup_full_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "gdpr", "--out-dir", str(tmp_path)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        run_ingestion()
        jsonl_files = list(tmp_path.glob("gdpr/*.jsonl"))
        assert len(jsonl_files) >= 1
        row = json.loads(jsonl_files[0].read_text().strip().split("\n")[0])
        assert row["metadata"]["source_type"] == "cjeu_case_law"

    def test_as3_skip_on_failure(self, monkeypatch, tmp_path, caplog):
        """AS-3: Second case fails, others succeed."""
        cases = [
            _make_case(
                celex_id="62020CJ0001", ecli="ECLI:EU:C:2020:1", case_number="C-1/20"
            ),
            _make_case(
                celex_id="62020CJ0002", ecli="ECLI:EU:C:2020:2", case_number="C-2/20"
            ),
            _make_case(
                celex_id="62020CJ0003", ecli="ECLI:EU:C:2020:3", case_number="C-3/20"
            ),
        ]
        self._setup_full_mocks(
            monkeypatch, tmp_path, cases=cases, download_fail_indices={1}
        )
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "gdpr", "--out-dir", str(tmp_path)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        with caplog.at_level(logging.WARNING):
            run_ingestion()

        jsonl_files = list(tmp_path.glob("gdpr/*.jsonl"))
        assert len(jsonl_files) == 2  # Cases 1 and 3

    def test_as4_summary_report(self, monkeypatch, tmp_path, capsys):
        """AS-4: Summary report has all stats."""
        self._setup_full_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "gdpr", "--out-dir", str(tmp_path)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        run_ingestion()
        out = capsys.readouterr().out
        # Summary must contain all stat labels
        assert "Cases discovered" in out
        assert "Cases processed" in out
        assert "Cases skipped" in out
        assert "Total chunks" in out

    def test_as5_feature_flag_disabled(self, monkeypatch, tmp_path, capsys):
        """AS-5: Feature flag disabled prints message, no CELLAR call."""
        self._setup_full_mocks(monkeypatch, tmp_path, enabled=False)

        fetch_called = []
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.fetch_cases_for_legislation",
            lambda celex_id: fetch_called.append(True) or [],
        )
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "gdpr", "--out-dir", str(tmp_path)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        run_ingestion()
        out = capsys.readouterr().out
        assert "disabled" in out.lower()
        assert len(fetch_called) == 0

    def test_as6_unknown_corpus(self, monkeypatch, tmp_path):
        """AS-6: Unknown corpus exits with error listing valid IDs."""
        self._setup_full_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "unknown", "--out-dir", str(tmp_path)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        with pytest.raises(SystemExit) as exc_info:
            run_ingestion()
        assert exc_info.value.code != 0

    def test_as7_limit_flag(self, monkeypatch, tmp_path):
        """AS-7: --limit 3 processes only 3 of 10 cases."""
        cases = [
            _make_case(
                celex_id=f"62020CJ{i:04d}",
                ecli=f"ECLI:EU:C:2020:{i}",
                case_number=f"C-{i}/20",
            )
            for i in range(10)
        ]
        self._setup_full_mocks(monkeypatch, tmp_path, cases=cases)
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "gdpr", "--limit", "3", "--out-dir", str(tmp_path)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        run_ingestion()
        jsonl_files = list(tmp_path.glob("gdpr/*.jsonl"))
        assert len(jsonl_files) == 3

    def test_as8_enrichment_disabled(self, monkeypatch, tmp_path, capsys):
        """AS-8: Enrichment disabled shows N/A in summary."""
        self._setup_full_mocks(monkeypatch, tmp_path, enrichment_enabled=False)
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "gdpr", "--out-dir", str(tmp_path)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        run_ingestion()
        out = capsys.readouterr().out
        assert "N/A" in out

    def test_as9_empty_results(self, monkeypatch, tmp_path, capsys):
        """AS-9: No cases found prints message, exits 0."""
        self._setup_full_mocks(monkeypatch, tmp_path, cases=[])
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "gdpr", "--out-dir", str(tmp_path)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        run_ingestion()
        out = capsys.readouterr().out
        assert "No cases found" in out or "no cases" in out.lower()

    def test_all_cases_fail(self, monkeypatch, tmp_path, capsys):
        """Edge 7: All cases fail, CLI completes with 0 processed."""
        cases = [
            _make_case(
                celex_id=f"62020CJ{i:04d}",
                ecli=f"ECLI:EU:C:2020:{i}",
                case_number=f"C-{i}/20",
            )
            for i in range(3)
        ]
        self._setup_full_mocks(
            monkeypatch, tmp_path, cases=cases, download_fail_indices={0, 1, 2}
        )
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "gdpr", "--out-dir", str(tmp_path)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        run_ingestion()  # Should not raise
        capsys.readouterr()  # drain captured output
        jsonl_files = list(tmp_path.glob("gdpr/*.jsonl"))
        assert len(jsonl_files) == 0

    def test_keyboard_interrupt_reraised(self, monkeypatch, tmp_path):
        """Edge 10: KeyboardInterrupt is re-raised."""
        self._setup_full_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.fetch_cases_for_legislation",
            MagicMock(side_effect=KeyboardInterrupt),
        )
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "gdpr", "--out-dir", str(tmp_path)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        with pytest.raises(KeyboardInterrupt):
            run_ingestion()

    def test_output_dir_created(self, monkeypatch, tmp_path):
        """REQ-8: Output directory created automatically."""
        out_dir = tmp_path / "nested" / "output"
        self._setup_full_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "gdpr", "--out-dir", str(out_dir)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        run_ingestion()
        assert (out_dir / "gdpr").is_dir()

    def test_discovered_count_is_pre_limit(self, monkeypatch, tmp_path, capsys):
        """Issue #1: cases_discovered reports total from CELLAR, not post-limit."""
        cases = [
            _make_case(
                celex_id=f"62020CJ{i:04d}",
                ecli=f"ECLI:EU:C:2020:{i}",
                case_number=f"C-{i}/20",
            )
            for i in range(10)
        ]
        self._setup_full_mocks(monkeypatch, tmp_path, cases=cases)
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "gdpr", "--limit", "3", "--out-dir", str(tmp_path)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        run_ingestion()
        out = capsys.readouterr().out
        # Summary should show 10 discovered (pre-limit), not 3
        assert "10" in out  # cases_discovered = 10

    def test_oserror_caught_per_case(self, monkeypatch, tmp_path, caplog):
        """Issue #2: OSError in per-case exception handler doesn't crash pipeline."""
        self._setup_full_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.process_single_case",
            MagicMock(side_effect=OSError("disk full")),
        )
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "gdpr", "--out-dir", str(tmp_path)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        with caplog.at_level(logging.WARNING):
            run_ingestion()  # Should not crash
        assert "disk full" in caplog.text

    @pytest.mark.parametrize(
        "exc_type,exc_msg",
        [
            (requests.RequestException, "connection error"),
            (ValueError, "bad value"),
            (OSError, "disk full"),
        ],
        ids=["RequestException", "ValueError", "OSError"],
    )
    def test_exception_types_caught_per_case(
        self, monkeypatch, tmp_path, caplog, exc_type, exc_msg
    ):
        """T1.10.12: Various exception types caught per-case without crash."""
        self._setup_full_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.process_single_case",
            MagicMock(side_effect=exc_type(exc_msg)),
        )
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "gdpr", "--out-dir", str(tmp_path)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        with caplog.at_level(logging.WARNING):
            run_ingestion()  # Should not raise
        assert exc_msg in caplog.text

    def test_cellar_query_error_caught_per_case(
        self, monkeypatch, tmp_path, caplog
    ):
        """T1.10.12 supplement: CellarQueryError caught per-case."""
        from src.ingestion.cellar_models import CellarQueryError

        self._setup_full_mocks(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.process_single_case",
            MagicMock(side_effect=CellarQueryError("sparql fail")),
        )
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "gdpr", "--out-dir", str(tmp_path)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        with caplog.at_level(logging.WARNING):
            run_ingestion()  # Should not raise
        assert "sparql fail" in caplog.text

    def test_enrichment_none_chunk_still_written(self, monkeypatch, tmp_path):
        """T1.10.13: generate_enrichment returns None — chunk written without enrichment."""
        self._setup_full_mocks(monkeypatch, tmp_path, enrichment_enabled=True)
        # Override enrichment to return None (enrichment failure for chunk)
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.generate_enrichment",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "gdpr", "--out-dir", str(tmp_path)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        run_ingestion()
        jsonl_files = list(tmp_path.glob("gdpr/*.jsonl"))
        assert len(jsonl_files) >= 1
        row = json.loads(jsonl_files[0].read_text().strip().split("\n")[0])
        # Chunk still present, but no enrichment metadata
        assert "text" in row
        assert "enrichment_terms" not in row.get("metadata", {})

    def test_logging_pipeline_start_and_discovery(self, monkeypatch, tmp_path, caplog):
        """T1.10.15: INFO log for pipeline start and case discovery."""
        cases = [
            _make_case(),
            _make_case(celex_id="62021CJ0252", ecli="ECLI:EU:C:2023:537"),
        ]
        self._setup_full_mocks(monkeypatch, tmp_path, cases=cases)
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "gdpr", "--out-dir", str(tmp_path)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        with caplog.at_level(logging.INFO, logger="src.ingestion.case_law_engine"):
            run_ingestion()
        assert "Starting case law ingestion" in caplog.text
        assert "Discovered 2 cases" in caplog.text

    def test_logging_failure_warning(self, monkeypatch, tmp_path, caplog):
        """T1.10.15: WARNING log for failure cases."""
        cases = [_make_case()]
        self._setup_full_mocks(
            monkeypatch, tmp_path, cases=cases, download_fail_indices={0}
        )
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--corpus", "gdpr", "--out-dir", str(tmp_path)],
        )

        from src.ingestion.case_law_engine import run_ingestion

        with caplog.at_level(logging.WARNING, logger="src.ingestion.case_law_engine"):
            run_ingestion()
        assert "download_error" in caplog.text


# ─────────────────────────────────────────────────────────────────────────────
# Security: Path traversal (Issue #5)
# ─────────────────────────────────────────────────────────────────────────────


class TestPathTraversalSecurity:
    """Security tests for path traversal via CELEX ID."""

    def _setup_mocks(self, monkeypatch, tmp_path, *, celex_id="62020CJ0790"):
        settings = _make_settings(enabled=True)
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.load_settings", lambda: settings
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.wait_for_rate_limit", lambda: None
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.fetch_case_document_url",
            lambda cid: f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{cid}",
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.validate_eurlex_url", lambda u: True
        )
        mock_resp = _make_mock_response(SAMPLE_HTML)
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.requests.get",
            MagicMock(return_value=mock_resp),
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.is_enrichment_enabled", lambda: False
        )

    def test_path_traversal_celex_rejected(self, monkeypatch, tmp_path, caplog):
        """CELEX ID with ../ must not write outside output dir."""
        from src.ingestion.case_law_engine import process_single_case

        self._setup_mocks(monkeypatch, tmp_path)
        case = _make_case(celex_id="../../../etc/passwd")

        with caplog.at_level(logging.WARNING):
            result = process_single_case(
                case,
                corpus_id="gdpr",
                out_dir=tmp_path,
                language="en",
                corpus_celex_mapping=CORPUS_CELEX_MAPPING,
                celex_document_type_mapping=CELEX_DOC_TYPE_MAPPING,
                timeout_secs=30,
                max_response_bytes=10_485_760,
                user_agent="TestAgent/1.0",
            )
        assert result is None
        assert "path_traversal" in caplog.text.lower() or "escapes" in caplog.text.lower()

    def test_path_traversal_output_stays_within_dir(self, monkeypatch, tmp_path):
        """Normal CELEX ID resolves within output dir."""
        from src.ingestion.case_law_engine import process_single_case

        self._setup_mocks(monkeypatch, tmp_path)
        case = _make_case(celex_id="62020CJ0790")
        result = process_single_case(
            case,
            corpus_id="gdpr",
            out_dir=tmp_path,
            language="en",
            corpus_celex_mapping=CORPUS_CELEX_MAPPING,
            celex_document_type_mapping=CELEX_DOC_TYPE_MAPPING,
            timeout_secs=30,
            max_response_bytes=10_485_760,
            user_agent="TestAgent/1.0",
        )
        assert result is not None
        output_path = tmp_path / "gdpr" / "62020CJ0790_chunks.jsonl"
        assert output_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case 6: zero chunks from parser
# ─────────────────────────────────────────────────────────────────────────────


class TestZeroChunks:
    """Edge 6: Parser returns judgment that produces zero chunks."""

    def test_zero_chunks_returns_none_with_warning(self, monkeypatch, tmp_path, caplog):
        from src.ingestion.case_law_engine import process_single_case

        settings = _make_settings(enabled=True)
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.load_settings", lambda: settings
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.wait_for_rate_limit", lambda: None
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.fetch_case_document_url",
            lambda cid: f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{cid}",
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.validate_eurlex_url", lambda u: True
        )
        mock_resp = _make_mock_response(SAMPLE_HTML)
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.requests.get",
            MagicMock(return_value=mock_resp),
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.is_enrichment_enabled", lambda: False
        )
        # Mock chunk_judgment to return empty list
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.chunk_judgment", lambda *a, **kw: []
        )

        case = _make_case()
        with caplog.at_level(logging.WARNING):
            result = process_single_case(
                case,
                corpus_id="gdpr",
                out_dir=tmp_path,
                language="en",
                corpus_celex_mapping=CORPUS_CELEX_MAPPING,
                celex_document_type_mapping=CELEX_DOC_TYPE_MAPPING,
                timeout_secs=30,
                max_response_bytes=10_485_760,
                user_agent="TestAgent/1.0",
            )
        assert result is None
        assert "no_chunks" in caplog.text


# ─────────────────────────────────────────────────────────────────────────────
# Issue #14: download_case_html user_agent parameter
# ─────────────────────────────────────────────────────────────────────────────


class TestDownloadCaseHtmlUserAgent:
    """Tests for user_agent parameter on download_case_html."""

    def test_user_agent_passed_to_request(self, monkeypatch):
        from src.ingestion.case_law_engine import download_case_html

        monkeypatch.setattr(
            "src.ingestion.case_law_engine.wait_for_rate_limit", lambda: None
        )
        monkeypatch.setattr(
            "src.ingestion.case_law_engine.validate_eurlex_url", lambda url: True
        )
        mock_get = MagicMock(return_value=_make_mock_response("<html/>"))
        monkeypatch.setattr("src.ingestion.case_law_engine.requests.get", mock_get)

        download_case_html(
            "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62020CJ0790",
            timeout_secs=30,
            max_response_bytes=10_485_760,
            user_agent="CustomAgent/2.0",
        )
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["User-Agent"] == "CustomAgent/2.0"


# ─────────────────────────────────────────────────────────────────────────────
# Issue #15/16: Config-driven defaults
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigDrivenDefaults:
    """Tests for config-driven default_language and out-dir."""

    def test_default_language_from_config(self):
        """Issue #16: default_language loaded from settings."""
        from src.common.config_loader import CaseLawSettings

        # Default language is configured in settings, not hardcoded in argparse
        assert CaseLawSettings.default_language == "en"
        # parse_args returns None when --language not specified (resolved from config later)
        from src.ingestion.case_law_engine import parse_args

        args = parse_args(["--corpus", "gdpr"])
        assert args.language is None  # will be resolved from settings in run_ingestion
