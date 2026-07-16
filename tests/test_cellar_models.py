"""Tests for cellar_models — frozen dataclasses and validation patterns."""

from __future__ import annotations

import pytest

from src.ingestion.cellar_models import (
    CASE_LAW_CELEX_PATTERN,
    ECLI_PATTERN,
    CaseDocumentRef,
    CaseMetadata,
    CellarQueryError,
)


class TestCaseMetadata:
    """REQ-4: Frozen CaseMetadata dataclass."""

    def test_construction(self) -> None:
        meta = CaseMetadata(
            ecli="ECLI:EU:C:2020:790",
            case_number="C-311/18",
            celex_id="62018CJ0311",
            date="2020-07-16",
            court="CJEU",
            articles_interpreted=["Art. 46", "Art. 49"],
            title="Schrems II",
        )
        assert meta.ecli == "ECLI:EU:C:2020:790"
        assert meta.case_number == "C-311/18"
        assert meta.celex_id == "62018CJ0311"
        assert meta.date == "2020-07-16"
        assert meta.court == "CJEU"
        assert meta.articles_interpreted == ["Art. 46", "Art. 49"]
        assert meta.title == "Schrems II"

    def test_frozen(self) -> None:
        meta = CaseMetadata(
            ecli="ECLI:EU:C:2020:790",
            case_number="C-311/18",
            celex_id="62018CJ0311",
            date="2020-07-16",
            court="CJEU",
            title="Schrems II",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            meta.ecli = "new"  # type: ignore[misc]

    def test_default_articles_interpreted(self) -> None:
        meta = CaseMetadata(
            ecli="ECLI:EU:C:2020:790",
            case_number="C-311/18",
            celex_id="62018CJ0311",
            date="2020-07-16",
            court="CJEU",
            title="Schrems II",
        )
        assert meta.articles_interpreted == []


class TestCaseDocumentRef:
    """REQ-4: Frozen CaseDocumentRef dataclass."""

    def test_construction(self) -> None:
        ref = CaseDocumentRef(
            celex_id="62020CJ0790",
            ecli="ECLI:EU:C:2020:790",
            html_url="https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62020CJ0790",
        )
        assert ref.celex_id == "62020CJ0790"
        assert ref.ecli == "ECLI:EU:C:2020:790"
        assert ref.html_url.startswith("https://eur-lex.europa.eu")

    def test_frozen(self) -> None:
        ref = CaseDocumentRef(
            celex_id="62020CJ0790",
            ecli="ECLI:EU:C:2020:790",
            html_url="https://example.com",
        )
        with pytest.raises(Exception):
            ref.celex_id = "new"  # type: ignore[misc]


class TestEcliPattern:
    """REQ-11, Edge 11, Edge 12: ECLI validation pattern."""

    @pytest.mark.parametrize(
        "ecli",
        [
            "ECLI:EU:C:2020:790",
            "ECLI:EU:T:2021:123",
            "ECLI:EU:C:2019:1",
        ],
    )
    def test_valid(self, ecli: str) -> None:
        assert ECLI_PATTERN.match(ecli) is not None

    def test_valid_with_dots(self) -> None:
        assert ECLI_PATTERN.match("ECLI:EU:C:2020:790.1") is not None

    @pytest.mark.parametrize(
        "ecli",
        [
            "not-an-ecli",
            "ECLI:EU:X:2020:790",  # invalid court
            "ECLI:EU:C:20:790",  # 2-digit year
            "",
        ],
    )
    def test_rejects_invalid(self, ecli: str) -> None:
        assert ECLI_PATTERN.match(ecli) is None

    def test_rejects_lowercase(self) -> None:
        # Normalization is the caller's responsibility, not the pattern's
        assert ECLI_PATTERN.match("ecli:eu:c:2020:790") is None


class TestCaseLawCelexPattern:
    """REQ-11: Case law CELEX validation pattern."""

    @pytest.mark.parametrize(
        "celex",
        [
            "62020CJ0790",  # CJ = Court of Justice
            "62020TJ0123",  # TJ = General Court
            "62018CJ0311",
            "62020C0790",  # single letter court code
        ],
    )
    def test_valid(self, celex: str) -> None:
        assert CASE_LAW_CELEX_PATTERN.match(celex) is not None

    @pytest.mark.parametrize(
        "celex",
        [
            "invalid",
            "",
            "6202XCJ0790",  # letter in year section
            "ABCDECJ0790",  # letters in digit section
        ],
    )
    def test_rejects_invalid(self, celex: str) -> None:
        assert CASE_LAW_CELEX_PATTERN.match(celex) is None


class TestCellarQueryError:
    """CellarQueryError is a proper Exception subclass."""

    def test_is_exception(self) -> None:
        assert issubclass(CellarQueryError, Exception)

    def test_message(self) -> None:
        err = CellarQueryError("SPARQL endpoint returned 503")
        assert "503" in str(err)
