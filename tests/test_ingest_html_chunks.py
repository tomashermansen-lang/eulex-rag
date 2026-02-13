import json
import re

from src.ingestion.html_chunks import HtmlChunkingConfig, chunk_html


def test_chunk_html_uses_heading_state_and_paragraph_boundaries():
    html = """
    <html><body>
      <h2>Kapitel I</h2>
      <h3>Artikel 1</h3>
      <p>Første afsnit.</p>
    <p>Andet afsnit. Se artikel 12, stk. 3.</p>
      <h3>Artikel 2</h3>
      <p>Tredje afsnit.</p>
    </body></html>
    """

    patterns = {
        "chapter": re.compile(r"(?i)kapitel\s+([ivxlcdm]+)"),
        "article": re.compile(r"(?i)artikel\s+(\d{1,3})"),
    }

    rows = list(
        chunk_html(
            html,
            config=HtmlChunkingConfig(chunk_tokens=1000, overlap=0),
            base_metadata={"source": "GDPR", "corpus_id": "gdpr"},
            reference_patterns=patterns,
            initial_reference_state={"chapter": None, "article": None},
            reset_on={"article": ()},
        )
    )

    assert len(rows) >= 2
    assert rows[0]["metadata"]["chapter"] == "I"
    assert rows[0]["metadata"]["article"] == "1"
    assert "Første afsnit" in rows[0]["text"]

    # Mentions should include cross-references found in text blocks.
    mentions_raw = rows[0]["metadata"].get("mentions")
    assert isinstance(mentions_raw, str) and mentions_raw
    mentions = json.loads(mentions_raw)
    assert "article" in mentions
    assert "12" in mentions["article"]
    assert "paragraph" in mentions
    assert "3" in mentions["paragraph"]

    # After Artikel 2 heading, metadata should update.
    assert rows[-1]["metadata"]["article"] == "2"


def test_chunk_html_splits_large_paragraph_without_midword_splits():
    html = "<html><body><p>" + ("ord " * 400) + "</p></body></html>"

    rows = list(
        chunk_html(
            html,
            config=HtmlChunkingConfig(chunk_tokens=80, overlap=0),
            base_metadata={"source": "X", "corpus_id": "x"},
        )
    )

    assert len(rows) > 1
    # Should not contain obvious mid-word splits from our side.
    for row in rows:
        assert "  " not in row["text"]
        assert row["text"].strip()


def test_chunk_html_extracts_paragraph_location_from_numbered_paragraphs_when_flushing_each_block():
    html = """
    <html><body>
      <h3>Artikel 10</h3>
      <p>1. Første paragraf.</p>
      <p>2. Anden paragraf.</p>
    </body></html>
    """

    patterns = {"article": re.compile(r"(?i)artikel\s*(\d{1,3})")}
    inline = {"paragraph": re.compile(r"^\s*(?:(\d{1,3})\.\s+|\((\d{1,3})\)\s+)")}

    rows = list(
        chunk_html(
            html,
            config=HtmlChunkingConfig(chunk_tokens=1000, overlap=0, flush_each_text_block=True),
            base_metadata={"source": "GDPR", "corpus_id": "gdpr"},
            reference_patterns=patterns,
            inline_location_patterns=inline,
            inline_location_requires={"paragraph": ("article",)},
            initial_reference_state={"article": None, "paragraph": None},
            reset_on={"article": ("paragraph",)},
        )
    )

    # Two <p> blocks -> two chunks.
    assert len(rows) == 2
    assert rows[0]["metadata"]["article"] == "10"
    assert rows[0]["metadata"]["paragraph"] == "1"
    assert rows[1]["metadata"]["paragraph"] == "2"


def test_chunk_html_promotes_p_heading_lines_to_structure():
    html = """
    <html><body>
      <p>Kapitel I</p>
      <p>Artikel 10</p>
      <p>1. Første paragraf.</p>
    </body></html>
    """

    patterns = {
        "chapter": re.compile(r"(?i)^\s*kapitel\s+([ivxlcdm]+)\b"),
        "article": re.compile(r"(?i)^\s*artikel\s*(\d{1,3})\b"),
    }

    rows = list(
        chunk_html(
            html,
            config=HtmlChunkingConfig(chunk_tokens=1000, overlap=0, flush_each_text_block=True),
            base_metadata={"source": "GDPR", "corpus_id": "gdpr"},
            reference_patterns=patterns,
            inline_location_patterns={"paragraph": re.compile(r"^\s*(\d{1,3})\.\s+")},
            inline_location_requires={"paragraph": ("article",)},
            initial_reference_state={"chapter": None, "article": None, "paragraph": None},
            reset_on={"article": ("paragraph",)},
        )
    )

    # Heading lines shouldn't become content chunks; the paragraph chunk should carry structure.
    assert len(rows) == 1
    assert rows[0]["metadata"]["chapter"] == "I"
    assert rows[0]["metadata"]["article"] == "10"
    assert rows[0]["metadata"]["paragraph"] == "1"


def test_chunk_html_eurlex_structural_div_ids_set_structure_when_enabled():
        html = """
        <html><body>
            <div id="cpt_I">
                <div id="cpt_I.sct_1">
                    <div id="art_5">
                        <p>Dette er en artikeltekst.</p>
                    </div>
                </div>
            </div>
        </body></html>
        """

        rows = list(
                chunk_html(
                        html,
                        config=HtmlChunkingConfig(chunk_tokens=1000, overlap=0, flush_each_text_block=True),
                        base_metadata={"source": "AI Act", "corpus_id": "ai-act"},
                        initial_reference_state={
                                "chapter": None,
                                "section": None,
                                "article": None,
                                "paragraph": None,
                                "litra": None,
                                "annex": None,
                        },
                        enable_eurlex_structural_ids=True,
                )
        )

        assert len(rows) == 1
        meta = rows[0]["metadata"]
        assert meta.get("chapter") == "I"
        assert meta.get("section") == "1"
        assert meta.get("article") == "5"
        # Canonical location_id should encode structure deterministically.
        assert "chapter:" in str(meta.get("location_id"))
        assert "section:" in str(meta.get("location_id"))
        assert "article:" in str(meta.get("location_id"))
        # We do not inject synthetic heading text into the chunk payload.
        assert "Chapter" not in rows[0]["text"]
