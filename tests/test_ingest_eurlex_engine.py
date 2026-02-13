import json
import pytest

from src.ingestion.eurlex_engine import run_ingestion_for_file


@pytest.mark.slow
def test_ingest_eurlex_engine_writes_chunks_jsonl(tmp_path):
    html = """
    <html><body>
      <p>Kapitel I</p>
      <p>Generelle bestemmelser</p>
      <p>Artikel 1</p>
      <p>Genstand og formål</p>
      <p>1. First paragraph.</p>
      <p>a) First litra</p>
    </body></html>
    """

    html_path = tmp_path / "GDPR.html"
    html_path.write_text(html, encoding="utf-8")

    out_dir = tmp_path / "processed"
    out = run_ingestion_for_file(
        corpus_id="gdpr",
        html_path=html_path,
        out_dir=out_dir,
        chunk_tokens=200,
        overlap=0,
    )

    lines = out.output_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines

    first = json.loads(lines[0])
    assert first["metadata"].get("corpus_id") == "gdpr"
    assert first["metadata"].get("chunk_id")
    assert first["metadata"].get("location_id")
