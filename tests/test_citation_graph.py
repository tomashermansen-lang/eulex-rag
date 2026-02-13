"""Tests for src/ingestion/citation_graph.py - Citation graph building."""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.ingestion.citation_graph import (
    CitationEdge,
    CitationNode,
    CitationGraph,
    _detect_roles_from_text,
    load_citation_graph,
    get_or_build_citation_graph,
)


# ---------------------------------------------------------------------------
# Test: CitationEdge
# ---------------------------------------------------------------------------


class TestCitationEdge:
    def test_to_dict(self):
        edge = CitationEdge(
            source="6",
            target="7",
            edge_type="cites",
            count=3,
            weight=0.75,
        )
        d = edge.to_dict()

        assert d["source"] == "6"
        assert d["target"] == "7"
        assert d["type"] == "cites"
        assert d["count"] == 3
        assert d["weight"] == 0.75

    def test_from_dict(self):
        d = {"source": "6", "target": "7", "type": "cites", "count": 2, "weight": 0.5}
        edge = CitationEdge.from_dict(d)

        assert edge.source == "6"
        assert edge.target == "7"
        assert edge.edge_type == "cites"
        assert edge.count == 2
        assert edge.weight == 0.5

    def test_from_dict_defaults(self):
        d = {"source": "6", "target": "7", "type": "cites"}
        edge = CitationEdge.from_dict(d)

        assert edge.count == 1
        assert edge.weight == 1.0


# ---------------------------------------------------------------------------
# Test: CitationNode
# ---------------------------------------------------------------------------


class TestCitationNode:
    def test_to_dict(self):
        node = CitationNode(
            article_id="6",
            node_type="article",
            title="Prohibited practices",
            chapter="II",
            mention_count=5,
            roles=["scope", "classification"],
        )
        d = node.to_dict()

        assert d["type"] == "article"
        assert d["title"] == "Prohibited practices"
        assert d["chapter"] == "II"
        assert d["roles"] == ["scope", "classification"]
        assert d["mention_count"] == 5

    def test_to_dict_minimal(self):
        node = CitationNode(article_id="6", node_type="article")
        d = node.to_dict()

        assert d == {"type": "article"}
        assert "title" not in d
        assert "mention_count" not in d

    def test_from_dict(self):
        d = {
            "type": "article",
            "title": "Test",
            "chapter": "I",
            "mention_count": 3,
            "roles": ["definitions"],
        }
        node = CitationNode.from_dict("6", d)

        assert node.article_id == "6"
        assert node.node_type == "article"
        assert node.title == "Test"
        assert node.chapter == "I"
        assert node.roles == ["definitions"]


# ---------------------------------------------------------------------------
# Test: CitationGraph
# ---------------------------------------------------------------------------


class TestCitationGraph:
    def test_add_node_new(self):
        graph = CitationGraph(corpus_id="test")
        node = CitationNode(article_id="6", node_type="article", title="Test")

        graph.add_node(node)

        assert "6" in graph.nodes
        assert graph.nodes["6"].title == "Test"

    def test_add_node_merges_existing(self):
        graph = CitationGraph(corpus_id="test")
        node1 = CitationNode(article_id="6", mention_count=2)
        node2 = CitationNode(article_id="6", mention_count=3, title="Added Title")

        graph.add_node(node1)
        graph.add_node(node2)

        assert graph.nodes["6"].mention_count == 5
        assert graph.nodes["6"].title == "Added Title"

    def test_add_node_accumulates_roles(self):
        graph = CitationGraph(corpus_id="test")
        node1 = CitationNode(article_id="6", roles=["scope"])
        node2 = CitationNode(article_id="6", roles=["definitions"])

        graph.add_node(node1)
        graph.add_node(node2)

        assert "scope" in graph.nodes["6"].roles
        assert "definitions" in graph.nodes["6"].roles

    def test_add_edge_new(self):
        graph = CitationGraph(corpus_id="test")
        graph.add_edge("6", "7", edge_type="cites")

        assert len(graph.edges) == 1
        assert graph.edges[0].source == "6"
        assert graph.edges[0].target == "7"

    def test_add_edge_increments_existing(self):
        graph = CitationGraph(corpus_id="test")
        graph.add_edge("6", "7", edge_type="cites")
        graph.add_edge("6", "7", edge_type="cites")

        assert len(graph.edges) == 1
        assert graph.edges[0].count == 2

    def test_add_edge_skips_self_reference(self):
        graph = CitationGraph(corpus_id="test")
        graph.add_edge("6", "6", edge_type="cites")

        assert len(graph.edges) == 0

    def test_normalize_weights(self):
        graph = CitationGraph(corpus_id="test")
        graph.add_edge("6", "7", count=4)
        graph.add_edge("6", "8", count=2)

        graph.normalize_weights()

        # Edge with count 4 should have weight 1.0
        edge_7 = next(e for e in graph.edges if e.target == "7")
        edge_8 = next(e for e in graph.edges if e.target == "8")
        assert edge_7.weight == 1.0
        assert edge_8.weight == 0.5

    def test_get_related_articles(self):
        graph = CitationGraph(corpus_id="test")
        graph.add_edge("6", "7", edge_type="cites")
        graph.add_edge("6", "8", edge_type="cites")
        graph.normalize_weights()

        related = graph.get_related_articles("6")
        article_ids = [a for a, _ in related]

        assert "7" in article_ids
        assert "8" in article_ids

    def test_get_related_articles_respects_min_weight(self):
        graph = CitationGraph(corpus_id="test")
        graph.add_edge("6", "7", count=10)
        graph.add_edge("6", "8", count=1)
        graph.normalize_weights()

        related = graph.get_related_articles("6", min_weight=0.5)
        article_ids = [a for a, _ in related]

        assert "7" in article_ids
        assert "8" not in article_ids

    def test_get_related_articles_filters_edge_types(self):
        graph = CitationGraph(corpus_id="test")
        graph.add_edge("6", "7", edge_type="cites")
        graph.add_edge("6", "8", edge_type="co-occurrence")
        graph.normalize_weights()

        related = graph.get_related_articles("6", edge_types={"cites"})
        article_ids = [a for a, _ in related]

        assert "7" in article_ids
        assert "8" not in article_ids

    def test_get_articles_by_role(self):
        graph = CitationGraph(corpus_id="test")
        graph.add_node(CitationNode(article_id="2", roles=["scope"]))
        graph.add_node(CitationNode(article_id="3", roles=["definitions"]))
        graph.add_node(CitationNode(article_id="6", roles=["classification", "scope"]))

        scope_articles = graph.get_articles_by_role("scope")

        assert "2" in scope_articles
        assert "6" in scope_articles
        assert "3" not in scope_articles

    def test_get_foundational_articles(self):
        graph = CitationGraph(corpus_id="test")
        graph.add_node(CitationNode(article_id="2", roles=["scope"]))
        graph.add_node(CitationNode(article_id="3", roles=["definitions"]))

        foundational = graph.get_foundational_articles()

        assert "scope" in foundational
        assert "definitions" in foundational
        assert "2" in foundational["scope"]

    def test_get_co_cited_articles(self):
        graph = CitationGraph(corpus_id="test")
        # Both 6 and 7 cite article 10
        graph.add_edge("6", "10", edge_type="cites")
        graph.add_edge("7", "10", edge_type="cites")
        graph.normalize_weights()

        co_cited = graph.get_co_cited_articles(["6", "7"])

        assert "10" in co_cited

    def test_to_dict_and_from_dict(self):
        graph = CitationGraph(corpus_id="test")
        graph.add_node(CitationNode(article_id="6", title="Test"))
        graph.add_edge("6", "7", edge_type="cites")

        d = graph.to_dict()
        restored = CitationGraph.from_dict(d)

        assert restored.corpus_id == "test"
        assert "6" in restored.nodes
        assert len(restored.edges) == 1

    def test_save_and_load(self):
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_graph.json"

            graph = CitationGraph(corpus_id="test")
            graph.add_node(CitationNode(article_id="6", title="Test"))
            graph.add_edge("6", "7")
            graph.save(path)

            loaded = CitationGraph.load(path)

            assert loaded.corpus_id == "test"
            assert "6" in loaded.nodes
            assert len(loaded.edges) == 1

    def test_bidirectional_edges_for_structural(self):
        graph = CitationGraph(corpus_id="test")
        graph.add_edge("6", "7", edge_type="co-occurrence")

        # Should be accessible from both directions
        assert "7" in graph._adjacency["6"]
        assert "6" in graph._adjacency["7"]

    def test_cites_edges_are_directional(self):
        graph = CitationGraph(corpus_id="test")
        graph.add_edge("6", "7", edge_type="cites")

        # Only accessible in one direction
        assert "7" in graph._adjacency["6"]
        assert "6" not in graph._adjacency.get("7", {})


# ---------------------------------------------------------------------------
# Test: _detect_roles_from_text
# ---------------------------------------------------------------------------


class TestDetectRolesFromText:
    def test_detects_scope(self):
        roles = _detect_roles_from_text("This regulation applies to...")
        assert "scope" in roles

    def test_detects_definitions(self):
        roles = _detect_roles_from_text("The following definitions apply")
        assert "definitions" in roles

    def test_detects_multiple_roles(self):
        roles = _detect_roles_from_text(
            "Definitions and scope of this regulation"
        )
        assert "definitions" in roles
        assert "scope" in roles

    def test_uses_title(self):
        roles = _detect_roles_from_text(
            "Article content",
            title="Anvendelsesområde"
        )
        assert "scope" in roles


# ---------------------------------------------------------------------------
# Test: load_citation_graph
# ---------------------------------------------------------------------------


class TestLoadCitationGraph:
    def test_returns_none_if_not_exists(self):
        with TemporaryDirectory() as tmpdir:
            result = load_citation_graph("nonexistent", data_dir=tmpdir)
            assert result is None

    def test_loads_existing_graph(self):
        with TemporaryDirectory() as tmpdir:
            # Create a graph file
            graph = CitationGraph(corpus_id="test")
            graph.add_node(CitationNode(article_id="6"))
            graph.save(Path(tmpdir) / "citation_graph_test.json")

            loaded = load_citation_graph("test", data_dir=tmpdir)

            assert loaded is not None
            assert loaded.corpus_id == "test"
            assert "6" in loaded.nodes


# ---------------------------------------------------------------------------
# Test: get_or_build_citation_graph
# ---------------------------------------------------------------------------


class TestGetOrBuildCitationGraph:
    def test_loads_cached_if_exists(self):
        with TemporaryDirectory() as tmpdir:
            # Create cached graph
            graph = CitationGraph(corpus_id="test")
            graph.add_node(CitationNode(article_id="cached"))
            graph.save(Path(tmpdir) / "citation_graph_test.json")

            # Should load from cache, not build
            result = get_or_build_citation_graph("test", data_dir=tmpdir)

            assert "cached" in result.nodes

    def test_rebuilds_when_forced(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create cached graph
            graph = CitationGraph(corpus_id="test")
            graph.add_node(CitationNode(article_id="cached"))
            graph.save(tmpdir_path / "citation_graph_test.json")

            # Create chunks file for rebuild
            chunks_path = tmpdir_path / "test_chunks.jsonl"
            chunks_path.write_text(
                json.dumps({
                    "text": "Article content",
                    "metadata": {"article": "rebuilt", "mentions": {"article": ["5"]}}
                })
            )

            result = get_or_build_citation_graph(
                "test",
                data_dir=tmpdir,
                rebuild=True
            )

            # Should have rebuilt from chunks
            assert "REBUILT" in result.nodes


# ---------------------------------------------------------------------------
# Test: CitationGraph.from_corpus
# ---------------------------------------------------------------------------


class TestCitationGraphFromCorpus:
    def test_builds_from_chunks(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create chunks file
            chunks = [
                {
                    "text": "Article 6 content referencing article 7",
                    "metadata": {
                        "article": "6",
                        "article_title": "Prohibited practices",
                        "chapter": "II",
                        "mentions": json.dumps({"article": ["7", "8"]})
                    }
                },
                {
                    "text": "Article 7 content",
                    "metadata": {
                        "article": "7",
                        "mentions": json.dumps({"article": ["6"]})
                    }
                },
            ]

            chunks_path = tmpdir_path / "test_chunks.jsonl"
            with open(chunks_path, "w") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk) + "\n")

            graph = CitationGraph.from_corpus("test", data_dir=tmpdir_path)

            assert "6" in graph.nodes
            assert "7" in graph.nodes
            # Article 6 cites 7 and 8
            related = graph.get_related_articles("6")
            targets = [a for a, _ in related]
            assert "7" in targets or "8" in targets

    def test_handles_annex_chunks(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            chunks = [
                {
                    "text": "Annex III content",
                    "metadata": {
                        "annex": "III",
                        "annex_title": "High-risk systems",
                        "mentions": json.dumps({"article": ["6"]})
                    }
                },
            ]

            chunks_path = tmpdir_path / "test_chunks.jsonl"
            with open(chunks_path, "w") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk) + "\n")

            graph = CitationGraph.from_corpus("test", data_dir=tmpdir_path)

            assert "ANNEX:III" in graph.nodes

    def test_raises_if_chunks_not_found(self):
        with TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                CitationGraph.from_corpus("nonexistent", data_dir=tmpdir)


# ---------------------------------------------------------------------------
# Test: Context expansion
# ---------------------------------------------------------------------------


class TestContextExpansion:
    def test_get_context_expansion_for_articles(self):
        graph = CitationGraph(corpus_id="test")
        graph.add_node(CitationNode(article_id="2", roles=["scope"]))
        graph.add_node(CitationNode(article_id="3", roles=["definitions"]))
        graph.add_node(CitationNode(article_id="6", roles=["classification"]))
        graph.add_edge("6", "7", edge_type="cites")
        graph.normalize_weights()

        # When classification article is retrieved, should expand to scope/definitions
        expansion = graph.get_context_expansion_for_articles(["6"])

        # Should include foundational articles for classification queries
        assert "2" in expansion or "3" in expansion or "7" in expansion
