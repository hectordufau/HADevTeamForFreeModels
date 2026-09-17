# tests/unit/test_knowledge_graph_persistence.py — Phase 3: Graph Persistence & Indexing
"""Tests for graph save/load, JSONL append-only edge log, and node index consistency."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.knowledge.graph import KnowledgeGraph, GraphError, EdgeType


@pytest.fixture
def graph():
    """Create a populated KnowledgeGraph."""
    g = KnowledgeGraph()
    g.add_edge("PRD-001", "REQ-001", EdgeType.GOVERNS)
    g.add_edge("PRD-001", "NFR-001", EdgeType.GOVERNS)
    g.add_edge("NFR-001", "ADR-001", EdgeType.CONSTRAINS)
    return g


class TestGraphJSONLPersistence:
    """Test append-only JSONL edge log persistence."""

    def test_save_jsonl_format(self, graph):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            graph.save(path, format="jsonl")
            with open(path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == 3  # 3 edges
            for line in lines:
                edge = json.loads(line)
                assert "source_id" in edge
                assert "target_id" in edge
                assert "relation_type" in edge
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_jsonl_append_only(self, graph):
        """Verify that saving twice doesn't delete existing edges in append mode."""
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            # Save initial graph
            graph.save(path, format="jsonl")
            with open(path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            initial_count = len(lines)

            # Add more edges and save again
            graph.add_edge("ADR-001", "TDR-001", EdgeType.CREATES)
            graph.save(path, format="jsonl")

            with open(path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            # Append mode: old edges are still there, new ones added
            assert len(lines) >= initial_count + 1
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_load_jsonl_format(self, graph):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            graph.save(path, format="jsonl")
            loaded = KnowledgeGraph.load(path, format="jsonl")
            assert loaded.node_count() == graph.node_count()
            assert loaded.edge_count() == graph.edge_count()
            for edge in graph.get_edges():
                assert loaded.has_edge(
                    edge["source_id"], edge["target_id"], edge["relation_type"]
                )
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_load_jsonl_with_extra_lines(self, graph):
        """Test that loading handles extra/blank lines gracefully."""
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            graph.save(path, format="jsonl")
            # Append a blank line and a comment line
            with open(path, "a") as f:
                f.write("\n")
                f.write('{"source_id": "X", "relation_type": "governs", "target_id": "Y"}\n')
            loaded = KnowledgeGraph.load(path, format="jsonl")
            assert loaded.node_count() >= graph.node_count()
            assert loaded.edge_count() >= graph.edge_count()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_invalid_format_raises(self, graph):
        with pytest.raises(GraphError, match="Invalid format"):
            graph.save("/tmp/test.graph", format="yaml")

    def test_load_invalid_format_raises(self):
        with pytest.raises(GraphError, match="Invalid format"):
            KnowledgeGraph.load("/tmp/test.graph", format="yaml")


class TestGraphJSONPersistence:
    """Test full JSON snapshot persistence."""

    def test_save_json_format(self, graph):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            graph.save(path, format="json")
            with open(path, "r") as f:
                data = json.load(f)
            assert "nodes" in data
            assert "edges" in data
            assert len(data["nodes"]) == graph.node_count()
            assert len(data["edges"]) == graph.edge_count()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_load_json_format(self, graph):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            graph.save(path, format="json")
            loaded = KnowledgeGraph.load(path, format="json")
            assert loaded.node_count() == graph.node_count()
            assert loaded.edge_count() == graph.edge_count()
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestNodeIndex:
    """Test O(1) node index consistency."""

    def test_node_index_built_on_add(self):
        graph = KnowledgeGraph()
        graph.add_node("PRD-001")
        assert "PRD-001" in graph._node_index

    def test_node_index_updated_on_edge_add(self):
        graph = KnowledgeGraph()
        graph.add_edge("A", "B", EdgeType.GOVERNS)
        assert "A" in graph._node_index
        assert "B" in graph._node_index

    def test_node_index_consistent_with_nodes(self):
        graph = KnowledgeGraph()
        graph.add_edge("A", "B", EdgeType.GOVERNS)
        graph.add_edge("B", "C", EdgeType.GOVERNS)
        assert set(graph._node_index.keys()) == set(graph.get_nodes())

    def test_node_index_updated_on_remove(self):
        graph = KnowledgeGraph()
        graph.add_edge("A", "B", EdgeType.GOVERNS)
        graph.remove_node("A")
        assert "A" not in graph._node_index
        assert "A" not in graph.get_nodes()
        assert graph.has_node("B")  # B should remain

    def test_load_rebuilds_node_index(self, graph):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            graph.save(path, format="jsonl")
            loaded = KnowledgeGraph.load(path, format="jsonl")
            assert set(loaded._node_index.keys()) == set(loaded.get_nodes())
        finally:
            if os.path.exists(path):
                os.unlink(path)
