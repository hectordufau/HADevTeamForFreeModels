# tests/unit/test_knowledge_graph.py — Phase 3: KnowledgeGraph Core Model
"""Tests for KnowledgeGraph: nodes, edges, traversal, cycle detection, edge types."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.knowledge.graph import (
    KnowledgeGraph,
    GraphError,
    EdgeType,
    EDGE_TYPES,
    VALID_EDGE_TYPES,
)


class TestEdgeTypes:
    """Test edge type definitions and validation."""

    def test_edge_type_constants(self):
        assert EdgeType.GOVERNS == "governs"
        assert EdgeType.INFORMS == "informs"
        assert EdgeType.CONSTRAINS == "constrains"
        assert EdgeType.ADDRESSES == "addresses"
        assert EdgeType.SATISFIES == "satisfies"
        assert EdgeType.CREATES == "creates"
        assert EdgeType.INTRODUCES == "introduces"
        assert EdgeType.ENFORCES == "enforces"
        assert EdgeType.MOTIVATES == "motivates"
        assert EdgeType.IDENTIFIES == "identifies"

    def test_all_defined_edge_types_are_valid(self):
        for et in EDGE_TYPES:
            assert et in VALID_EDGE_TYPES

    def test_edge_types_are_unique(self):
        assert len(EDGE_TYPES) == len(set(EDGE_TYPES))


class TestKnowledgeGraphCreation:
    """Test basic graph creation and node/edge operations."""

    def test_create_empty_graph(self):
        graph = KnowledgeGraph()
        assert graph.node_count() == 0
        assert graph.edge_count() == 0

    def test_add_node(self):
        graph = KnowledgeGraph()
        graph.add_node("PRD-001")
        assert graph.node_count() == 1
        assert graph.has_node("PRD-001")

    def test_add_duplicate_node_is_noop(self):
        graph = KnowledgeGraph()
        graph.add_node("PRD-001")
        graph.add_node("PRD-001")
        assert graph.node_count() == 1

    def test_add_edge_creates_nodes(self):
        graph = KnowledgeGraph()
        graph.add_edge("PRD-001", "NFR-001", EdgeType.GOVERNS)
        assert graph.node_count() == 2
        assert graph.has_node("PRD-001")
        assert graph.has_node("NFR-001")
        assert graph.edge_count() == 1

    def test_add_edge_with_string_type(self):
        graph = KnowledgeGraph()
        graph.add_edge("PRD-001", "NFR-001", "governs")
        assert graph.edge_count() == 1

    def test_add_edge_invalid_type_raises(self):
        graph = KnowledgeGraph()
        with pytest.raises(GraphError):
            graph.add_edge("PRD-001", "NFR-001", "invalid_type")

    def test_add_edge_self_reference_raises(self):
        graph = KnowledgeGraph()
        with pytest.raises(GraphError):
            graph.add_edge("PRD-001", "PRD-001", EdgeType.GOVERNS)

    def test_add_duplicate_edge_is_noop(self):
        graph = KnowledgeGraph()
        graph.add_edge("PRD-001", "NFR-001", EdgeType.GOVERNS)
        graph.add_edge("PRD-001", "NFR-001", EdgeType.GOVERNS)
        assert graph.edge_count() == 1


class TestGetRelated:
    """Test get_related() for outgoing and incoming edges."""

    def test_get_related_outgoing(self):
        graph = KnowledgeGraph()
        graph.add_edge("PRD-001", "NFR-001", EdgeType.GOVERNS)
        graph.add_edge("PRD-001", "ADR-001", EdgeType.INFORMS)
        related = graph.get_related("PRD-001")
        assert set(related) == {"NFR-001", "ADR-001"}

    def test_get_related_incoming(self):
        graph = KnowledgeGraph()
        graph.add_edge("PRD-001", "NFR-001", EdgeType.GOVERNS)
        graph.add_edge("ADR-001", "NFR-001", EdgeType.CONSTRAINS)
        related = graph.get_related("NFR-001", direction="incoming")
        assert set(related) == {"PRD-001", "ADR-001"}

    def test_get_related_both_directions(self):
        graph = KnowledgeGraph()
        graph.add_edge("PRD-001", "NFR-001", EdgeType.GOVERNS)
        graph.add_edge("ADR-001", "NFR-001", EdgeType.CONSTRAINS)
        related = graph.get_related("NFR-001", direction="both")
        assert set(related) == {"PRD-001", "ADR-001"}

    def test_get_related_by_edge_type(self):
        graph = KnowledgeGraph()
        graph.add_edge("PRD-001", "NFR-001", EdgeType.GOVERNS)
        graph.add_edge("PRD-001", "ADR-001", EdgeType.INFORMS)
        related = graph.get_related("PRD-001", edge_type=EdgeType.GOVERNS)
        assert related == ["NFR-001"]

    def test_get_related_nonexistent_node(self):
        graph = KnowledgeGraph()
        assert graph.get_related("NONEXISTENT") == []

    def test_get_related_invalid_direction_raises(self):
        graph = KnowledgeGraph()
        with pytest.raises(GraphError):
            graph.get_related("PRD-001", direction="sideways")


class TestTraversal:
    """Test BFS and DFS traversal."""

    def test_bfs_traversal(self):
        graph = KnowledgeGraph()
        # PRD-001 → NFR-001 → ADR-001
        # PRD-001 → ADR-002
        graph.add_edge("PRD-001", "NFR-001", EdgeType.GOVERNS)
        graph.add_edge("NFR-001", "ADR-001", EdgeType.CONSTRAINS)
        graph.add_edge("PRD-001", "ADR-002", EdgeType.INFORMS)
        result = graph.traverse("PRD-001", method="bfs")
        assert result[0] == "PRD-001"
        assert set(result[1:]) == {"NFR-001", "ADR-001", "ADR-002"}

    def test_dfs_traversal(self):
        graph = KnowledgeGraph()
        graph.add_edge("PRD-001", "NFR-001", EdgeType.GOVERNS)
        graph.add_edge("NFR-001", "ADR-001", EdgeType.CONSTRAINS)
        graph.add_edge("PRD-001", "ADR-002", EdgeType.INFORMS)
        result = graph.traverse("PRD-001", method="dfs")
        assert result[0] == "PRD-001"
        assert set(result[1:]) == {"NFR-001", "ADR-001", "ADR-002"}

    def test_traversal_max_depth(self):
        graph = KnowledgeGraph()
        graph.add_edge("A", "B", EdgeType.GOVERNS)
        graph.add_edge("B", "C", EdgeType.GOVERNS)
        graph.add_edge("C", "D", EdgeType.GOVERNS)
        result = graph.traverse("A", method="bfs", max_depth=2)
        assert "A" in result
        assert "B" in result
        assert "C" in result
        assert "D" not in result

    def test_traversal_invalid_method_raises(self):
        graph = KnowledgeGraph()
        with pytest.raises(GraphError):
            graph.traverse("A", method="astar")

    def test_traversal_nonexistent_start(self):
        graph = KnowledgeGraph()
        assert graph.traverse("NONEXISTENT") == []

    def test_traversal_with_edge_type_filter(self):
        graph = KnowledgeGraph()
        graph.add_edge("PRD-001", "NFR-001", EdgeType.GOVERNS)
        graph.add_edge("PRD-001", "ADR-001", EdgeType.INFORMS)
        result = graph.traverse("PRD-001", method="bfs", edge_type=EdgeType.GOVERNS)
        assert "NFR-001" in result
        assert "ADR-001" not in result


class TestCycleDetection:
    """Test that the graph enforces DAG invariant (no cycles)."""

    def test_no_cycle_in_dag(self):
        graph = KnowledgeGraph()
        graph.add_edge("A", "B", EdgeType.GOVERNS)
        graph.add_edge("B", "C", EdgeType.GOVERNS)
        assert not graph.has_cycle()

    def test_back_edge_rejected_when_it_creates_cycle(self):
        graph = KnowledgeGraph()
        graph.add_edge("A", "B", EdgeType.GOVERNS)
        graph.add_edge("B", "C", EdgeType.GOVERNS)
        # C → A would create a cycle → must be rejected
        with pytest.raises(GraphError, match="would create a cycle"):
            graph.add_edge("C", "A", EdgeType.GOVERNS)
        # Graph remains acyclic
        assert not graph.has_cycle()
        assert graph.edge_count() == 2

    def test_self_loop_via_manual_edge_is_still_acyclic(self):
        graph = KnowledgeGraph()
        graph.add_edge("A", "B", EdgeType.GOVERNS)
        # Self-references are rejected at add_edge
        with pytest.raises(GraphError, match="Self-references"):
            graph.add_edge("A", "A", EdgeType.GOVERNS)
        assert not graph.has_cycle()

    def test_multiple_paths_no_cycle(self):
        graph = KnowledgeGraph()
        graph.add_edge("A", "B", EdgeType.GOVERNS)
        graph.add_edge("A", "C", EdgeType.GOVERNS)
        graph.add_edge("B", "D", EdgeType.GOVERNS)
        graph.add_edge("C", "D", EdgeType.GOVERNS)
        assert not graph.has_cycle()
        # Attempting back-edge D → A must be rejected
        with pytest.raises(GraphError, match="would create a cycle"):
            graph.add_edge("D", "A", EdgeType.GOVERNS)
        assert not graph.has_cycle()

    def test_complex_dag_remains_acyclic(self):
        """Verify DAG property with a more complex graph."""
        graph = KnowledgeGraph()
        graph.add_edge("PRD-001", "REQ-001", EdgeType.GOVERNS)
        graph.add_edge("PRD-001", "REQ-002", EdgeType.GOVERNS)
        graph.add_edge("REQ-001", "NFR-001", EdgeType.GOVERNS)
        graph.add_edge("REQ-002", "NFR-002", EdgeType.GOVERNS)
        graph.add_edge("NFR-001", "ADR-001", EdgeType.CONSTRAINS)
        graph.add_edge("NFR-002", "ADR-001", EdgeType.CONSTRAINS)
        assert not graph.has_cycle()
        assert graph.node_count() == 6
        assert graph.edge_count() == 6


class TestGraphPersistence:
    """Test graph serialization and deserialization."""

    def test_save_and_load(self):
        graph = KnowledgeGraph()
        graph.add_edge("PRD-001", "NFR-001", EdgeType.GOVERNS)
        graph.add_edge("NFR-001", "ADR-001", EdgeType.CONSTRAINS)

        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            graph.save(path)
            loaded = KnowledgeGraph.load(path)
            assert loaded.node_count() == 3
            assert loaded.edge_count() == 2
            assert loaded.has_node("PRD-001")
            assert loaded.has_node("NFR-001")
            assert loaded.has_node("ADR-001")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_creates_parent_directories(self):
        graph = KnowledgeGraph()
        graph.add_edge("A", "B", EdgeType.GOVERNS)
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        os.unlink(path)
        nested = os.path.join(os.path.dirname(path), "subdir", "graph.jsonl")
        try:
            graph.save(nested)
            assert os.path.exists(nested)
        finally:
            if os.path.exists(nested):
                os.unlink(nested)

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(GraphError):
            KnowledgeGraph.load("/nonexistent/path/graph.jsonl")

    def test_to_dict(self):
        graph = KnowledgeGraph()
        graph.add_edge("PRD-001", "NFR-001", EdgeType.GOVERNS)
        d = graph.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert "PRD-001" in d["nodes"]
        assert "NFR-001" in d["nodes"]
        assert len(d["edges"]) == 1
        assert d["edges"][0]["source_id"] == "PRD-001"
        assert d["edges"][0]["target_id"] == "NFR-001"
        assert d["edges"][0]["relation_type"] == "governs"

    def test_from_dict(self):
        data = {
            "nodes": ["A", "B", "C"],
            "edges": [
                {"source_id": "A", "relation_type": "governs", "target_id": "B"},
                {"source_id": "B", "relation_type": "constrains", "target_id": "C"},
            ],
        }
        graph = KnowledgeGraph.from_dict(data)
        assert graph.node_count() == 3
        assert graph.edge_count() == 2
        assert graph.has_node("A")
        assert graph.has_node("B")
        assert graph.has_node("C")


class TestNodeIndex:
    """Test O(1) node lookup index."""

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
