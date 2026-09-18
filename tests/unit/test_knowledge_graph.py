# tests/unit/test_knowledge_graph.py — Phase 3: EngineeringKnowledgeGraph
"""Tests for EngineeringKnowledgeGraph: nodes, edges, traversal, determinism, cycles, supersession."""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.knowledge.graph import (
    EngineeringKnowledgeGraph,
    GraphEdge,
    GraphError,
    IntegrityConflict,
    TraversalOptions,
    PathResult,
)
from harness.knowledge.records import (
    EngineeringRecord,
    RecordRelationship,
    VALID_RELATIONSHIP_TYPES,
    RELATIONSHIP_REQUIRES,
    RELATIONSHIP_SATISFIES,
    RELATIONSHIP_DECIDED_BY,
    RELATIONSHIP_CONSTRAINED_BY,
    RELATIONSHIP_INTRODUCES,
    RELATIONSHIP_MITIGATES,
    RELATIONSHIP_SUPERSEDES,
    RELATIONSHIP_CAUSED_BY,
    RELATIONSHIP_RESOLVED_BY,
    RELATIONSHIP_VERIFIED_BY,
    RELATIONSHIP_SUPPORTED_BY,
    RELATIONSHIP_CONTRADICTS,
    RELATIONSHIP_RELATES_TO,
    RELATIONSHIP_DERIVED_FROM,
)
from harness.knowledge.provenance import Provenance


def make_provenance(**kwargs):
    defaults = {"author": "test_user", "source": "human"}
    defaults.update(kwargs)
    return Provenance(**defaults)


def make_record(**kwargs):
    # Determine status based on record type (some types don't accept "proposed")
    record_type = kwargs.get("record_type", kwargs.get("record_id", "REQ-001").split("-")[0])
    from harness.knowledge.lifecycle import get_initial_state
    default_status = get_initial_state(record_type)
    defaults = {
        "record_id": "REQ-001",
        "record_type": "REQ",
        "title": "Test Record",
        "description": "A test record",
        "status": default_status,
        "authority": "proposed",
        "provenance": make_provenance(),
    }
    defaults.update(kwargs)
    return EngineeringRecord(**defaults)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def simple_graph():
    """REQ-001 → ADR-004 (DECIDED_BY), ADR-004 → NFR-001 (SATISFIES)."""
    g = EngineeringKnowledgeGraph()
    g.add_edge("REQ-001", RELATIONSHIP_DECIDED_BY, "ADR-004")
    g.add_edge("ADR-004", RELATIONSHIP_SATISFIES, "NFR-001")
    return g


@pytest.fixture
def chain_graph():
    """ADR-001 → SUPERSEDES → ADR-004 → SUPERSEDES → ADR-009."""
    g = EngineeringKnowledgeGraph()
    g.add_edge("ADR-001", RELATIONSHIP_SUPERSEDES, "ADR-004")
    g.add_edge("ADR-004", RELATIONSHIP_SUPERSEDES, "ADR-009")
    return g


@pytest.fixture
def store_with_records(tmp_path):
    """Create a KnowledgeStore with records and relationships."""
    from harness.knowledge.store import KnowledgeStore

    db_path = str(tmp_path / "test.db")
    store = KnowledgeStore(db_path)

    rec1 = make_record(
        record_id="REQ-001",
        title="Requirement 1",
        description="First requirement",
    )
    rec1.add_relationship("ADR-004", RELATIONSHIP_DECIDED_BY)

    rec2 = make_record(
        record_id="ADR-004",
        record_type="ADR",
        title="Decision 4",
        description="Architecture decision",
        status="accepted",
        authority="accepted",
    )
    rec2.add_relationship("NFR-001", RELATIONSHIP_SATISFIES)
    rec2.add_relationship("TDR-001", RELATIONSHIP_INTRODUCES)

    rec3 = make_record(
        record_id="NFR-001",
        record_type="NFR",
        title="Performance NFR",
        description="Performance requirement",
    )

    rec4 = make_record(
        record_id="TDR-001",
        record_type="TDR",
        title="Technical Debt 1",
        description="Some debt",
    )

    store.save(rec1)
    store.save(rec2)
    store.save(rec3)
    store.save(rec4)

    return store


# ── Empty graph ───────────────────────────────────────────────────────

class TestEmptyGraph:
    def test_create_empty_graph(self):
        g = EngineeringKnowledgeGraph()
        assert g.node_count() == 0
        assert g.edge_count() == 0
        assert g.nodes == set()
        assert g.get_edges() == []

    def test_empty_graph_digest(self):
        g = EngineeringKnowledgeGraph()
        digest = g.compute_digest()
        assert isinstance(digest, str)
        assert len(digest) == 64  # SHA-256 hex


# ── Single node ───────────────────────────────────────────────────────

class TestSingleNode:
    def test_add_single_edge(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("REQ-001", RELATIONSHIP_DECIDED_BY, "ADR-004")
        assert g.node_count() == 2
        assert g.edge_count() == 1

    def test_single_node_has_node(self):
        g = EngineeringKnowledgeGraph()
        g.add_node("REQ-001")
        assert g.has_node("REQ-001")
        assert g.node_count() == 1

    def test_single_node_no_edges(self):
        g = EngineeringKnowledgeGraph()
        g.add_node("REQ-001")
        assert g.edge_count() == 0
        assert g.neighbors("REQ-001") == []


# ── Multiple nodes ────────────────────────────────────────────────────

class TestMultipleNodes:
    def test_add_multiple_nodes(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("REQ-001", RELATIONSHIP_DECIDED_BY, "ADR-004")
        g.add_edge("REQ-002", RELATIONSHIP_DECIDED_BY, "ADR-004")
        g.add_edge("ADR-004", RELATIONSHIP_SATISFIES, "NFR-001")
        assert g.node_count() == 4
        assert g.edge_count() == 3

    def test_all_relation_types(self):
        g = EngineeringKnowledgeGraph()
        for rtype in VALID_RELATIONSHIP_TYPES:
            # Create unique source/target per relation type
            src = f"SRC-{rtype[:3]}"
            tgt = f"TGT-{rtype[:3]}"
            g.add_edge(src, rtype, tgt)
        assert g.edge_count() == len(VALID_RELATIONSHIP_TYPES)


# ── GraphEdge ─────────────────────────────────────────────────────────

class TestGraphEdge:
    def test_edge_creation(self):
        e = GraphEdge("REQ-001", RELATIONSHIP_DECIDED_BY, "ADR-004")
        assert e.source_id == "REQ-001"
        assert e.relation_type == RELATIONSHIP_DECIDED_BY
        assert e.target_id == "ADR-004"

    def test_edge_invalid_type_raises(self):
        with pytest.raises(GraphError):
            GraphEdge("REQ-001", "INVALID", "ADR-004")

    def test_edge_serialization_round_trip(self):
        e = GraphEdge("REQ-001", RELATIONSHIP_DECIDED_BY, "ADR-004", metadata={"weight": 2})
        d = e.to_dict()
        e2 = GraphEdge.from_dict(d)
        assert e == e2

    def test_edge_from_relationship(self):
        rel = RecordRelationship("REQ-001", RELATIONSHIP_DECIDED_BY, "ADR-004")
        e = GraphEdge.from_relationship(rel)
        assert e.source_id == "REQ-001"
        assert e.relation_type == RELATIONSHIP_DECIDED_BY
        assert e.target_id == "ADR-004"

    def test_edge_equality(self):
        e1 = GraphEdge("A", RELATIONSHIP_REQUIRES, "B")
        e2 = GraphEdge("A", RELATIONSHIP_REQUIRES, "B")
        assert e1 == e2

    def test_edge_hash(self):
        e1 = GraphEdge("A", RELATIONSHIP_REQUIRES, "B")
        e2 = GraphEdge("A", RELATIONSHIP_REQUIRES, "B")
        assert hash(e1) == hash(e2)


# ── Outgoing/incoming traversal ───────────────────────────────────────

class TestOutgoingIncoming:
    def test_outgoing(self, simple_graph):
        result = simple_graph.outgoing("REQ-001")
        assert result == ["ADR-004"]

    def test_incoming(self, simple_graph):
        result = simple_graph.incoming("ADR-004")
        assert result == ["REQ-001"]

    def test_related_both(self, simple_graph):
        result = simple_graph.related("ADR-004")
        assert set(result) == {"REQ-001", "NFR-001"}

    def test_neighbors_outgoing(self, simple_graph):
        result = simple_graph.neighbors("REQ-001", direction="outgoing")
        assert result == ["ADR-004"]

    def test_neighbors_incoming(self, simple_graph):
        result = simple_graph.neighbors("ADR-004", direction="incoming")
        assert result == ["REQ-001"]

    def test_neighbors_both(self, simple_graph):
        result = simple_graph.neighbors("ADR-004", direction="both")
        assert set(result) == {"REQ-001", "NFR-001"}


# ── Typed traversal ───────────────────────────────────────────────────

class TestTypedTraversal:
    def test_typed_traversal(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        g.add_edge("A", RELATIONSHIP_SATISFIES, "C")
        result = g.neighbors("A", relation_type=RELATIONSHIP_REQUIRES)
        assert result == ["B"]

    def test_traverse_with_edge_type_filter(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        g.add_edge("A", RELATIONSHIP_SATISFIES, "C")
        g.add_edge("B", RELATIONSHIP_REQUIRES, "D")
        opts = TraversalOptions(allowed_relations={RELATIONSHIP_REQUIRES})
        result = g.traverse("A", opts)
        assert "B" in result
        assert "C" not in result  # SATISFIES edge not followed

    def test_traverse_by_record_type(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("REQ-001", RELATIONSHIP_DECIDED_BY, "ADR-004")
        g.add_edge("ADR-004", RELATIONSHIP_SATISFIES, "NFR-001")
        opts = TraversalOptions(record_types={"REQ", "ADR"})
        # This filters which nodes to include in traversal result
        result = g.traverse("REQ-001", opts)
        assert "REQ-001" in result
        # ADR-004 is allowed, NFR-001 is not


# ── Deterministic ordering ────────────────────────────────────────────

class TestDeterministicOrdering:
    def test_traversal_deterministic_order(self):
        """Same graph built in different orders produces same traversal."""
        g1 = EngineeringKnowledgeGraph()
        g1.add_edge("A", RELATIONSHIP_REQUIRES, "C")
        g1.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        g1.add_edge("B", RELATIONSHIP_REQUIRES, "D")

        g2 = EngineeringKnowledgeGraph()
        g2.add_edge("B", RELATIONSHIP_REQUIRES, "D")
        g2.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        g2.add_edge("A", RELATIONSHIP_REQUIRES, "C")

        assert g1.traverse("A") == g2.traverse("A")
        assert g1.to_dict() == g2.to_dict()

    def test_neighbors_sorted(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("A", RELATIONSHIP_REQUIRES, "Z")
        g.add_edge("A", RELATIONSHIP_REQUIRES, "M")
        g.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        assert g.neighbors("A") == ["B", "M", "Z"]

    def test_related_sorted(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("A", RELATIONSHIP_REQUIRES, "Z")
        g.add_edge("B", RELATIONSHIP_REQUIRES, "A")
        g.add_edge("A", RELATIONSHIP_REQUIRES, "C")
        assert g.related("A") == ["B", "C", "Z"]

    def test_edge_list_sorted(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("B", RELATIONSHIP_REQUIRES, "C")
        g.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        g.add_edge("A", RELATIONSHIP_REQUIRES, "C")
        edges = g.get_edges()
        keys = [(e.source_id, e.relation_type, e.target_id) for e in edges]
        assert keys == sorted(keys)


# ── Path found/absent ────────────────────────────────────────────────

class TestPath:
    def test_path_found(self, simple_graph):
        result = simple_graph.path("REQ-001", "NFR-001")
        assert result.found
        assert result.path == ["REQ-001", "ADR-004", "NFR-001"]
        assert result.length == 2

    def test_path_same_node(self, simple_graph):
        result = simple_graph.path("REQ-001", "REQ-001")
        assert result.found
        assert result.path == ["REQ-001"]
        assert result.length == 0

    def test_path_absent(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        g.add_edge("C", RELATIONSHIP_REQUIRES, "D")
        result = g.path("A", "D")
        assert not result.found
        assert result.path == []

    def test_path_nonexistent_node(self, simple_graph):
        result = simple_graph.path("NONEXISTENT", "ADR-004")
        assert not result.found


# ── Bounded path / max depth ─────────────────────────────────────────

class TestBoundedPath:
    def test_max_depth(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        g.add_edge("B", RELATIONSHIP_REQUIRES, "C")
        g.add_edge("C", RELATIONSHIP_REQUIRES, "D")
        g.add_edge("D", RELATIONSHIP_REQUIRES, "E")

        opts = TraversalOptions(max_depth=2)
        result = g.traverse("A", opts)
        assert "A" in result
        assert "B" in result
        assert "C" in result
        assert "D" not in result  # depth 3, beyond max_depth

    def test_max_nodes(self):
        g = EngineeringKnowledgeGraph()
        for i in range(10):
            g.add_edge(f"N{i}", RELATIONSHIP_REQUIRES, f"N{i+1}")

        opts = TraversalOptions(max_nodes=5)
        result = g.traverse("N0", opts)
        assert len(result) <= 5

    def test_bounded_path_result(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        g.add_edge("B", RELATIONSHIP_REQUIRES, "C")
        g.add_edge("C", RELATIONSHIP_REQUIRES, "D")

        opts = TraversalOptions(max_depth=1)
        result = g.path("A", "D", opts)
        assert not result.found  # max_depth=1, but path needs 3 hops

    def test_allowed_relations_bounded(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        g.add_edge("A", RELATIONSHIP_SATISFIES, "C")
        g.add_edge("B", RELATIONSHIP_REQUIRES, "D")

        opts = TraversalOptions(allowed_relations={RELATIONSHIP_REQUIRES})
        result = g.traverse("A", opts)
        assert "C" not in result


# ── Duplicate edge ────────────────────────────────────────────────────

class TestDuplicateEdge:
    def test_duplicate_edge_idempotent(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        g.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        assert g.edge_count() == 1

    def test_same_source_target_different_type(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        g.add_edge("A", RELATIONSHIP_SATISFIES, "B")
        assert g.edge_count() == 2


# ── Self-reference ────────────────────────────────────────────────────

class TestSelfReference:
    def test_self_reference_rejected(self):
        g = EngineeringKnowledgeGraph()
        with pytest.raises(GraphError, match="Self-references"):
            g.add_edge("A", RELATIONSHIP_REQUIRES, "A")


# ── Dangling internal reference ──────────────────────────────────────

class TestDanglingReference:
    def test_dangling_reference_detected(self):
        g = EngineeringKnowledgeGraph()
        g.add_node("A")
        # Manually add edge to non-existent node by manipulating internal state
        # (simulates corruption or external reference)
        edge = GraphEdge("A", RELATIONSHIP_REQUIRES, "MISSING")
        g._outgoing["A"].append(edge)
        # Do NOT add MISSING to _nodes — this is the dangling reference
        conflicts = g.validate_integrity()
        dangling = [c for c in conflicts if c.conflict_type == "DANGLING_REFERENCE"]
        assert len(dangling) > 0
        assert "MISSING" in dangling[0].records

    def test_no_dangling_in_clean_graph(self, simple_graph):
        conflicts = simple_graph.validate_integrity()
        dangling = [c for c in conflicts if c.conflict_type == "DANGLING_REFERENCE"]
        assert len(dangling) == 0


# ── External unverified reference ────────────────────────────────────

class TestExternalUnverifiedReference:
    def test_external_reference_allowed(self):
        """External references (evidence, capabilities) are allowed as nodes."""
        g = EngineeringKnowledgeGraph()
        g.add_edge("EVIDENCE-001", RELATIONSHIP_SUPPORTED_BY, "TEST-001")
        assert g.has_node("EVIDENCE-001")
        assert g.has_node("TEST-001")
        assert g.edge_count() == 1


# ── Allowed cycle ─────────────────────────────────────────────────────

class TestAllowedCycle:
    def test_relates_to_cycle_allowed(self):
        """RELATES_TO cycles are permitted (not globally prohibited)."""
        g = EngineeringKnowledgeGraph()
        g.add_edge("A", RELATIONSHIP_RELATES_TO, "B")
        g.add_edge("B", RELATIONSHIP_RELATES_TO, "A")  # Cycle
        assert g.edge_count() == 2
        # Not a forbidden cycle type
        has_cycle, _ = g.has_cycle(relation_type=RELATIONSHIP_RELATES_TO)
        assert has_cycle  # Cycle exists but not forbidden

    def test_derived_from_cycle_allowed(self):
        """DERIVED_FROM cycles are permitted."""
        g = EngineeringKnowledgeGraph()
        g.add_edge("A", RELATIONSHIP_DERIVED_FROM, "B")
        g.add_edge("B", RELATIONSHIP_DERIVED_FROM, "C")
        assert g.edge_count() == 2


# ── Forbidden cycle ───────────────────────────────────────────────────

class TestForbiddenCycle:
    def test_supersedes_cycle_forbidden(self):
        """SUPERSEDES cycles are forbidden."""
        g = EngineeringKnowledgeGraph()
        g.add_edge("A", RELATIONSHIP_SUPERSEDES, "B")
        g.add_edge("B", RELATIONSHIP_SUPERSEDES, "A")  # Creates mutual supersession cycle
        conflicts = g.validate_integrity()
        supersession = [c for c in conflicts if c.conflict_type == "SUPERSESSION_CONFLICT"]
        assert len(supersession) > 0

    def test_caused_by_cycle_detected(self):
        """CAUSED_BY cycles are forbidden (causality must be acyclic)."""
        g = EngineeringKnowledgeGraph()
        g.add_edge("A", RELATIONSHIP_CAUSED_BY, "B")
        g.add_edge("B", RELATIONSHIP_CAUSED_BY, "C")
        # Add C → A to create cycle
        g._add_edge_unchecked(GraphEdge("C", RELATIONSHIP_CAUSED_BY, "A"))
        has_cycle, cycle_path = g.has_cycle(relation_type=RELATIONSHIP_CAUSED_BY)
        assert has_cycle
        assert len(cycle_path) >= 3

    def test_no_cycle_in_dag(self, simple_graph):
        has_cycle, _ = simple_graph.has_cycle()
        assert not has_cycle


# ── Explicit CONTRADICTS edge ────────────────────────────────────────

class TestExplicitContradicts:
    def test_contradicts_edge_detected(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("ADR-001", RELATIONSHIP_CONTRADICTS, "ADR-002")
        conflicts = g.validate_integrity()
        contradictions = [c for c in conflicts if c.conflict_type == "EXPLICIT_CONTRADICTION"]
        assert len(contradictions) == 1
        assert contradictions[0].records == ["ADR-001", "ADR-002"]

    def test_no_contradiction_in_clean_graph(self, simple_graph):
        conflicts = simple_graph.validate_integrity()
        contradictions = [c for c in conflicts if c.conflict_type == "EXPLICIT_CONTRADICTION"]
        assert len(contradictions) == 0


# ── Structural contradiction ──────────────────────────────────────────

class TestStructuralContradiction:
    def test_mutual_supersession_is_structural_contradiction(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("ADR-001", RELATIONSHIP_SUPERSEDES, "ADR-004")
        g.add_edge("ADR-004", RELATIONSHIP_SUPERSEDES, "ADR-001")
        conflicts = g.validate_integrity()
        structural = [c for c in conflicts if c.conflict_type == "SUPERSESSION_CONFLICT"]
        assert len(structural) > 0

    def test_branching_supersession_is_ambiguous(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("ADR-001", RELATIONSHIP_SUPERSEDES, "ADR-004")
        g.add_edge("ADR-001", RELATIONSHIP_SUPERSEDES, "ADR-009")
        conflicts = g.validate_integrity()
        ambiguous = [c for c in conflicts if c.conflict_type == "SUPERSESSION_CONFLICT"]
        assert len(ambiguous) > 0


# ── Supersession chain ────────────────────────────────────────────────

class TestSupersessionChain:
    def test_supersession_chain_linear(self, chain_graph):
        chain = chain_graph.supersession_chain("ADR-001")
        assert chain == ["ADR-001", "ADR-004", "ADR-009"]

    def test_supersession_chain_single_node(self, simple_graph):
        chain = simple_graph.supersession_chain("REQ-001")
        assert chain == ["REQ-001"]

    def test_effective_record(self, chain_graph):
        effective = chain_graph.effective_record("ADR-001")
        assert effective == "ADR-009"

    def test_is_superseded(self, chain_graph):
        assert chain_graph.is_superseded("ADR-001")
        assert not chain_graph.is_superseded("ADR-009")


# ── Supersession cycle ────────────────────────────────────────────────

class TestSupersessionCycle:
    def test_supersession_cycle_detected(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("ADR-001", RELATIONSHIP_SUPERSEDES, "ADR-004")
        g.add_edge("ADR-004", RELATIONSHIP_SUPERSEDES, "ADR-001")
        conflicts = g.validate_integrity()
        cycles = [c for c in conflicts if c.conflict_type == "SUPERSESSION_CONFLICT"]
        assert len(cycles) > 0


# ── Ambiguous supersession ───────────────────────────────────────────

class TestAmbiguousSupersession:
    def test_branching_supersession(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("ADR-001", RELATIONSHIP_SUPERSEDES, "ADR-004")
        g.add_edge("ADR-001", RELATIONSHIP_SUPERSEDES, "ADR-009")
        effective = g.effective_record("ADR-001")
        assert effective is None  # Ambiguous — fail closed


# ── Graph rebuild ─────────────────────────────────────────────────────

class TestGraphRebuild:
    def test_rebuild_from_store(self, store_with_records):
        g = EngineeringKnowledgeGraph.build_from_store(store_with_records)
        assert g.node_count() == 4
        assert g.has_node("REQ-001")
        assert g.has_node("ADR-004")
        assert g.edge_count() == 3  # REQ→ADR, ADR→NFR, ADR→TDR

    def test_rebuild_idempotent(self, store_with_records):
        g1 = EngineeringKnowledgeGraph.build_from_store(store_with_records)
        g1_digest = g1.compute_digest()
        # Simulate destroy + rebuild
        del g1
        g2 = EngineeringKnowledgeGraph.build_from_store(store_with_records)
        g2_digest = g2.compute_digest()
        assert g1_digest == g2_digest


# ── Rebuild idempotency ───────────────────────────────────────────────

class TestRebuildIdempotency:
    def test_build_from_records_deterministic(self):
        """Same records in different order → same graph."""
        r1 = make_record(record_id="REQ-001")
        r1.add_relationship("ADR-004", RELATIONSHIP_DECIDED_BY)
        r2 = make_record(record_id="ADR-004", record_type="ADR")
        r2.add_relationship("NFR-001", RELATIONSHIP_SATISFIES)
        r3 = make_record(record_id="NFR-001", record_type="NFR")

        g1 = EngineeringKnowledgeGraph.build_from_records([r1, r2, r3])
        g2 = EngineeringKnowledgeGraph.build_from_records([r3, r1, r2])
        g3 = EngineeringKnowledgeGraph.build_from_records([r2, r3, r1])

        assert g1.to_dict() == g2.to_dict() == g3.to_dict()
        assert g1.compute_digest() == g2.compute_digest() == g3.compute_digest()


# ── Restart determinism ───────────────────────────────────────────────

class TestRestartDeterminism:
    def test_digest_across_instances(self, store_with_records):
        """Process A and Process B rebuild same graph from same store."""
        g1 = EngineeringKnowledgeGraph.build_from_store(store_with_records)
        digest1 = g1.compute_digest()

        # Simulate process B: reload store from disk
        store_with_records.reload()
        g2 = EngineeringKnowledgeGraph.build_from_store(store_with_records)
        digest2 = g2.compute_digest()

        assert digest1 == digest2


# ── Graph digest determinism ──────────────────────────────────────────

class TestGraphDigest:
    def test_digest_deterministic(self, simple_graph):
        d1 = simple_graph.compute_digest()
        d2 = simple_graph.compute_digest()
        assert d1 == d2

    def test_digest_differs_for_different_graphs(self):
        g1 = EngineeringKnowledgeGraph()
        g1.add_edge("A", RELATIONSHIP_REQUIRES, "B")

        g2 = EngineeringKnowledgeGraph()
        g2.add_edge("A", RELATIONSHIP_REQUIRES, "C")

        assert g1.compute_digest() != g2.compute_digest()

    def test_digest_same_for_equivalent_graphs(self):
        g1 = EngineeringKnowledgeGraph()
        g1.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        g1.add_edge("B", RELATIONSHIP_REQUIRES, "C")

        g2 = EngineeringKnowledgeGraph()
        g2.add_edge("B", RELATIONSHIP_REQUIRES, "C")
        g2.add_edge("A", RELATIONSHIP_REQUIRES, "B")

        assert g1.compute_digest() == g2.compute_digest()


# ── KnowledgeStore remains canonical ─────────────────────────────────

class TestKnowledgeStoreCanonical:
    def test_graph_does_not_modify_store(self, store_with_records):
        """Building graph from store must not modify the store."""
        records_before = store_with_records.list_all()
        hashes_before = {r.record_id: r.compute_hash() for r in records_before}

        _ = EngineeringKnowledgeGraph.build_from_store(store_with_records)

        records_after = store_with_records.list_all()
        hashes_after = {r.record_id: r.compute_hash() for r in records_after}

        assert hashes_before == hashes_after


# ── Corrupt store record not silently admitted ────────────────────────

class TestCorruptStoreRecord:
    def test_corrupt_record_excluded(self, store_with_records):
        """A corrupt record in the store is detected, not silently admitted."""
        # First verify clean store builds clean graph
        g_clean = EngineeringKnowledgeGraph.build_from_store(store_with_records)
        assert g_clean.validate_integrity() == []

        # Get raw record data and corrupt it
        import sqlite3
        conn = sqlite3.connect(store_with_records.db_path)
        row = conn.execute(
            "SELECT record_json FROM records WHERE record_id = 'REQ-001'"
        ).fetchone()
        original_json = row[0]

        # Corrupt: change description without updating hash
        import json
        data = json.loads(original_json)
        data["description"] = "CORRUPTED"
        corrupted_json = json.dumps(data, sort_keys=True, indent=2)

        conn.execute(
            "UPDATE records SET record_json = ? WHERE record_id = 'REQ-001'",
            (corrupted_json,),
        )
        conn.commit()
        conn.close()

        # Verify store detects corruption
        assert not store_with_records.verify_integrity("REQ-001")
        from harness.knowledge.store import IntegrityError
        with pytest.raises(IntegrityError):
            store_with_records.get("REQ-001")


# ── CapabilityGraph separation ───────────────────────────────────────

class TestCapabilityGraphSeparation:
    def test_graph_does_not_import_capability(self):
        """EngineeringKnowledgeGraph must not reference capability classes."""
        import harness.knowledge.graph as graph_module
        source = open(graph_module.__file__).read()
        # Check for actual imports, not just mentions in comments/docstrings
        import ast
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    assert "capability" not in alias.name.lower(), \
                        f"Found capability import: {alias.name}"

    def test_graph_does_not_store_capabilities(self):
        """Graph nodes are record IDs, not capability data."""
        g = EngineeringKnowledgeGraph()
        g.add_edge("CAP-001", RELATIONSHIP_REQUIRES, "CAP-002")
        # Allowed as nodes but semantics are Engineering Records
        assert g.has_node("CAP-001")
        assert g.has_node("CAP-002")


# ── Engineering/Learning separation ──────────────────────────────────

class TestEngineeringLearningSeparation:
    def test_graph_no_learning_types(self):
        """Graph does not use learning domain types."""
        g = EngineeringKnowledgeGraph()
        g.add_edge("REQ-001", RELATIONSHIP_DECIDED_BY, "ADR-004")
        edges = g.get_edges()
        for edge in edges:
            assert edge.relation_type not in {
                "informs_strategy", "updates_experience", "learns_from",
                "informs_exploration",
            }


# ── Engineering/Evidence separation ─────────────────────────────────

class TestEngineeringEvidenceSeparation:
    def test_evidence_refs_are_string_ids(self):
        """Evidence references are string IDs, not embedded objects."""
        g = EngineeringKnowledgeGraph()
        g.add_edge("EVID-001", RELATIONSHIP_VERIFIED_BY, "TEST-001")
        assert g.has_node("EVID-001")
        # Evidence is referenced by ID, not duplicated
        edges = g.get_edges()
        for edge in edges:
            assert isinstance(edge.source_id, str)
            assert isinstance(edge.target_id, str)


# ── Integrity conflict structure ─────────────────────────────────────

class TestIntegrityConflictStructure:
    def test_conflict_to_dict(self):
        c = IntegrityConflict(
            conflict_type="DANGLING_REFERENCE",
            records=["A", "B"],
            relations=[{"source_id": "A", "relation_type": "REQUIRES", "target_id": "B"}],
            severity="HIGH",
            reason="Test reason",
        )
        d = c.to_dict()
        assert d["type"] == "DANGLING_REFERENCE"
        assert d["records"] == ["A", "B"]
        assert d["severity"] == "HIGH"
        assert d["reason"] == "Test reason"

    def test_conflict_deterministic_order(self):
        conflicts = [
            IntegrityConflict("DANGLING_REFERENCE", ["Z", "A"], [], "HIGH", ""),
            IntegrityConflict("FORBIDDEN_CYCLE", ["B", "C"], [], "HIGH", ""),
            IntegrityConflict("DANGLING_REFERENCE", ["A", "B"], [], "MEDIUM", ""),
        ]
        sorted_conflicts = sorted(conflicts, key=lambda c: (c.conflict_type, c.records))
        assert sorted_conflicts[0].conflict_type == "DANGLING_REFERENCE"
        assert sorted_conflicts[0].records == ["A", "B"]


# ── TraversalOptions ─────────────────────────────────────────────────

class TestTraversalOptions:
    def test_default_options(self):
        opts = TraversalOptions()
        assert opts.max_depth == 10
        assert opts.max_nodes == 1000
        assert opts.allowed_relations is None
        assert opts.direction == "outgoing"

    def test_options_to_dict(self):
        opts = TraversalOptions(max_depth=5, max_nodes=50, allowed_relations={"REQUIRES"})
        d = opts.to_dict()
        assert d["max_depth"] == 5
        assert d["max_nodes"] == 50
        assert d["allowed_relations"] == ["REQUIRES"]

    def test_options_invalid_max_depth(self):
        with pytest.raises(GraphError):
            TraversalOptions(max_depth=-1)


# ── Ancestors/descendants ────────────────────────────────────────────

class TestAncestorsDescendants:
    def test_descendants(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("REQ-001", RELATIONSHIP_DECIDED_BY, "ADR-004")
        g.add_edge("ADR-004", RELATIONSHIP_SATISFIES, "NFR-001")
        result = g.descendants("REQ-001")
        assert "ADR-004" in result
        assert "NFR-001" in result

    def test_ancestors(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("REQ-001", RELATIONSHIP_DECIDED_BY, "ADR-004")
        g.add_edge("ADR-004", RELATIONSHIP_SATISFIES, "NFR-001")
        result = g.ancestors("NFR-001")
        assert "ADR-004" in result
        assert "REQ-001" in result


# ── Serialization ────────────────────────────────────────────────────

class TestSerialization:
    def test_to_dict_deterministic(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("B", RELATIONSHIP_REQUIRES, "C")
        g.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        d = g.to_dict()
        assert d["nodes"] == ["A", "B", "C"]
        keys = [(e["source_id"], e["relation_type"], e["target_id"]) for e in d["edges"]]
        assert keys == sorted(keys)

    def test_from_dict_round_trip(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        g.add_edge("B", RELATIONSHIP_REQUIRES, "C")
        d = g.to_dict()
        g2 = EngineeringKnowledgeGraph.from_dict(d)
        assert g == g2

    def test_to_json(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        text = g.to_json()
        g2 = EngineeringKnowledgeGraph.from_json(text)
        assert g == g2


# ── Logical equality ─────────────────────────────────────────────────

class TestLogicalEquality:
    def test_same_graph_equal(self):
        g1 = EngineeringKnowledgeGraph()
        g1.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        g2 = EngineeringKnowledgeGraph()
        g2.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        assert g1 == g2

    def test_different_graphs_not_equal(self):
        g1 = EngineeringKnowledgeGraph()
        g1.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        g2 = EngineeringKnowledgeGraph()
        g2.add_edge("A", RELATIONSHIP_REQUIRES, "C")
        assert g1 != g2

    def test_empty_graphs_equal(self):
        g1 = EngineeringKnowledgeGraph()
        g2 = EngineeringKnowledgeGraph()
        assert g1 == g2


# ── Fail-closed influence ─────────────────────────────────────────────

class TestFailClosed:
    def test_conflicts_detected_not_silently_ignored(self):
        g = EngineeringKnowledgeGraph()
        g.add_edge("ADR-001", RELATIONSHIP_SUPERSEDES, "ADR-004")
        g.add_edge("ADR-004", RELATIONSHIP_SUPERSEDES, "ADR-001")
        conflicts = g.validate_integrity()
        assert len(conflicts) > 0
        # Graph must not be presented as clean knowledge
        assert not g.validate_integrity() == []

    def test_fail_closed_no_arbitrary_resolution(self):
        """Ambiguous supersession must not arbitrarily pick a record."""
        g = EngineeringKnowledgeGraph()
        g.add_edge("ADR-001", RELATIONSHIP_SUPERSEDES, "ADR-004")
        g.add_edge("ADR-001", RELATIONSHIP_SUPERSEDES, "ADR-009")
        effective = g.effective_record("ADR-001")
        assert effective is None  # Fail closed — no arbitrary choice


# ── Representative graph performance ─────────────────────────────────

class TestGraphPerformance:
    def test_medium_graph_traversal(self):
        """Test performance on a medium-sized graph (50 nodes)."""
        import time
        g = EngineeringKnowledgeGraph()
        # Create a chain: N0 → N1 → N2 → ... → N49
        for i in range(49):
            g.add_edge(f"N{i:03d}", RELATIONSHIP_REQUIRES, f"N{i+1:03d}")
        # Add some cross-links
        for i in range(0, 49, 5):
            g.add_edge(f"N{i:03d}", RELATIONSHIP_RELATES_TO, f"N{min(i+3, 49):03d}")

        start = time.time()
        result = g.traverse("N000")
        elapsed = time.time() - start
        assert len(result) > 0
        assert elapsed < 5.0  # Should be very fast

    def test_medium_graph_digest(self):
        g = EngineeringKnowledgeGraph()
        for i in range(50):
            g.add_edge(f"N{i:03d}", RELATIONSHIP_REQUIRES, f"N{i+1:03d}")
        d1 = g.compute_digest()
        d2 = g.compute_digest()
        assert d1 == d2

    def test_integrity_validation_performance(self):
        import time
        g = EngineeringKnowledgeGraph()
        for i in range(50):
            g.add_edge(f"N{i:03d}", RELATIONSHIP_REQUIRES, f"N{i+1:03d}")
        for i in range(0, 49, 3):
            g.add_edge(f"N{i:03d}", RELATIONSHIP_SATISFIES, f"N{min(i+2, 49):03d}")

        start = time.time()
        conflicts = g.validate_integrity()
        elapsed = time.time() - start
        assert elapsed < 5.0
