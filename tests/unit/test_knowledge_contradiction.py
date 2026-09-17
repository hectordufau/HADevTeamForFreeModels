# tests/unit/test_knowledge_contradiction.py — Phase 3: Structural Contradiction Detection
"""Tests for ContradictionDetector: structural contradiction detection."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.knowledge.contradiction import (
    ContradictionDetector,
    ContradictionError,
    CONFLICT_TYPE_DANGLING_REFERENCE,
    CONFLICT_TYPE_INVALID_RELATION,
    CONFLICT_TYPE_FORBIDDEN_CYCLE,
    CONFLICT_TYPE_SUPERSESSION_CONFLICT,
    CONFLICT_TYPE_EXPLICIT_CONTRADICTION,
    CONFLICT_TYPE_STATUS_CONFLICT,
    VALID_CONFLICT_TYPES,
)
from harness.knowledge.graph import (
    EngineeringKnowledgeGraph,
    GraphEdge,
    GraphError,
    IntegrityConflict,
)
from harness.knowledge.records import (
    EngineeringRecord,
    RecordRelationship,
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
def detector():
    g = EngineeringKnowledgeGraph()
    return ContradictionDetector(g)


@pytest.fixture
def graph_with_contradictions():
    """Graph with mutual supersession and explicit contradiction."""
    g = EngineeringKnowledgeGraph()
    g.add_edge("ADR-001", RELATIONSHIP_SUPERSEDES, "ADR-004")
    g.add_edge("ADR-004", RELATIONSHIP_SUPERSEDES, "ADR-001")  # Mutual
    g.add_edge("ADR-001", RELATIONSHIP_CONTRADICTS, "ADR-009")
    return g


# ── Detector creation ─────────────────────────────────────────────────

class TestDetectorCreation:
    def test_create_detector(self, detector):
        assert detector.graph is not None

    def test_detect_empty_graph(self, detector):
        conflicts = detector.detect_all()
        assert conflicts == []

    def test_is_clean_empty(self, detector):
        assert detector.is_clean()


# ── Dangling reference detection ─────────────────────────────────────

class TestDanglingReference:
    def test_dangling_reference_detected(self, detector):
        g = detector.graph
        g.add_node("A")
        # Manually add edge to non-existent node (simulates corruption)
        edge = GraphEdge("A", RELATIONSHIP_REQUIRES, "NONEXISTENT")
        g._outgoing["A"].append(edge)
        # Do NOT add NONEXISTENT to _nodes — this is the dangling reference
        conflicts = detector.detect_all()
        dangling = [c for c in conflicts if c.conflict_type == CONFLICT_TYPE_DANGLING_REFERENCE]
        assert len(dangling) > 0
        assert "NONEXISTENT" in dangling[0].records

    def test_no_dangling_in_clean_graph(self, detector):
        g = detector.graph
        g.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        conflicts = detector.detect_all()
        dangling = [c for c in conflicts if c.conflict_type == CONFLICT_TYPE_DANGLING_REFERENCE]
        assert len(dangling) == 0


# ── Forbidden cycle detection ─────────────────────────────────────────

class TestForbiddenCycle:
    def test_supersedes_cycle_detected(self, detector):
        g = detector.graph
        g.add_edge("A", RELATIONSHIP_SUPERSEDES, "B")
        g._add_edge_unchecked(GraphEdge("B", RELATIONSHIP_SUPERSEDES, "A"))
        conflicts = detector.detect_all()
        cycles = [c for c in conflicts if c.conflict_type == CONFLICT_TYPE_FORBIDDEN_CYCLE]
        assert len(cycles) > 0

    def test_caused_by_cycle_detected(self, detector):
        g = detector.graph
        g.add_edge("A", RELATIONSHIP_CAUSED_BY, "B")
        g.add_edge("B", RELATIONSHIP_CAUSED_BY, "C")
        g._add_edge_unchecked(GraphEdge("C", RELATIONSHIP_CAUSED_BY, "A"))
        conflicts = detector.detect_all()
        cycles = [c for c in conflicts if c.conflict_type == CONFLICT_TYPE_FORBIDDEN_CYCLE]
        assert len(cycles) > 0

    def test_no_cycle_in_dag(self, detector):
        g = detector.graph
        g.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        g.add_edge("B", RELATIONSHIP_REQUIRES, "C")
        conflicts = detector.detect_all()
        cycles = [c for c in conflicts if c.conflict_type == CONFLICT_TYPE_FORBIDDEN_CYCLE]
        assert len(cycles) == 0


# ── Supersession conflicts ───────────────────────────────────────────

class TestSupersessionConflicts:
    def test_mutual_supersession(self, detector):
        g = detector.graph
        g.add_edge("ADR-001", RELATIONSHIP_SUPERSEDES, "ADR-004")
        g.add_edge("ADR-004", RELATIONSHIP_SUPERSEDES, "ADR-001")
        conflicts = detector.detect_all()
        supersession = [c for c in conflicts if c.conflict_type == CONFLICT_TYPE_SUPERSESSION_CONFLICT]
        assert len(supersession) > 0

    def test_branching_supersession(self, detector):
        g = detector.graph
        g.add_edge("ADR-001", RELATIONSHIP_SUPERSEDES, "ADR-004")
        g.add_edge("ADR-001", RELATIONSHIP_SUPERSEDES, "ADR-009")
        conflicts = detector.detect_all()
        ambiguous = [c for c in conflicts if c.conflict_type == CONFLICT_TYPE_SUPERSESSION_CONFLICT]
        assert len(ambiguous) > 0

    def test_clean_supersession_chain(self, detector):
        """Linear supersession chain has no conflicts."""
        g = detector.graph
        g.add_edge("ADR-001", RELATIONSHIP_SUPERSEDES, "ADR-004")
        g.add_edge("ADR-004", RELATIONSHIP_SUPERSEDES, "ADR-009")
        conflicts = detector.detect_all()
        supersession = [c for c in conflicts if c.conflict_type == CONFLICT_TYPE_SUPERSESSION_CONFLICT]
        assert len(supersession) == 0


# ── Explicit contradiction ────────────────────────────────────────────

class TestExplicitContradiction:
    def test_contradicts_edge_detected(self, detector):
        g = detector.graph
        g.add_edge("ADR-001", RELATIONSHIP_CONTRADICTS, "ADR-002")
        conflicts = detector.detect_all()
        contradictions = [c for c in conflicts if c.conflict_type == CONFLICT_TYPE_EXPLICIT_CONTRADICTION]
        assert len(contradictions) == 1
        assert contradictions[0].records == ["ADR-001", "ADR-002"]

    def test_no_contradiction_in_clean_graph(self, detector):
        g = detector.graph
        g.add_edge("A", RELATIONSHIP_REQUIRES, "B")
        conflicts = detector.detect_all()
        contradictions = [c for c in conflicts if c.conflict_type == CONFLICT_TYPE_EXPLICIT_CONTRADICTION]
        assert len(contradictions) == 0


# ── Conflict type validation ──────────────────────────────────────────

class TestConflictTypeValidation:
    def test_valid_conflict_types(self):
        assert CONFLICT_TYPE_DANGLING_REFERENCE in VALID_CONFLICT_TYPES
        assert CONFLICT_TYPE_INVALID_RELATION in VALID_CONFLICT_TYPES
        assert CONFLICT_TYPE_FORBIDDEN_CYCLE in VALID_CONFLICT_TYPES
        assert CONFLICT_TYPE_SUPERSESSION_CONFLICT in VALID_CONFLICT_TYPES
        assert CONFLICT_TYPE_EXPLICIT_CONTRADICTION in VALID_CONFLICT_TYPES
        assert CONFLICT_TYPE_STATUS_CONFLICT in VALID_CONFLICT_TYPES

    def test_conflict_type_values(self):
        assert CONFLICT_TYPE_DANGLING_REFERENCE == "DANGLING_REFERENCE"
        assert CONFLICT_TYPE_INVALID_RELATION == "INVALID_RELATION"
        assert CONFLICT_TYPE_FORBIDDEN_CYCLE == "FORBIDDEN_CYCLE"
        assert CONFLICT_TYPE_SUPERSESSION_CONFLICT == "SUPERSESSION_CONFLICT"
        assert CONFLICT_TYPE_EXPLICIT_CONTRADICTION == "EXPLICIT_CONTRADICTION"
        assert CONFLICT_TYPE_STATUS_CONFLICT == "STATUS_CONFLICT"


# ── Conflict representation ──────────────────────────────────────────

class TestConflictRepresentation:
    def test_conflict_to_dict(self):
        c = IntegrityConflict(
            conflict_type=CONFLICT_TYPE_DANGLING_REFERENCE,
            records=["A", "B"],
            relations=[{"source_id": "A", "relation_type": "REQUIRES", "target_id": "B"}],
            severity="HIGH",
            reason="Test reason",
        )
        d = c.to_dict()
        assert d["type"] == CONFLICT_TYPE_DANGLING_REFERENCE
        assert d["records"] == ["A", "B"]
        assert d["severity"] == "HIGH"
        assert d["reason"] == "Test reason"
        assert len(d["relations"]) == 1

    def test_conflict_sorting(self):
        """Conflicts must sort deterministically."""
        conflicts = [
            IntegrityConflict(CONFLICT_TYPE_DANGLING_REFERENCE, ["Z", "A"], [], "HIGH", ""),
            IntegrityConflict(CONFLICT_TYPE_FORBIDDEN_CYCLE, ["B", "C"], [], "HIGH", ""),
            IntegrityConflict(CONFLICT_TYPE_DANGLING_REFERENCE, ["A", "B"], [], "MEDIUM", ""),
        ]
        sorted_conflicts = sorted(conflicts, key=lambda c: (c.conflict_type, c.records))
        assert sorted_conflicts[0].conflict_type == CONFLICT_TYPE_DANGLING_REFERENCE
        assert sorted_conflicts[0].records == ["A", "B"]


# ── Integrated detection ─────────────────────────────────────────────

class TestIntegratedDetection:
    def test_multiple_conflict_types(self, detector):
        """Graph with multiple conflict types produces multiple conflicts."""
        g = detector.graph
        g.add_edge("ADR-001", RELATIONSHIP_SUPERSEDES, "ADR-004")
        g.add_edge("ADR-004", RELATIONSHIP_SUPERSEDES, "ADR-001")  # Mutual supersession
        g.add_edge("ADR-001", RELATIONSHIP_CONTRADICTS, "ADR-009")  # Explicit contradiction

        conflicts = detector.detect_all()
        types = {c.conflict_type for c in conflicts}
        assert CONFLICT_TYPE_SUPERSESSION_CONFLICT in types
        assert CONFLICT_TYPE_EXPLICIT_CONTRADICTION in types

    def test_detect_deterministic(self, detector):
        """Same graph produces same conflicts every time."""
        g = detector.graph
        g.add_edge("A", RELATIONSHIP_SUPERSEDES, "B")
        g.add_edge("B", RELATIONSHIP_SUPERSEDES, "A")
        c1 = detector.detect_all()
        c2 = detector.detect_all()
        assert [c.to_dict() for c in c1] == [c.to_dict() for c in c2]


# ── Boundary conditions ──────────────────────────────────────────────

class TestBoundaryConditions:
    def test_self_reference_detected_as_contradiction(self, detector):
        """Self-references should be detected."""
        g = detector.graph
        # Self-references are rejected at add_edge
        with pytest.raises(GraphError, match="Self-references"):
            g.add_edge("A", RELATIONSHIP_REQUIRES, "A")
        # Manually add edge to non-existent node (dangling)
        g.add_node("A")  # Ensure node exists in _outgoing
        edge = GraphEdge("A", RELATIONSHIP_REQUIRES, "MISSING")
        g._outgoing["A"].append(edge)
        conflicts = detector.detect_all()
        dangling = [c for c in conflicts if c.conflict_type == CONFLICT_TYPE_DANGLING_REFERENCE]
        assert len(dangling) > 0

    def test_large_graph_detection(self):
        """Detector handles large graphs."""
        g = EngineeringKnowledgeGraph()
        for i in range(50):
            g.add_edge(f"N{i:03d}", RELATIONSHIP_REQUIRES, f"N{i+1:03d}")
        detector = ContradictionDetector(g)
        conflicts = detector.detect_all()
        assert conflicts == []  # Clean graph
