# tests/unit/test_knowledge_contradiction.py — Phase 3: Contradiction Detection
"""Tests for ContradictionDetector: detecting contradictions between records."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.knowledge.records import EngineeringRecord
from harness.knowledge.provenance import Provenance, AUTHORITY_PROPOSED, AUTHORITY_ACCEPTED
from harness.knowledge.lifecycle import get_initial_state
from harness.knowledge.graph import KnowledgeGraph, EdgeType
from harness.knowledge.contradiction import (
    ContradictionDetector,
    ContradictionError,
    ContradictionStatus,
)


def make_provenance(**kwargs):
    defaults = {"author": "test_user", "source": "human"}
    defaults.update(kwargs)
    return Provenance(**defaults)


def make_record(**kwargs):
    defaults = {
        "record_id": "ADR-001",
        "record_type": "ADR",
        "title": "Test Decision",
        "description": "A test decision",
        "status": "proposed",
        "authority": "proposed",
        "provenance": make_provenance(),
    }
    defaults.update(kwargs)
    return EngineeringRecord(**defaults)


class TestContradictionDetector:
    """Test ContradictionDetector creation and basic operation."""

    def test_create_detector(self):
        graph = KnowledgeGraph()
        detector = ContradictionDetector(graph)
        assert detector.graph is graph

    def test_detect_empty_graph(self):
        graph = KnowledgeGraph()
        detector = ContradictionDetector(graph)
        contradictions = detector.detect()
        assert contradictions == []

    def test_detect_single_record_no_contradiction(self):
        graph = KnowledgeGraph()
        detector = ContradictionDetector(graph)
        record = make_record()
        detector.index_record(record)
        contradictions = detector.detect()
        assert contradictions == []


class TestADRContradictions:
    """Test contradiction detection between ADRs on the same topic."""

    def test_two_adrs_same_topic_different_decisions(self):
        graph = KnowledgeGraph()
        detector = ContradictionDetector(graph)

        adr1 = make_record(
            record_id="ADR-001",
            title="Use PostgreSQL",
            description="Use PostgreSQL as primary database",
            authority="accepted",
            tags=["database"],
        )
        adr2 = make_record(
            record_id="ADR-002",
            title="Use MongoDB",
            description="Use MongoDB as primary database",
            authority="accepted",
            tags=["database"],
        )

        detector.index_record(adr1)
        detector.index_record(adr2)

        contradictions = detector.detect()
        assert len(contradictions) == 1
        assert contradictions[0].source_id == "ADR-001"
        assert contradictions[0].target_id == "ADR-002"
        assert contradictions[0].contradiction_type == "ADR_CONFLICT"

    def test_two_adrs_different_topics_no_contradiction(self):
        graph = KnowledgeGraph()
        detector = ContradictionDetector(graph)

        adr1 = make_record(
            record_id="ADR-001",
            title="Use PostgreSQL",
            description="Use PostgreSQL as primary database",
            authority="accepted",
            tags=["database"],
        )
        adr2 = make_record(
            record_id="ADR-002",
            title="Use React",
            description="Use React for frontend framework",
            authority="accepted",
            tags=["frontend"],
        )

        detector.index_record(adr1)
        detector.index_record(adr2)

        contradictions = detector.detect()
        assert contradictions == []

    def test_proposed_adr_contradicting_accepted_is_not_contradiction(self):
        graph = KnowledgeGraph()
        detector = ContradictionDetector(graph)

        adr1 = make_record(
            record_id="ADR-001",
            title="Use PostgreSQL",
            description="Use PostgreSQL as primary database",
            authority="accepted",
            tags=["database"],
        )
        adr2 = make_record(
            record_id="ADR-002",
            title="Use MongoDB",
            description="Use MongoDB as primary database",
            authority="proposed",  # Not accepted yet
            tags=["database"],
        )

        detector.index_record(adr1)
        detector.index_record(adr2)

        contradictions = detector.detect()
        assert contradictions == []

    def test_deprecated_adr_contradicting_accepted_is_not_contradiction(self):
        graph = KnowledgeGraph()
        detector = ContradictionDetector(graph)

        adr1 = make_record(
            record_id="ADR-001",
            title="Use PostgreSQL",
            description="Use PostgreSQL as primary database",
            authority="accepted",
            tags=["database"],
        )
        adr2 = make_record(
            record_id="ADR-002",
            title="Use MongoDB",
            description="Use MongoDB as primary database",
            authority="deprecated",
            tags=["database"],
        )

        detector.index_record(adr1)
        detector.index_record(adr2)

        contradictions = detector.detect()
        assert contradictions == []


class TestNFRContradictions:
    """Test contradiction detection between ADRs and NFRs."""

    def test_adr_violates_nfr_latency(self):
        graph = KnowledgeGraph()
        detector = ContradictionDetector(graph)

        nfr = make_record(
            record_id="NFR-001",
            title="Response time < 200ms",
            description="API response time must be under 200ms",
            status="accepted",
            record_type="NFR",
            authority="accepted",
            tags=["performance"],
        )
        adr = make_record(
            record_id="ADR-001",
            title="Introduce caching layer",
            description="Add caching that introduces >500ms latency",
            record_type="ADR",
            authority="accepted",
            tags=["performance"],
        )

        detector.index_record(nfr)
        detector.index_record(adr)

        contradictions = detector.detect()
        assert len(contradictions) == 1
        assert contradictions[0].contradiction_type == "NFR_VIOLATION"

    def test_adr_satisfies_nfr_no_contradiction(self):
        graph = KnowledgeGraph()
        detector = ContradictionDetector(graph)

        nfr = make_record(
            record_id="NFR-001",
            title="Response time < 200ms",
            description="API response time must be under 200ms",
            status="accepted",
            record_type="NFR",
            authority="accepted",
            tags=["performance"],
        )
        adr = make_record(
            record_id="ADR-001",
            title="Optimize database queries",
            description="Add indexes to reduce query time to under 200ms",
            record_type="ADR",
            authority="accepted",
            tags=["performance"],
        )

        detector.index_record(nfr)
        detector.index_record(adr)

        contradictions = detector.detect()
        assert contradictions == []


class TestContradictionFlagging:
    """Test flagging contradictions for human review."""

    def test_flag_contradiction(self):
        graph = KnowledgeGraph()
        detector = ContradictionDetector(graph)

        adr1 = make_record(
            record_id="ADR-001",
            title="Use PostgreSQL",
            description="Use PostgreSQL as primary database",
            authority="accepted",
            tags=["database"],
        )
        adr2 = make_record(
            record_id="ADR-002",
            title="Use MongoDB",
            description="Use MongoDB as primary database",
            authority="accepted",
            tags=["database"],
        )

        detector.index_record(adr1)
        detector.index_record(adr2)

        contradictions = detector.detect()
        assert len(contradictions) == 1

        flagged = detector.flag_contradiction(contradictions[0].contradiction_id)
        assert flagged.status == ContradictionStatus.FLAGGED
        assert flagged.flagged_at is not None

    def test_flag_nonexistent_contradiction_raises(self):
        graph = KnowledgeGraph()
        detector = ContradictionDetector(graph)

        with pytest.raises(ContradictionError):
            detector.flag_contradiction("NONEXISTENT")

    def test_resolve_contradiction(self):
        graph = KnowledgeGraph()
        detector = ContradictionDetector(graph)

        adr1 = make_record(
            record_id="ADR-001",
            title="Use PostgreSQL",
            description="Use PostgreSQL as primary database",
            authority="accepted",
            tags=["database"],
        )
        adr2 = make_record(
            record_id="ADR-002",
            title="Use MongoDB",
            description="Use MongoDB as primary database",
            authority="accepted",
            tags=["database"],
        )

        detector.index_record(adr1)
        detector.index_record(adr2)

        contradictions = detector.detect()
        resolved = detector.resolve_contradiction(
            contradictions[0].contradiction_id, resolution="ADR-002 supersedes ADR-001"
        )
        assert resolved.status == ContradictionStatus.RESOLVED
        assert resolved.resolution == "ADR-002 supersedes ADR-001"


class TestContradictionStatus:
    """Test contradiction status lifecycle."""

    def test_contradiction_status_values(self):
        assert ContradictionStatus.PENDING.value == "pending"
        assert ContradictionStatus.FLAGGED.value == "flagged"
        assert ContradictionStatus.RESOLVED.value == "resolved"
        assert ContradictionStatus.REJECTED.value == "rejected"


class TestContradictionWithGraph:
    """Test contradiction detection integrated with graph."""

    def test_related_records_same_topic_detected(self):
        graph = KnowledgeGraph()
        graph.add_edge("PRD-001", "ADR-001", EdgeType.INFORMS)
        graph.add_edge("PRD-001", "ADR-002", EdgeType.INFORMS)

        detector = ContradictionDetector(graph)

        adr1 = make_record(
            record_id="ADR-001",
            title="Use PostgreSQL",
            description="Use PostgreSQL as primary database",
            authority="accepted",
            tags=["database"],
        )
        adr2 = make_record(
            record_id="ADR-002",
            title="Use MongoDB",
            description="Use MongoDB as primary database",
            authority="accepted",
            tags=["database"],
        )

        detector.index_record(adr1)
        detector.index_record(adr2)

        contradictions = detector.detect()
        assert len(contradictions) == 1

    def test_detect_with_graph_context(self):
        graph = KnowledgeGraph()
        graph.add_edge("ADR-001", "ADR-002", EdgeType.INFORMS)

        detector = ContradictionDetector(graph)

        adr1 = make_record(
            record_id="ADR-001",
            title="Use PostgreSQL",
            description="Use PostgreSQL as primary database",
            authority="accepted",
            tags=["database"],
        )
        adr2 = make_record(
            record_id="ADR-002",
            title="Use MongoDB",
            description="Use MongoDB as primary database",
            authority="accepted",
            tags=["database"],
        )

        detector.index_record(adr1)
        detector.index_record(adr2)

        contradictions = detector.detect()
        assert len(contradictions) == 1
