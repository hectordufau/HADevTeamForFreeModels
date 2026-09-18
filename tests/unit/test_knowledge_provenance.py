# tests/unit/test_knowledge_provenance.py — Tests for V3.3 Provenance & Authority
"""Tests for Provenance dataclass, authority levels, and validation."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.knowledge.provenance import (
    Provenance,
    ProvenanceError,
    validate_authority_level,
    is_authority_at_least,
    can_transition_authority,
    AUTHORITY_PROPOSED,
    AUTHORITY_ACCEPTED,
    AUTHORITY_DEPRECATED,
    AUTHORITY_ARCHIVED,
    VALID_AUTHORITY_LEVELS,
    VALID_SOURCES,
    SOURCE_HUMAN,
    SOURCE_PIPELINE,
    SOURCE_IMPORT,
    SOURCE_GENERATED,
)


class TestProvenanceCreation:
    """Test provenance creation and validation."""

    def test_create_minimal_provenance(self):
        prov = Provenance(author="alice", source="human")
        assert prov.author == "alice"
        assert prov.source == "human"
        assert prov.evidence_refs == []
        assert prov.parent_record is None

    def test_create_full_provenance(self):
        prov = Provenance(
            author="bob",
            source="pipeline",
            evidence_refs=["ev-001", "ev-002"],
            parent_record="PRD-000",
            validation_run_id="run_001",
            benchmark_id="bench_test",
            mode="COLD",
        )
        assert prov.author == "bob"
        assert prov.source == "pipeline"
        assert len(prov.evidence_refs) == 2
        assert prov.parent_record == "PRD-000"
        assert prov.validation_run_id == "run_001"

    def test_default_timestamp(self):
        prov = Provenance(author="alice", source="human")
        assert prov.timestamp is not None
        assert isinstance(prov.timestamp, str)

    def test_invalid_source(self):
        with pytest.raises(ProvenanceError, match="Invalid source"):
            Provenance(author="alice", source="invalid_source")

    def test_empty_source_is_allowed(self):
        """Empty source is allowed (incomplete but not invalid)."""
        prov = Provenance(author="alice", source="")
        assert prov.source == ""

    def test_invalid_author_type(self):
        with pytest.raises(ProvenanceError, match="author must be a string"):
            Provenance(author=123, source="human")

    def test_invalid_evidence_refs_type(self):
        with pytest.raises(ProvenanceError, match="evidence_refs must be a list"):
            Provenance(author="alice", source="human", evidence_refs="not_a_list")


class TestProvenanceCompleteness:
    """Test provenance completeness check."""

    def test_complete_provenance(self):
        prov = Provenance(author="alice", source="human")
        assert prov.is_complete()

    def test_incomplete_no_author(self):
        prov = Provenance(author="", source="human")
        assert not prov.is_complete()

    def test_incomplete_no_source(self):
        prov = Provenance(author="alice", source="")
        assert not prov.is_complete()

    def test_incomplete_empty(self):
        prov = Provenance(author="", source="")
        assert not prov.is_complete()


class TestProvenanceSerialization:
    """Test provenance serialization."""

    def test_to_dict_deterministic(self):
        p1 = Provenance(author="alice", source="human", timestamp="2026-01-01T00:00:00")
        p2 = Provenance(author="alice", source="human", timestamp="2026-01-01T00:00:00")
        assert p1.to_dict() == p2.to_dict()

    def test_round_trip(self):
        prov = Provenance(
            author="alice",
            source="pipeline",
            evidence_refs=["ev-001"],
            parent_record="PRD-000",
            timestamp="2026-01-01T00:00:00",
        )
        restored = Provenance.from_dict(prov.to_dict())
        assert restored == prov

    def test_round_trip_with_experiment_context(self):
        from harness.learning.context import ExperimentContext
        ctx = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench_test",
            task_id="task_001",
            execution_id="exec_001",
            mode="COLD",
        )
        prov = Provenance(author="alice", source="human", experiment_context=ctx, timestamp="2026-01-01T00:00:00")
        restored = Provenance.from_dict(prov.to_dict())
        assert restored.experiment_context is not None
        assert restored.experiment_context.validation_run_id == "run_001"

    def test_json_serialization(self):
        prov = Provenance(author="alice", source="human", timestamp="2026-01-01T00:00:00")
        j = prov.to_json()
        assert isinstance(j, str)
        data = json.loads(j)
        assert data["author"] == "alice"


class TestAuthorityLevels:
    """Test authority level constants and validation."""

    def test_valid_authority_levels(self):
        assert VALID_AUTHORITY_LEVELS == {
            "proposed",
            "accepted",
            "deprecated",
            "archived",
        }

    def test_validate_authority_level_valid(self):
        for level in VALID_AUTHORITY_LEVELS:
            validate_authority_level(level)  # Should not raise

    def test_validate_authority_level_invalid(self):
        with pytest.raises(ProvenanceError, match="Invalid authority level"):
            validate_authority_level("superuser")

    def test_authority_hierarchy(self):
        assert is_authority_at_least("accepted", "proposed")
        assert is_authority_at_least("deprecated", "accepted")
        assert is_authority_at_least("archived", "deprecated")
        assert not is_authority_at_least("proposed", "accepted")
        assert not is_authority_at_least("accepted", "deprecated")


class TestAuthorityTransitions:
    """Test authority level transitions."""

    def test_proposed_to_accepted(self):
        assert can_transition_authority("proposed", "accepted")

    def test_accepted_to_deprecated(self):
        assert can_transition_authority("accepted", "deprecated")

    def test_deprecated_to_archived(self):
        assert can_transition_authority("deprecated", "archived")

    def test_proposed_cannot_skip_to_deprecated(self):
        assert not can_transition_authority("proposed", "deprecated")

    def test_proposed_cannot_skip_to_archived(self):
        assert not can_transition_authority("proposed", "archived")

    def test_accepted_cannot_go_back_to_proposed(self):
        assert not can_transition_authority("accepted", "proposed")

    def test_archived_is_terminal(self):
        assert not can_transition_authority("archived", "proposed")
        assert not can_transition_authority("archived", "accepted")
        assert not can_transition_authority("archived", "deprecated")

    def test_invalid_authority_transition(self):
        with pytest.raises(ProvenanceError, match="Invalid authority level"):
            can_transition_authority("invalid", "accepted")


class TestSourceTypes:
    """Test source type constants."""

    def test_valid_sources(self):
        assert VALID_SOURCES == {"human", "pipeline", "import", "generated"}

    def test_all_sources_accepted(self):
        for source in VALID_SOURCES:
            prov = Provenance(author="alice", source=source)
            assert prov.source == source


class TestProvenanceEquality:
    """Test provenance equality and hashing."""

    def test_equal_provenance(self):
        p1 = Provenance(author="alice", source="human", timestamp="2026-01-01T00:00:00")
        p2 = Provenance(author="alice", source="human", timestamp="2026-01-01T00:00:00")
        assert p1 == p2

    def test_different_provenance(self):
        p1 = Provenance(author="alice", source="human", timestamp="2026-01-01T00:00:00")
        p2 = Provenance(author="bob", source="human", timestamp="2026-01-01T00:00:00")
        assert p1 != p2

    def test_hash_deterministic(self):
        p1 = Provenance(author="alice", source="human", timestamp="2026-01-01T00:00:00")
        p2 = Provenance(author="alice", source="human", timestamp="2026-01-01T00:00:00")
        assert hash(p1) == hash(p2)

    def test_hash_differs(self):
        p1 = Provenance(author="alice", source="human", timestamp="2026-01-01T00:00:00")
        p2 = Provenance(author="bob", source="human", timestamp="2026-01-01T00:00:00")
        assert hash(p1) != hash(p2)
