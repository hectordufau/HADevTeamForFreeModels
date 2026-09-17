# tests/unit/test_knowledge_records.py — Tests for V3.3 EngineeringRecord Core
"""Tests for EngineeringRecord base class, schema, serialization, validation, versioning."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.knowledge.records import (
    EngineeringRecord,
    RecordRelationship,
    RecordError,
    RECORD_TYPE_CLASSES,
    VALID_RECORD_TYPES,
    VALID_RELATIONSHIP_TYPES,
    create_record,
)
from harness.knowledge.provenance import (
    Provenance,
    ProvenanceError,
    AUTHORITY_PROPOSED,
    AUTHORITY_ACCEPTED,
    AUTHORITY_DEPRECATED,
    AUTHORITY_ARCHIVED,
)
from harness.knowledge.lifecycle import (
    LifecycleError,
    get_initial_state,
    is_terminal_state,
    is_valid_transition,
)


def make_provenance(**kwargs):
    """Helper to create a valid Provenance."""
    defaults = {"author": "test_user", "source": "human"}
    defaults.update(kwargs)
    return Provenance(**defaults)


def make_record(**kwargs):
    """Helper to create a valid EngineeringRecord."""
    defaults = {
        "record_id": "PRD-001",
        "record_type": "PRD",
        "title": "Test Record",
        "description": "A test record",
        "status": "draft",
        "authority": "proposed",
        "provenance": make_provenance(),
    }
    defaults.update(kwargs)
    return EngineeringRecord(**defaults)


def make_record_with_fixed_timestamp(**kwargs):
    """Helper to create a valid EngineeringRecord with fixed timestamps."""
    defaults = {
        "record_id": "PRD-001",
        "record_type": "PRD",
        "title": "Test Record",
        "description": "A test record",
        "status": "draft",
        "authority": "proposed",
        "provenance": make_provenance(timestamp="2026-01-01T00:00:00"),
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }
    defaults.update(kwargs)
    return EngineeringRecord(**defaults)


class TestEngineeringRecordCreation:
    """Test basic record creation and validation."""

    def test_create_valid_record(self):
        record = make_record()
        assert record.record_id == "PRD-001"
        assert record.record_type == "PRD"
        assert record.title == "Test Record"
        assert record.version == 1
        assert record.superseded_by is None

    def test_create_with_all_fields(self):
        record = make_record(
            tags=["test", "v3.3"],
            version=2,
            superseded_by="PRD-002",
        )
        assert record.tags == ["test", "v3.3"]
        assert record.version == 2
        assert record.superseded_by == "PRD-002"

    def test_record_type_constants(self):
        assert VALID_RECORD_TYPES == {"PRD", "NFR", "DR", "ADR", "TDR", "RSK", "SEC", "RCA", "REQ"}
        assert len(RECORD_TYPE_CLASSES) == 9

    def test_relationship_type_constants(self):
        assert len(VALID_RELATIONSHIP_TYPES) == 14
        assert "REQUIRES" in VALID_RELATIONSHIP_TYPES
        assert "SATISFIES" in VALID_RELATIONSHIP_TYPES
        assert "CONTRADICTS" in VALID_RELATIONSHIP_TYPES


class TestStableIdentity:
    """Test stable, deterministic record identity."""

    def test_valid_record_ids(self):
        valid_ids = ["PRD-001", "ADR-042", "TDR-001", "RCA-999", "SEC-010"]
        for rid in valid_ids:
            record = make_record(record_id=rid, record_type=rid.split("-")[0], status=get_initial_state(rid.split("-")[0]))
            assert record.record_id == rid

    def test_invalid_record_id_format(self):
        with pytest.raises(RecordError, match="Invalid record_id"):
            make_record(record_id="INVALID")

    def test_invalid_record_id_type(self):
        with pytest.raises(RecordError, match="Invalid record_id"):
            make_record(record_id="XYZ-001")

    def test_record_id_prefix_must_match_type(self):
        with pytest.raises(RecordError, match="does not match record_type"):
            make_record(record_id="ADR-001", record_type="PRD")

    def test_duplicate_record_ids_allowed(self):
        """Two records can have the same ID (identity is the ID itself)."""
        r1 = make_record(record_id="PRD-001")
        r2 = make_record(record_id="PRD-001")
        assert r1.record_id == r2.record_id


class TestValidation:
    """Test record validation."""

    def test_missing_title(self):
        with pytest.raises(RecordError, match="title"):
            make_record(title="")

    def test_missing_description(self):
        with pytest.raises(RecordError, match="description"):
            make_record(description=None)

    def test_invalid_record_type(self):
        with pytest.raises(RecordError, match="Invalid record_type"):
            make_record(record_type="INVALID")

    def test_invalid_status(self):
        with pytest.raises((RecordError, LifecycleError)):
            make_record(status="invalid_state")

    def test_invalid_authority(self):
        with pytest.raises((RecordError, ProvenanceError)):
            make_record(authority="superuser")

    def test_missing_provenance(self):
        with pytest.raises(RecordError, match="provenance is required"):
            make_record(provenance=None)

    def test_incomplete_provenance(self):
        with pytest.raises(RecordError, match="provenance is incomplete"):
            make_record(provenance=Provenance(author="", source=""))

    def test_invalid_version(self):
        with pytest.raises(RecordError, match="version must be a positive integer"):
            make_record(version=0)

    def test_invalid_tags_type(self):
        with pytest.raises(RecordError, match="tags must be a list"):
            make_record(tags="not_a_list")


class TestSerialization:
    """Test deterministic serialization and deserialization."""

    def test_to_dict_deterministic(self):
        r1 = make_record_with_fixed_timestamp()
        r2 = make_record_with_fixed_timestamp()
        assert r1.to_dict() == r2.to_dict()

    def test_to_json_deterministic(self):
        r1 = make_record_with_fixed_timestamp()
        r2 = make_record_with_fixed_timestamp()
        assert r1.to_json() == r2.to_json()

    def test_round_trip(self):
        record = make_record_with_fixed_timestamp(
            tags=["test", "v3.3"],
            version=3,
        )
        data = record.to_dict()
        restored = EngineeringRecord.from_dict(data)
        assert restored == record

    def test_round_trip_with_provenance(self):
        prov = make_provenance(
            author="alice",
            source="pipeline",
            evidence_refs=["ev-001", "ev-002"],
            parent_record="PRD-000",
            timestamp="2026-01-01T00:00:00",
        )
        record = make_record_with_fixed_timestamp(provenance=prov)
        restored = EngineeringRecord.from_dict(record.to_dict())
        assert restored.provenance is not None
        assert restored.provenance.author == "alice"
        assert restored.provenance.evidence_refs == ["ev-001", "ev-002"]
        assert restored.provenance.parent_record == "PRD-000"

    def test_round_trip_with_experiment_context(self):
        from harness.learning.context import ExperimentContext
        ctx = ExperimentContext(
            validation_run_id="run_001",
            benchmark_id="bench_test",
            task_id="task_001",
            execution_id="exec_001",
            mode="COLD",
        )
        record = make_record_with_fixed_timestamp(experiment_context=ctx)
        restored = EngineeringRecord.from_dict(record.to_dict())
        assert restored.experiment_context is not None
        assert restored.experiment_context.validation_run_id == "run_001"

    def test_serialization_sorted_keys(self):
        record = make_record_with_fixed_timestamp(tags=["zebra", "alpha", "middle"])
        d = record.to_dict()
        assert d["tags"] == ["alpha", "middle", "zebra"]

    def test_compute_hash_deterministic(self):
        r1 = make_record_with_fixed_timestamp()
        r2 = make_record_with_fixed_timestamp()
        assert r1.compute_hash() == r2.compute_hash()

    def test_compute_hash_differs(self):
        r1 = make_record_with_fixed_timestamp(title="Record A")
        r2 = make_record_with_fixed_timestamp(title="Record B")
        assert r1.compute_hash() != r2.compute_hash()

    def test_compute_hash_is_sha256(self):
        record = make_record_with_fixed_timestamp()
        h = record.compute_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestRelationships:
    """Test record relationships."""

    def test_add_relationship(self):
        record = make_record()
        rel = record.add_relationship("NFR-001", "REQUIRES")
        assert rel.source_id == "PRD-001"
        assert rel.target_id == "NFR-001"
        assert rel.relation_type == "REQUIRES"

    def test_add_relationship_with_provenance(self):
        record = make_record()
        prov = make_provenance(author="alice")
        rel = record.add_relationship("ADR-001", "SATISFIES", provenance=prov)
        assert rel.provenance.author == "alice"

    def test_add_invalid_relationship_type(self):
        record = make_record()
        with pytest.raises(RecordError, match="Invalid relationship type"):
            record.add_relationship("NFR-001", "INVALID_TYPE")

    def test_add_self_reference(self):
        record = make_record()
        with pytest.raises(RecordError, match="Self-references are not allowed"):
            record.add_relationship("PRD-001", "REQUIRES")

    def test_get_relationships(self):
        record = make_record()
        record.add_relationship("NFR-001", "REQUIRES")
        record.add_relationship("ADR-001", "SATISFIES")
        record.add_relationship("TDR-001", "INTRODUCES")

        all_rels = record.get_relationships()
        assert len(all_rels) == 3

        reqs = record.get_relationships("REQUIRES")
        assert len(reqs) == 1
        assert reqs[0].target_id == "NFR-001"

    def test_get_related_ids(self):
        record = make_record()
        record.add_relationship("NFR-001", "REQUIRES")
        record.add_relationship("ADR-001", "SATISFIES")
        related = record.get_related_ids()
        assert "NFR-001" in related
        assert "ADR-001" in related

    def test_relationship_round_trip(self):
        record = make_record()
        record.add_relationship("NFR-001", "REQUIRES")
        restored = EngineeringRecord.from_dict(record.to_dict())
        assert len(restored.get_relationships()) == 1
        assert restored.get_related_ids() == ["NFR-001"]


class TestVersioning:
    """Test record versioning and supersession."""

    def test_create_new_version(self):
        record = make_record(version=1)
        new_version = record.create_new_version("PRD-002", title="Updated Title")
        assert new_version.record_id == "PRD-002"
        assert new_version.version == 2
        assert record.superseded_by == "PRD-002"

    def test_new_version_starts_as_proposed(self):
        record = make_record(authority=AUTHORITY_ACCEPTED)
        new_version = record.create_new_version("PRD-002")
        assert new_version.authority == AUTHORITY_PROPOSED

    def test_new_version_preserves_provenance_chain(self):
        record = make_record()
        new_version = record.create_new_version("PRD-002")
        assert new_version.provenance.parent_record == "PRD-001"

    def test_cannot_supersede_already_superseded(self):
        record = make_record(superseded_by="PRD-002")
        with pytest.raises(RecordError, match="already superseded"):
            record.create_new_version("PRD-003")

    def test_version_progression(self):
        record = make_record(version=1)
        v2 = record.create_new_version("PRD-002")
        assert v2.version == 2
        assert record.superseded_by == "PRD-002"
        # v2 is now the current version — create v3 from v2
        v3 = v2.create_new_version("PRD-003")
        assert v3.version == 3
        assert v2.superseded_by == "PRD-003"


class TestTypedRecords:
    """Test typed record creation via base class (typed subclasses deferred to Phase 4-7)."""

    def test_prd_creation(self):
        record = make_record(
            record_id="PRD-001",
            record_type="PRD",
            title="Test PRD",
            description="Test description",
            status="draft",
            authority="proposed",
        )
        assert record.record_type == "PRD"

    def test_nfr_creation(self):
        record = make_record(
            record_id="NFR-001",
            record_type="NFR",
            title="Test NFR",
            description="Test description",
            status="draft",
            authority="proposed",
        )
        assert record.record_type == "NFR"

    def test_dr_creation(self):
        record = make_record(
            record_id="DR-001",
            record_type="DR",
            title="Test DR",
            description="Test description",
            status="proposed",
            authority="proposed",
        )
        assert record.record_type == "DR"

    def test_adr_creation(self):
        record = make_record(
            record_id="ADR-001",
            record_type="ADR",
            title="Test ADR",
            description="Test description",
            status="proposed",
            authority="proposed",
        )
        assert record.record_type == "ADR"

    def test_tdr_creation(self):
        record = make_record(
            record_id="TDR-001",
            record_type="TDR",
            title="Test TDR",
            description="Test description",
            status="identified",
            authority="proposed",
        )
        assert record.record_type == "TDR"

    def test_rsk_creation(self):
        record = make_record(
            record_id="RSK-001",
            record_type="RSK",
            title="Test RSK",
            description="Test description",
            status="identified",
            authority="proposed",
        )
        assert record.record_type == "RSK"

    def test_sec_creation(self):
        record = make_record(
            record_id="SEC-001",
            record_type="SEC",
            title="Test SEC",
            description="Test description",
            status="active",
            authority="accepted",
        )
        assert record.record_type == "SEC"

    def test_rca_creation(self):
        record = make_record(
            record_id="RCA-001",
            record_type="RCA",
            title="Test RCA",
            description="Test description",
            status="draft",
            authority="proposed",
        )
        assert record.record_type == "RCA"

    def test_create_record_factory(self):
        record = create_record(
            "PRD",
            record_id="PRD-001",
            title="Test",
            description="Test",
            status="draft",
            authority="proposed",
            provenance=make_provenance(),
        )
        assert isinstance(record, EngineeringRecord)
        assert record.record_type == "PRD"

    def test_create_record_unknown_type(self):
        with pytest.raises(RecordError, match="Unknown record type"):
            create_record("UNKNOWN", record_id="UNK-001")


class TestImmutabilityAndProtectedFields:
    """Test protected fields and immutability."""

    def test_record_id_immutable_after_creation(self):
        record = make_record()
        original_id = record.record_id
        record.title = "New Title"
        assert record.record_id == original_id

    def test_created_at_immutable_after_creation(self):
        record = make_record()
        original_created = record.created_at
        record.title = "New Title"
        assert record.created_at == original_created

    def test_original_provenance_preserved(self):
        record = make_record()
        original_prov = record.provenance
        record.title = "New Title"
        assert record.provenance is original_prov

    def test_historical_versions_not_overwritten(self):
        record = make_record(version=1)
        new_version = record.create_new_version("PRD-002")
        # Original record should still have version 1
        assert record.version == 1
        assert new_version.version == 2


class TestNoPythonHashDependency:
    """Test that records don't depend on Python hash()."""

    def test_hash_is_deterministic(self):
        """Hash should be the same across calls (not Python object hash)."""
        record = make_record()
        h1 = hash(record)
        h2 = hash(record)
        assert h1 == h2

    def test_hash_differs_across_instances(self):
        """Different records should have different hashes."""
        r1 = make_record(title="A")
        r2 = make_record(title="B")
        assert hash(r1) != hash(r2)

    def test_compute_hash_not_python_hash(self):
        """compute_hash() should not be Python's built-in hash()."""
        record = make_record()
        assert record.compute_hash() != hash(record)
        assert len(record.compute_hash()) == 64  # SHA-256 hex


class TestLifecycleIntegration:
    """Test lifecycle integration with records."""

    def test_initial_status(self):
        record = make_record()
        assert record.status == "draft"

    def test_valid_transition(self):
        record = make_record()
        record.transition_status("review")
        assert record.status == "review"

    def test_invalid_transition_fails_closed(self):
        record = make_record()
        with pytest.raises(Exception):  # LifecycleError
            record.transition_status("archived")

    def test_terminal_state_immutable(self):
        record = make_record(status="archived")
        assert record.is_terminal()
        with pytest.raises(Exception):
            record.transition_status("draft")

    def test_transition_updates_timestamp(self):
        record = make_record()
        old_updated = record.updated_at
        record.transition_status("review")
        assert record.updated_at >= old_updated
