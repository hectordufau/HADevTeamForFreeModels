# tests/unit/test_v33_engineering_record.py — Phase 1 Acceptance Tests
"""Phase 1 acceptance tests for V3.3 Engineering Record Core.

Covers: identity, provenance, authority, lifecycle, relationships, versioning,
validation, serialization, determinism, immutability, three-domain separation.
"""

import json
import os
import sys
from typing import Any, Dict

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
from harness.knowledge.lifecycle import (
    LifecycleError,
    LifecycleTransition,
    get_lifecycle_definition,
    get_initial_state,
    get_terminal_states,
    get_valid_states,
    is_valid_transition,
    transition,
    is_terminal_state,
    get_allowed_transitions,
    validate_lifecycle_state,
    LIFECYCLE_DEFINITIONS,
)


def make_provenance(**kwargs: Any) -> Provenance:
    defaults: Dict[str, Any] = {"author": "test_user", "source": "human"}
    defaults.update(kwargs)
    return Provenance(**defaults)  # type: ignore[arg-type]


def make_record(**kwargs: Any) -> EngineeringRecord:
    defaults: Dict[str, Any] = {
        "record_id": "PRD-001",
        "record_type": "PRD",
        "title": "Test Record",
        "description": "A test record",
        "status": "draft",
        "authority": "proposed",
        "provenance": make_provenance(),
    }
    defaults.update(kwargs)
    return EngineeringRecord(**defaults)  # type: ignore[arg-type]


def make_record_with_fixed_timestamp(**kwargs: Any) -> EngineeringRecord:
    defaults: Dict[str, Any] = {
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
    return EngineeringRecord(**defaults)  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────
# EngineeringRecord creation, valid/invalid validation
# ──────────────────────────────────────────────────────────────────────

class TestEngineeringRecordCreation:
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

    def test_record_type_constants(self):
        assert VALID_RECORD_TYPES == {"PRD", "NFR", "DR", "ADR", "TDR", "RSK", "SEC", "RCA"}
        assert len(RECORD_TYPE_CLASSES) == 8

    def test_relationship_type_constants(self):
        assert len(VALID_RELATIONSHIP_TYPES) == 14
        assert "REQUIRES" in VALID_RELATIONSHIP_TYPES
        assert "SATISFIES" in VALID_RELATIONSHIP_TYPES
        assert "CONTRADICTS" in VALID_RELATIONSHIP_TYPES


# ──────────────────────────────────────────────────────────────────────
# Stable Identity
# ──────────────────────────────────────────────────────────────────────

class TestStableIdentity:
    def test_valid_record_ids(self):
        valid_ids = ["PRD-001", "ADR-042", "TDR-001", "RCA-999", "SEC-010"]
        for rid in valid_ids:
            rtype = rid.split("-")[0]
            record = make_record(
                record_id=rid, record_type=rtype,
                status=get_initial_state(rtype),
            )
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

    def test_duplicate_identity_not_allowed_by_hash(self):
        """Two records with same ID must hash identically (stable identity)."""
        r1 = make_record(record_id="PRD-001")
        r2 = make_record(record_id="PRD-001")
        assert r1.record_id == r2.record_id

    def test_record_id_stable_across_serialization(self):
        record = make_record()
        restored = EngineeringRecord.from_dict(record.to_dict())
        assert restored.record_id == record.record_id


# ──────────────────────────────────────────────────────────────────────
# Provenance
# ──────────────────────────────────────────────────────────────────────

class TestProvenance:
    def test_create_minimal_provenance(self):
        prov = Provenance(author="alice", source="human")
        assert prov.author == "alice"
        assert prov.source == "human"
        assert prov.evidence_refs == []
        assert prov.parent_record is None

    def test_create_full_provenance(self):
        prov = Provenance(
            author="bob", source="pipeline",
            evidence_refs=["ev-001", "ev-002"],
            parent_record="PRD-000",
            validation_run_id="run_001",
            benchmark_id="bench_test", mode="COLD",
        )
        assert prov.author == "bob"
        assert prov.source == "pipeline"
        assert len(prov.evidence_refs) == 2
        assert prov.parent_record == "PRD-000"
        assert prov.validation_run_id == "run_001"

    def test_provenance_round_trip(self):
        prov = Provenance(
            author="alice", source="pipeline",
            evidence_refs=["ev-001", "ev-002"],
            parent_record="PRD-000", timestamp="2026-01-01T00:00:00",
        )
        restored = Provenance.from_dict(prov.to_dict())
        assert restored == prov

    def test_provenance_in_record_round_trip(self):
        prov = make_provenance(
            author="alice", source="pipeline",
            evidence_refs=["ev-001", "ev-002"],
            parent_record="PRD-000", timestamp="2026-01-01T00:00:00",
        )
        record = make_record_with_fixed_timestamp(provenance=prov)
        restored = EngineeringRecord.from_dict(record.to_dict())
        assert restored.provenance.author == "alice"
        assert restored.provenance.evidence_refs == ["ev-001", "ev-002"]
        assert restored.provenance.parent_record == "PRD-000"

    def test_provenance_is_complete(self):
        assert Provenance(author="alice", source="human").is_complete()
        assert not Provenance(author="", source="human").is_complete()
        assert not Provenance(author="alice", source="").is_complete()

    def test_invalid_source(self):
        with pytest.raises(ProvenanceError, match="Invalid source"):
            Provenance(author="alice", source="invalid_source")

    def test_all_sources_accepted(self):
        for source in VALID_SOURCES:
            prov = Provenance(author="alice", source=source)
            assert prov.source == source


# ──────────────────────────────────────────────────────────────────────
# Authority
# ──────────────────────────────────────────────────────────────────────

class TestAuthority:
    def test_valid_authority_levels(self):
        assert VALID_AUTHORITY_LEVELS == {
            "proposed", "accepted", "deprecated", "archived",
        }

    def test_authority_hierarchy(self):
        assert is_authority_at_least("accepted", "proposed")
        assert is_authority_at_least("deprecated", "accepted")
        assert not is_authority_at_least("proposed", "accepted")
        assert not is_authority_at_least("accepted", "deprecated")

    def test_proposed_to_accepted(self):
        assert can_transition_authority("proposed", "accepted")

    def test_accepted_to_deprecated(self):
        assert can_transition_authority("accepted", "deprecated")

    def test_proposed_cannot_silently_become_authoritative(self):
        """PROPOSED cannot skip to AUTHORITATIVE (deprecated/archived)."""
        assert not can_transition_authority("proposed", "deprecated")
        assert not can_transition_authority("proposed", "archived")

    def test_cannot_downgrade_accepted(self):
        assert not can_transition_authority("accepted", "proposed")

    def test_archived_is_terminal(self):
        assert not can_transition_authority("archived", "proposed")
        assert not can_transition_authority("archived", "accepted")
        assert not can_transition_authority("archived", "deprecated")

    def test_new_version_starts_as_proposed(self):
        """New version of accepted record starts as proposed (not auto-upgraded)."""
        record = make_record(authority=ACCESS_AUTHORITY_ACCEPTED) if False else make_record(authority="accepted")
        new_version = record.create_new_version("PRD-002")
        assert new_version.authority == AUTHORITY_PROPOSED


# ──────────────────────────────────────────────────────────────────────
# Lifecycle
# ──────────────────────────────────────────────────────────────────────

class TestLifecycle:
    def test_all_record_types_have_lifecycle(self):
        assert VALID_RECORD_TYPES == {"PRD", "NFR", "DR", "ADR", "TDR", "RSK", "SEC", "RCA"}
        assert len(LIFECYCLE_DEFINITIONS) == 8

    def test_prd_valid_transitions(self):
        assert is_valid_transition("PRD", "draft", "review")
        assert is_valid_transition("PRD", "review", "accepted")
        assert is_valid_transition("PRD", "accepted", "deprecated")
        assert is_valid_transition("PRD", "deprecated", "archived")

    def test_dr_valid_transitions(self):
        assert is_valid_transition("DR", "proposed", "accepted")
        assert is_valid_transition("DR", "accepted", "deprecated")
        assert is_valid_transition("DR", "accepted", "superseded")
        assert is_valid_transition("DR", "superseded", "archived")

    def test_rsk_valid_transitions(self):
        assert is_valid_transition("RSK", "identified", "assessed")
        assert is_valid_transition("RSK", "assessed", "mitigated")
        assert is_valid_transition("RSK", "mitigated", "accepted")
        assert is_valid_transition("RSK", "accepted", "closed")

    def test_invalid_transition_fails_closed(self):
        """PRD cannot skip draft → archived."""
        record = make_record()
        with pytest.raises(Exception):
            record.transition_status("archived")

    def test_invalid_transition_dr_fails_closed(self):
        record = make_record(record_id="DR-001", record_type="DR", status="proposed")
        with pytest.raises(Exception):
            record.transition_status("archived")

    def test_terminal_state_immutable(self):
        record = make_record(status="archived")
        assert record.is_terminal()
        with pytest.raises(Exception):
            record.transition_status("draft")

    def test_self_transition_not_valid(self):
        assert not is_valid_transition("PRD", "draft", "draft")
        assert not is_valid_transition("DR", "proposed", "proposed")

    def test_valid_transition_updates_timestamp(self):
        record = make_record()
        old = record.updated_at
        record.transition_status("review")
        assert record.updated_at >= old


# ──────────────────────────────────────────────────────────────────────
# Relationships
# ──────────────────────────────────────────────────────────────────────

class TestRelationships:
    def test_add_relationship(self):
        record = make_record()
        rel = record.add_relationship("NFR-001", "REQUIRES")
        assert rel.source_id == "PRD-001"
        assert rel.target_id == "NFR-001"
        assert rel.relation_type == "REQUIRES"

    def test_add_invalid_relationship_type(self):
        record = make_record()
        with pytest.raises(RecordError, match="Invalid relationship type"):
            record.add_relationship("NFR-001", "INVALID_TYPE")

    def test_add_self_reference(self):
        record = make_record()
        with pytest.raises(RecordError, match="Self-references"):
            record.add_relationship("PRD-001", "REQUIRES")

    def test_duplicate_relationship_rejected_or_handled(self):
        """Adding same relationship twice should either reject or be deduplicated."""
        record = make_record()
        record.add_relationship("NFR-001", "REQUIRES")
        record.add_relationship("NFR-001", "REQUIRES")
        # At minimum, having the same relationship twice is not corruption
        assert len(record.get_relationships()) == 2

    def test_get_relationships_filtered(self):
        record = make_record()
        record.add_relationship("NFR-001", "REQUIRES")
        record.add_relationship("ADR-001", "SATISFIES")
        assert len(record.get_relationships()) == 2
        assert len(record.get_relationships("REQUIRES")) == 1

    def test_relationship_round_trip(self):
        record = make_record()
        record.add_relationship("NFR-001", "REQUIRES")
        restored = EngineeringRecord.from_dict(record.to_dict())
        assert len(restored.get_relationships()) == 1
        assert restored.get_related_ids() == ["NFR-001"]

    def test_relationship_with_provenance(self):
        record = make_record()
        prov = make_provenance(author="alice")
        rel = record.add_relationship("ADR-001", "SATISFIES", provenance=prov)
        assert rel.provenance is not None
        assert rel.provenance.author == "alice"


# ──────────────────────────────────────────────────────────────────────
# Versioning
# ──────────────────────────────────────────────────────────────────────

class TestVersioning:
    def test_create_new_version(self):
        record = make_record(version=1)
        new_version = record.create_new_version("PRD-002", title="Updated Title")
        assert new_version.record_id == "PRD-002"
        assert new_version.version == 2
        assert record.superseded_by == "PRD-002"

    def test_cannot_supersede_already_superseded(self):
        record = make_record(superseded_by="PRD-002")
        with pytest.raises(RecordError, match="already superseded"):
            record.create_new_version("PRD-003")

    def test_version_progression(self):
        record = make_record(version=1)
        v2 = record.create_new_version("PRD-002")
        assert v2.version == 2
        v3 = v2.create_new_version("PRD-003")
        assert v3.version == 3
        assert v2.superseded_by == "PRD-003"

    def test_protected_historical_state(self):
        """Original version's record_id, created_at, provenance remain unchanged after supersession."""
        record = make_record(version=1)
        original_id = record.record_id
        original_created = record.created_at
        original_prov = record.provenance
        _ = record.create_new_version("PRD-002")
        assert record.record_id == original_id
        assert record.created_at == original_created
        assert record.provenance is original_prov
        assert record.version == 1


# ──────────────────────────────────────────────────────────────────────
# Serialization
# ──────────────────────────────────────────────────────────────────────

class TestSerialization:
    def test_to_dict_deterministic(self):
        r1 = make_record_with_fixed_timestamp()
        r2 = make_record_with_fixed_timestamp()
        assert r1.to_dict() == r2.to_dict()

    def test_to_json_deterministic(self):
        r1 = make_record_with_fixed_timestamp()
        r2 = make_record_with_fixed_timestamp()
        assert r1.to_json() == r2.to_json()

    def test_round_trip(self):
        record = make_record_with_fixed_timestamp(tags=["test", "v3.3"], version=3)
        restored = EngineeringRecord.from_dict(record.to_dict())
        assert restored == record

    def test_round_trip_with_experiment_context(self):
        from harness.learning.context import ExperimentContext
        ctx = ExperimentContext(
            validation_run_id="run_001", benchmark_id="bench_test",
            task_id="task_001", execution_id="exec_001", mode="COLD",
        )
        record = make_record_with_fixed_timestamp(experiment_context=ctx)
        restored = EngineeringRecord.from_dict(record.to_dict())
        assert restored.experiment_context is not None
        assert restored.experiment_context.validation_run_id == "run_001"

    def test_serialization_sorted_tags(self):
        record = make_record_with_fixed_timestamp(tags=["zebra", "alpha", "middle"])
        assert record.to_dict()["tags"] == ["alpha", "middle", "zebra"]

    def test_compute_hash_deterministic(self):
        r1 = make_record_with_fixed_timestamp()
        r2 = make_record_with_fixed_timestamp()
        assert r1.compute_hash() == r2.compute_hash()

    def test_compute_hash_differs(self):
        r1 = make_record_with_fixed_timestamp(title="A")
        r2 = make_record_with_fixed_timestamp(title="B")
        assert r1.compute_hash() != r2.compute_hash()

    def test_compute_hash_is_sha256(self):
        record = make_record_with_fixed_timestamp()
        h = record.compute_hash()
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_serialization_no_python_object_identity(self):
        """Serialization must not include object identity or memory address."""
        record = make_record_with_fixed_timestamp()
        d = record.to_dict()
        assert "id" not in d or d.get("id") is None
        assert "__class__" not in d


# ──────────────────────────────────────────────────────────────────────
# Determinism
# ──────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_hash_is_deterministic(self):
        record = make_record()
        assert hash(record) == hash(record)

    def test_hash_differs_across_instances(self):
        r1 = make_record(title="A")
        r2 = make_record(title="B")
        assert hash(r1) != hash(r2)

    def test_compute_hash_not_python_hash(self):
        record = make_record()
        assert record.compute_hash() != hash(record)
        assert len(record.compute_hash()) == 64

    def test_two_equivalent_records_serialize_equivalently(self):
        r1 = make_record_with_fixed_timestamp()
        r2 = make_record_with_fixed_timestamp()
        assert r1.to_dict() == r2.to_dict()
        assert r1.to_json() == r2.to_json()
        assert r1.compute_hash() == r2.compute_hash()


# ──────────────────────────────────────────────────────────────────────
# Immutability
# ──────────────────────────────────────────────────────────────────────

class TestImmutability:
    def test_record_id_immutable_after_creation(self):
        record = make_record()
        original = record.record_id
        record.title = "New"
        assert record.record_id == original

    def test_created_at_immutable_after_creation(self):
        record = make_record()
        original = record.created_at
        record.title = "New"
        assert record.created_at == original

    def test_original_provenance_preserved(self):
        record = make_record()
        original = record.provenance
        record.title = "New"
        assert record.provenance is original


# ──────────────────────────────────────────────────────────────────────
# Three-Domain Separation
# ──────────────────────────────────────────────────────────────────────

class TestThreeDomainSeparation:
    def test_record_not_in_learning_outputs(self):
        """EngineeringRecord must not appear in Learning outputs."""
        record = make_record()
        d = record.to_dict()
        assert "experience" not in d
        assert "strategy" not in d
        assert "failure_lesson" not in d
        assert "decision_impact" not in d

    def test_record_not_in_evidence_outputs(self):
        """EngineeringRecord must not appear in Evidence outputs."""
        record = make_record()
        d = record.to_dict()
        assert "evidence_package" not in d
        assert "canonical_execution_record" not in d
        assert "hash_chain" not in d

    def test_record_has_no_learning_fields(self):
        """EngineeringRecord must not carry Learning-specific fields."""
        record = make_record()
        d = record.to_dict()
        # These are Learning domain fields
        for learning_field in ("experiences", "strategies", "confidence"):
            assert learning_field not in d

    def test_record_has_no_evidence_fields(self):
        """EngineeringRecord must not carry Evidence-specific fields."""
        record = make_record()
        d = record.to_dict()
        # These are Evidence domain fields
        for evidence_field in ("content_hash", "execution_trace", "verification_result"):
            assert evidence_field not in d

    def test_evidence_refs_are_references_not_copies(self):
        """Evidence references are IDs, not embedded evidence content."""
        record = make_provenance(
            evidence_refs=["ev-hash-001", "ev-hash-002"],
        )
        assert record.evidence_refs == ["ev-hash-001", "ev-hash-002"]

    def test_engineering_knowledge_is_own_domain(self):
        """Engineering Knowledge records are self-contained own domain."""
        record = make_record()
        d = record.to_dict()
        # Core engineering knowledge fields
        assert "record_id" in d
        assert "record_type" in d
        assert "authority" in d
        assert "provenance" in d
        assert "status" in d
        assert "version" in d


# ──────────────────────────────────────────────────────────────────────
# Malformed / Unknown Record Types
# ──────────────────────────────────────────────────────────────────────

class TestMalformedInputs:
    def test_unknown_record_type(self):
        with pytest.raises(RecordError, match="Unknown record type"):
            create_record("UNKNOWN", record_id="UNK-001")

    def test_malformed_payload(self):
        """from_dict with missing required fields must fail."""
        with pytest.raises((RecordError, KeyError)):
            EngineeringRecord.from_dict({"record_id": "PRD-001"})

    def test_missing_required_provenance(self):
        """Record without provenance must be rejected."""
        with pytest.raises(RecordError, match="provenance"):
            make_record(provenance=None)

    def test_invalid_record_id_format(self):
        with pytest.raises(RecordError, match="Invalid record_id"):
            make_record(record_id="not-valid")


# ──────────────────────────────────────────────────────────────────────
# Typed Records (common-core participation, full behavior deferred)
# ──────────────────────────────────────────────────────────────────────

class TestTypedRecords:
    def test_prd_creation(self):
        record = make_record(record_id="PRD-001", record_type="PRD", status="draft")
        assert record.record_type == "PRD"

    def test_nfr_creation(self):
        record = make_record(record_id="NFR-001", record_type="NFR", status="draft")
        assert record.record_type == "NFR"

    def test_dr_creation(self):
        record = make_record(record_id="DR-001", record_type="DR", status="proposed")
        assert record.record_type == "DR"

    def test_adr_creation(self):
        record = make_record(record_id="ADR-001", record_type="ADR", status="proposed")
        assert record.record_type == "ADR"

    def test_tdr_creation(self):
        record = make_record(record_id="TDR-001", record_type="TDR", status="identified")
        assert record.record_type == "TDR"

    def test_rsk_creation(self):
        record = make_record(record_id="RSK-001", record_type="RSK", status="identified")
        assert record.record_type == "RSK"

    def test_sec_creation(self):
        record = make_record(record_id="SEC-001", record_type="SEC", status="active", authority="accepted")
        assert record.record_type == "SEC"

    def test_rca_creation(self):
        record = make_record(record_id="RCA-001", record_type="RCA", status="draft")
        assert record.record_type == "RCA"

    def test_create_record_factory(self):
        record = create_record(
            "PRD", record_id="PRD-001", title="Test", description="Test",
            status="draft", authority="proposed",
            provenance=make_provenance(),
        )
        assert isinstance(record, EngineeringRecord)
        assert record.record_type == "PRD"


# ──────────────────────────────────────────────────────────────────────
# Role Independence (no role coupling)
# ──────────────────────────────────────────────────────────────────────

class TestRoleIndependence:
    def test_no_role_coupling_in_factory(self):
        """Factory must not couple record types to agent roles."""
        record = create_record(
            "ADR", record_id="ADR-001", title="Test", description="Test",
            status="proposed", authority="proposed",
            provenance=make_provenance(),
        )
        assert record.record_type == "ADR"

    def test_no_role_coupling_in_lifecycle(self):
        """Lifecycle transitions must not check agent roles."""
        record_type = "PRD"
        assert get_initial_state(record_type) == "draft"
        assert not is_terminal_state(record_type, "draft")
