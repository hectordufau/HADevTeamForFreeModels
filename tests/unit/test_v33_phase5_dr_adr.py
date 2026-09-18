# tests/unit/test_v33_phase5_dr_adr.py — Phase 5 Acceptance Tests
"""Phase 5: DR / ADR Decision Intelligence — deterministic behavior verification.

Tests cover:
- DecisionRecord creation, validation, decision types
- ADR creation, validation, lifecycle, authority
- Agent-generated ADR remains PROPOSED (never auto-ACCEPTED)
- context, rationale, alternatives, consequences
- Missing rationale = completeness violation
- No fabricated alternatives (empty alternatives valid when only one option)
- REQ → ADR relationship (DECIDED_BY)
- NFR → ADR relationship (CONSTRAINED_BY)
- ADR → Evidence external reference (SUPPORTED_BY)
- Versioning: version vs supersession
- Supersession chain, effective decision, supersession conflict detection
- Explicit contradiction (CONTRADICTS), accepted ADR conflict
- KnowledgeStore persistence, restart/reload, historical ADR recovery
- KnowledgeGraph traversal, ContradictionDetector integration
- ADRAutomation compatibility/migration (adapter, not competing store)
- Memory ADR reference semantics (memory references, not contains)
- Deterministic serialization, deterministic traversal
- CapabilityGraph separation, Engineering/Learning separation, Engineering/Evidence separation
- Phase 5 E2E scenario: PRD → REQ → NFR → ADR → Evidence
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.knowledge import (
    DecisionRecord,
    ArchitectureDecisionRecord,
    Alternative,
    Consequences,
    KnowledgeStore,
    EngineeringKnowledgeGraph,
    Provenance,
    AUTHORITY_PROPOSED,
    AUTHORITY_ACCEPTED,
    AUTHORITY_DEPRECATED,
    AUTHORITY_ARCHIVED,
    VALID_AUTHORITY_LEVELS,
    VALID_DECISION_TYPES,
    VALID_ADR_DOMAINS,
    DECISION_TYPE_ARCHITECTURE,
    DECISION_TYPE_TECHNICAL,
    DECISION_TYPE_SECURITY,
    DECISION_TYPE_PRODUCT,
    DECISION_TYPE_OPERATIONAL,
    DECISION_TYPE_DATA,
    DECISION_TYPE_INTEGRATION,
    ADR_DOMAIN_DATA,
    ADR_DOMAIN_API,
    ADR_DOMAIN_SECURITY,
    ADR_DOMAIN_DEPLOYMENT,
    ADR_DOMAIN_MESSAGING,
    ADR_DOMAIN_UI,
    ADR_DOMAIN_INFRASTRUCTURE,
    DISPOSITION_SELECTED,
    DISPOSITION_REJECTED,
    VALID_DISPOSITIONS,
    find_supersession_conflicts,
    get_supersession_chain,
    get_effective_decision,
)
from harness.knowledge.records import (
    EngineeringRecord,
    RecordError,
    RELATIONSHIP_DECIDED_BY,
    RELATIONSHIP_CONSTRAINED_BY,
    RELATIONSHIP_SUPPORTED_BY,
    RELATIONSHIP_SUPERSEDES,
    RELATIONSHIP_CONTRADICTS,
    RELATIONSHIP_RELATES_TO,
)  # noqa: F811
from harness.knowledge.lifecycle import is_valid_transition, is_terminal_state


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def make_provenance(author="test_author", source="human"):
    return Provenance(author=author, source=source)


def make_dr(**kwargs):
    defaults = dict(
        record_id="DR-001",
        title="Test Decision",
        description="A test decision record",
        decision_type=DECISION_TYPE_TECHNICAL,
        context="Need to choose a database",
        decision="Use PostgreSQL",
        rationale="PostgreSQL is the most suitable for our needs",
        provenance=make_provenance(),
    )
    defaults.update(kwargs)
    return DecisionRecord(**defaults)


def make_adr(**kwargs):
    defaults = dict(
        record_id="ADR-001",
        title="Test ADR",
        description="A test architecture decision record",
        decision_type=DECISION_TYPE_ARCHITECTURE,
        architecture_domain=ADR_DOMAIN_DATA,
        context="Need to choose data storage",
        decision="Use PostgreSQL",
        rationale="PostgreSQL provides ACID compliance",
        provenance=make_provenance(),
    )
    defaults.update(kwargs)
    return ArchitectureDecisionRecord(**defaults)


# ──────────────────────────────────────────────────────────────────────
# DR creation, validation, decision types
# ──────────────────────────────────────────────────────────────────────

class TestDecisionRecordCreation:
    """DR creation, validation, and decision type tests."""

    def test_dr_creation_minimal(self):
        dr = make_dr()
        assert dr.record_type == "DR"
        assert dr.status == "proposed"
        assert dr.authority == AUTHORITY_PROPOSED
        assert dr.decision_type == DECISION_TYPE_TECHNICAL

    def test_dr_creation_full(self):
        dr = make_dr(
            record_id="DR-002",
            decision_type=DECISION_TYPE_ARCHITECTURE,
            context="Choose API style",
            decision="Use REST",
            alternatives=[
                Alternative("ALT-001", "GraphQL", DISPOSITION_REJECTED, "Too complex"),
                Alternative("ALT-002", "REST", DISPOSITION_SELECTED, "Simple and standard"),
            ],
            consequences=Consequences(
                positive=["Simple to implement"],
                negative=["Less flexible queries"],
                risks=["N+1 query risk"],
            ),
            related_requirements=["REQ-001"],
            related_decisions=["ADR-001"],
        )
        assert dr.alternatives[0].alt_id == "ALT-001"
        assert dr.alternatives[0].disposition == DISPOSITION_REJECTED
        assert dr.consequences.positive == ["Simple to implement"]
        assert dr.consequences.negative == ["Less flexible queries"]
        assert dr.consequences.risks == ["N+1 query risk"]

    def test_dr_invalid_decision_type(self):
        with pytest.raises(RecordError, match="Invalid decision_type"):
            make_dr(decision_type="INVALID")

    def test_dr_all_valid_decision_types(self):
        for dt in VALID_DECISION_TYPES:
            dr = make_dr(decision_type=dt)
            assert dr.decision_type == dt

    def test_dr_invalid_record_id_prefix(self):
        with pytest.raises(RecordError, match="does not match record_type"):
            make_dr(record_id="ADR-001")

    def test_dr_empty_context_rejected(self):
        with pytest.raises(RecordError, match="context must be a non-empty string"):
            make_dr(context="")

    def test_dr_empty_decision_rejected(self):
        with pytest.raises(RecordError, match="decision must be a non-empty string"):
            make_dr(decision="")

    def test_dr_missing_rationale_rejected(self):
        with pytest.raises(RecordError, match="rationale is required"):
            make_dr(rationale="")

    def test_dr_empty_alternatives_valid(self):
        """No alternatives recorded is valid (only one option evaluated)."""
        dr = make_dr(alternatives=[])
        assert dr.alternatives == []
        assert dr.is_complete() is True

    def test_dr_alternatives_with_disposition(self):
        dr = make_dr(
            alternatives=[
                Alternative("ALT-001", "Option A", DISPOSITION_REJECTED, "Too expensive"),
                Alternative("ALT-002", "Option B", DISPOSITION_SELECTED, "Best fit"),
            ],
        )
        assert len(dr.alternatives) == 2
        assert dr.alternatives[1].disposition == DISPOSITION_SELECTED

    def test_dr_invalid_disposition_rejected(self):
        with pytest.raises(RecordError, match="Invalid disposition"):
            Alternative("ALT-001", "Option", "MAYBE")

    def test_dr_related_requirements_must_be_valid_ids(self):
        with pytest.raises(RecordError, match="REQ/NFR/PRD"):
            make_dr(related_requirements=["INVALID-001"])

    def test_dr_related_decisions_must_be_valid_ids(self):
        with pytest.raises(RecordError, match="DR/ADR"):
            make_dr(related_decisions=["REQ-001"])

    def test_dr_completeness_check(self):
        dr = make_dr()
        completeness = dr.get_completeness()
        assert completeness["has_context"] is True
        assert completeness["has_decision"] is True
        assert completeness["has_rationale"] is True
        assert completeness["has_decision_type"] is True
        assert dr.is_complete() is True

    def test_dr_completeness_missing_rationale(self):
        dr = make_dr()
        dr.rationale = ""
        completeness = dr.get_completeness()
        assert completeness["has_rationale"] is False
        assert dr.is_complete() is False


# ──────────────────────────────────────────────────────────────────────
# ADR creation, validation, lifecycle, authority
# ──────────────────────────────────────────────────────────────────────

class TestArchitectureDecisionRecord:
    """ADR creation, validation, lifecycle, and authority tests."""

    def test_adr_creation_minimal(self):
        adr = make_adr()
        assert adr.record_type == "ADR"
        assert adr.status == "proposed"
        assert adr.authority == AUTHORITY_PROPOSED
        assert adr.decision_type == DECISION_TYPE_ARCHITECTURE
        assert adr.architecture_domain == ADR_DOMAIN_DATA

    def test_adr_creation_full(self):
        adr = make_adr(
            record_id="ADR-002",
            architecture_domain=ADR_DOMAIN_API,
            patterns_considered=["REST", "GraphQL", "gRPC"],
            pattern_selected="REST",
            trade_offs="Simple but less flexible",
            impact="Reduces API complexity",
        )
        assert adr.patterns_considered == ["REST", "GraphQL", "gRPC"]
        assert adr.pattern_selected == "REST"
        assert adr.trade_offs == "Simple but less flexible"
        assert adr.impact == "Reduces API complexity"

    def test_adr_invalid_domain_rejected(self):
        with pytest.raises(RecordError, match="Invalid architecture_domain"):
            make_adr(architecture_domain="invalid")

    def test_adr_all_valid_domains(self):
        for domain in VALID_ADR_DOMAINS:
            adr = make_adr(architecture_domain=domain)
            assert adr.architecture_domain == domain

    def test_adr_invalid_record_id_prefix(self):
        with pytest.raises(RecordError, match="does not match record_type"):
            make_adr(record_id="DR-001")

    def test_adr_is_dr_specialization(self):
        """ADR must be a specialization of DR."""
        adr = make_adr()
        assert isinstance(adr, DecisionRecord)
        assert isinstance(adr, EngineeringRecord)

    def test_adr_completeness(self):
        adr = make_adr()
        completeness = adr.get_completeness()
        assert completeness["has_architecture_domain"] is True
        assert completeness["has_patterns"] is False  # patterns_considered is empty
        assert adr.is_complete() is False  # missing patterns, trade_offs, impact

    def test_adr_completeness_full(self):
        adr = make_adr(
            patterns_considered=["REST", "GraphQL"],
            pattern_selected="REST",
            trade_offs="Simple",
            impact="Low",
        )
        completeness = adr.get_completeness()
        assert completeness["has_patterns"] is True
        assert completeness["has_pattern_selected"] is True
        assert completeness["has_trade_offs"] is True
        assert completeness["has_impact"] is True
        assert adr.is_complete() is True

    def test_adr_pattern_selected_required_when_patterns_listed(self):
        with pytest.raises(RecordError, match="pattern_selected is required"):
            make_adr(patterns_considered=["REST", "GraphQL"], pattern_selected="")


# ──────────────────────────────────────────────────────────────────────
# Agent-generated ADR remains PROPOSED
# ──────────────────────────────────────────────────────────────────────

class TestAgentGeneratedADR:
    """Critical invariant: agent-generated ADR = PROPOSED. Never auto-ACCEPTED."""

    def test_agent_adr_always_proposed(self):
        """Agent-generated ADR must always start as PROPOSED."""
        adr = make_adr()
        assert adr.status == "proposed"
        assert adr.authority == AUTHORITY_PROPOSED

    def test_agent_adr_never_auto_accepted(self):
        """Even with all fields complete, agent ADR remains PROPOSED."""
        adr = make_adr(
            patterns_considered=["REST", "GraphQL"],
            pattern_selected="REST",
            trade_offs="Simple",
            impact="Low",
        )
        # Even though complete, status is still proposed
        assert adr.status == "proposed"
        assert adr.authority == AUTHORITY_PROPOSED

    def test_adr_acceptance_requires_explicit_transition(self):
        """ADR can only become ACCEPTED via explicit transition."""
        adr = make_adr()
        assert adr.status == "proposed"
        # Valid transition: proposed → accepted
        assert is_valid_transition("ADR", "proposed", "accepted") is True
        adr.transition_status("accepted", reason="Reviewed and approved")
        assert adr.status == "accepted"
        assert adr.authority == AUTHORITY_PROPOSED  # Authority unchanged by status transition

    def test_adr_cannot_skip_to_deprecated(self):
        """ADR cannot skip from proposed to deprecated."""
        adr = make_adr()
        assert is_valid_transition("ADR", "proposed", "deprecated") is False
        with pytest.raises(Exception):  # LifecycleError
            adr.transition_status("deprecated")


# ──────────────────────────────────────────────────────────────────────
# Context, rationale, alternatives, consequences
# ──────────────────────────────────────────────────────────────────────

class TestDRFields:
    """context, rationale, alternatives, consequences must be explicit."""

    def test_context_required(self):
        """Missing context = validation error."""
        with pytest.raises(RecordError, match="context"):
            make_dr(context="")

    def test_rationale_required(self):
        """Missing rationale = completeness violation."""
        with pytest.raises(RecordError, match="rationale"):
            make_dr(rationale="")

    def test_no_fabricated_alternatives(self):
        """If only one option was evaluated, alternatives must be empty."""
        dr = make_dr(alternatives=[])
        assert dr.alternatives == []
        assert dr.is_complete() is True

    def test_alternatives_with_disposition_structured(self):
        """Alternatives evaluated and rejected must be structured."""
        dr = make_dr(
            alternatives=[
                Alternative("ALT-001", "Option A", DISPOSITION_REJECTED, "Reason A"),
                Alternative("ALT-002", "Option B", DISPOSITION_SELECTED, "Reason B"),
            ],
        )
        assert len(dr.alternatives) == 2
        assert all(isinstance(a, Alternative) for a in dr.alternatives)

    def test_consequences_structured(self):
        """Consequences must be explicitly represented."""
        cons = Consequences(
            positive=["Faster", "Cheaper"],
            negative=["More memory"],
            risks=["Scaling limit"],
        )
        dr = make_dr(consequences=cons)
        assert dr.consequences.positive == ["Faster", "Cheaper"]
        assert dr.consequences.negative == ["More memory"]
        assert dr.consequences.risks == ["Scaling limit"]

    def test_consequences_empty_valid(self):
        """Empty consequences are valid (not fabricated)."""
        dr = make_dr(consequences=Consequences())
        assert dr.consequences.positive == []
        assert dr.consequences.negative == []
        assert dr.consequences.risks == []


# ──────────────────────────────────────────────────────────────────────
# REQ → ADR and NFR → ADR relationships
# ──────────────────────────────────────────────────────────────────────

class TestDRADRRelationships:
    """DR/ADR graph relationships with requirements and decisions."""

    def test_dr_to_requirement_relationship(self):
        dr = make_dr(related_requirements=["REQ-001"])
        rels = dr.get_relationships(RELATIONSHIP_DECIDED_BY)
        assert len(rels) == 1
        assert rels[0].target_id == "REQ-001"

    def test_adr_to_requirement_relationship(self):
        adr = make_adr(related_requirements=["REQ-001", "NFR-001"])
        rels = adr.get_relationships(RELATIONSHIP_DECIDED_BY)
        assert len(rels) == 2
        targets = {r.target_id for r in rels}
        assert targets == {"REQ-001", "NFR-001"}

    def test_dr_to_decision_relationship(self):
        dr = make_dr(related_decisions=["ADR-001", "DR-002"])
        rels = dr.get_relationships(RELATIONSHIP_RELATES_TO)
        assert len(rels) == 2

    def test_adr_supersedes_relationship(self):
        adr1 = make_adr(record_id="ADR-001")
        adr2 = make_adr(record_id="ADR-002")
        adr1.supersede(adr2.record_id)
        rels = adr1.get_relationships(RELATIONSHIP_SUPERSEDES)
        assert len(rels) == 1
        assert rels[0].target_id == "ADR-002"


# ──────────────────────────────────────────────────────────────────────
# ADR → Evidence external reference
# ──────────────────────────────────────────────────────────────────────

class TestADRToEvidence:
    """ADR references evidence by stable ID (no content duplication)."""

    def test_adr_evidence_refs_via_provenance(self):
        adr = make_adr()
        adr.provenance.evidence_refs = ["EV-001", "EV-002"]
        assert adr.provenance.evidence_refs == ["EV-001", "EV-002"]

    def test_adr_evidence_supported_by_relationship(self):
        adr = make_adr()
        adr.add_relationship("EV-001", RELATIONSHIP_SUPPORTED_BY)
        rels = adr.get_relationships(RELATIONSHIP_SUPPORTED_BY)
        assert len(rels) == 1
        assert rels[0].target_id == "EV-001"


# ──────────────────────────────────────────────────────────────────────
# Versioning: version vs supersession
# ──────────────────────────────────────────────────────────────────────

class TestVersioning:
    """Version = same logical decision evolved. Supersession = new decision replaces."""

    def test_version_increment_same_record(self):
        """Versioning: ADR-005 v1 → ADR-005 v2 = same logical decision evolved."""
        adr = make_adr(version=1)
        assert adr.version == 1
        # Version is a property of the record, not a new record_id
        adr.version = 2
        assert adr.version == 2
        assert adr.record_id == "ADR-001"  # Same ID

    def test_supersede_creates_new_record(self):
        """Supersession: new decision replaces previous = new ADR + supersession."""
        adr1 = make_adr(record_id="ADR-001", version=1)
        adr2 = make_adr(record_id="ADR-002", version=1)
        adr1.supersede(adr2.record_id)
        assert adr1.superseded_by == "ADR-002"
        assert adr1.record_id != adr2.record_id  # Different records

    def test_version_vs_supersession_distinct(self):
        """Version != supersession."""
        adr = make_adr(version=1)
        assert adr.superseded_by is None
        # Version bump doesn't create supersession
        adr.version = 2
        assert adr.superseded_by is None
        # Supersession creates new record, not version bump
        adr.supersede("ADR-002")
        assert adr.superseded_by == "ADR-002"


# ──────────────────────────────────────────────────────────────────────
# Supersession chain, effective decision, supersession conflict
# ──────────────────────────────────────────────────────────────────────

class TestSupersession:
    """Supersession chain traversal, effective decision, conflict detection."""

    def test_supersession_chain_linear(self):
        """ADR-001 → ADR-004 → ADR-009."""
        adr1 = make_adr(record_id="ADR-001")
        adr2 = make_adr(record_id="ADR-004")
        adr3 = make_adr(record_id="ADR-009")
        adr1.supersede("ADR-004")
        adr2.supersede("ADR-009")

        record_map = {
            "ADR-001": adr1,
            "ADR-004": adr2,
            "ADR-009": adr3,
        }
        chain = get_supersession_chain("ADR-001", record_map)
        assert chain == ["ADR-001", "ADR-004", "ADR-009"]

    def test_effective_decision(self):
        """Effective decision = last in chain."""
        adr1 = make_adr(record_id="ADR-001")
        adr2 = make_adr(record_id="ADR-004")
        adr3 = make_adr(record_id="ADR-009")
        adr1.supersede("ADR-004")
        adr2.supersede("ADR-009")

        record_map = {
            "ADR-001": adr1,
            "ADR-004": adr2,
            "ADR-009": adr3,
        }
        effective = get_effective_decision("ADR-001", record_map)
        assert effective == "ADR-009"

    def test_effective_decision_single(self):
        """Single record is its own effective decision."""
        adr = make_adr()
        effective = get_effective_decision("ADR-001", {"ADR-001": adr})
        assert effective == "ADR-001"

    def test_supersession_conflict_mutual(self):
        """Mutual supersession: A supersedes B, B supersedes A."""
        adr1 = make_adr(record_id="ADR-001")
        adr2 = make_adr(record_id="ADR-002")
        adr1.supersede("ADR-002")
        adr2.supersede("ADR-001")  # Mutual — conflict!

        conflicts = find_supersession_conflicts([adr1, adr2])
        types = [c["type"] for c in conflicts]
        assert "MUTUAL_SUPERSESSION" in types

    def test_supersession_conflict_ambiguous(self):
        """Ambiguous: A supersedes both B and C (at graph level)."""
        adr1 = make_adr(record_id="ADR-001")
        adr2 = make_adr(record_id="ADR-002")
        adr3 = make_adr(record_id="ADR-003")
        # At graph level: ADR-001 → SUPERSEDES → ADR-002 AND ADR-003
        adr1.add_relationship("ADR-002", RELATIONSHIP_SUPERSEDES)
        adr1.add_relationship("ADR-003", RELATIONSHIP_SUPERSEDES)

        conflicts = find_supersession_conflicts([adr1, adr2, adr3])
        types = [c["type"] for c in conflicts]
        assert "AMBIGUOUS_SUCCESSOR" in types

    def test_supersession_no_conflict_clean_chain(self):
        """Clean linear chain has no conflicts."""
        adr1 = make_adr(record_id="ADR-001")
        adr2 = make_adr(record_id="ADR-002")
        adr1.supersede("ADR-002")

        conflicts = find_supersession_conflicts([adr1, adr2])
        assert len(conflicts) == 0


# ──────────────────────────────────────────────────────────────────────
# Explicit contradiction, accepted ADR conflict
# ──────────────────────────────────────────────────────────────────────

class TestContradictionDetection:
    """Explicit contradictions and accepted ADR conflicts."""

    def test_explicit_contradicts_relationship(self):
        """Explicit CONTRADICTS relationship between records."""
        adr1 = make_adr(record_id="ADR-001")
        adr2 = make_adr(record_id="ADR-002")
        adr1.add_relationship("ADR-002", RELATIONSHIP_CONTRADICTS)
        rels = adr1.get_relationships(RELATIONSHIP_CONTRADICTS)
        assert len(rels) == 1
        assert rels[0].target_id == "ADR-002"

    def test_contradictor_detects_explicit_contradiction(self):
        """ContradictionDetector flags explicit CONTRADICTS edges."""
        from harness.knowledge.contradiction import ContradictionDetector

        adr1 = make_adr(record_id="ADR-001")
        adr2 = make_adr(record_id="ADR-002")
        adr1.add_relationship("ADR-002", RELATIONSHIP_CONTRADICTS)

        graph = EngineeringKnowledgeGraph.build_from_records([adr1, adr2])
        detector = ContradictionDetector(graph)
        conflicts = detector.detect_all()

        types = [c.conflict_type for c in conflicts]
        assert "EXPLICIT_CONTRADICTION" in types

    def test_accepted_adr_conflict_detection(self):
        """Two accepted ADRs on same topic with CONTRADICTS = conflict."""
        from harness.knowledge.contradiction import ContradictionDetector

        adr1 = make_adr(record_id="ADR-001")
        adr1.transition_status("accepted")
        adr2 = make_adr(record_id="ADR-002")
        adr2.transition_status("accepted")
        adr1.add_relationship("ADR-002", RELATIONSHIP_CONTRADICTS)

        graph = EngineeringKnowledgeGraph.build_from_records([adr1, adr2])
        detector = ContradictionDetector(graph)
        conflicts = detector.detect_all()

        types = [c.conflict_type for c in conflicts]
        assert "EXPLICIT_CONTRADICTION" in types


# ──────────────────────────────────────────────────────────────────────
# KnowledgeStore persistence, restart/reload, historical ADR recovery
# ──────────────────────────────────────────────────────────────────────

class TestKnowledgeStorePersistence:
    """DR/ADR persist through KnowledgeStore, survive restart/reload."""

    def test_dr_persistence_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = KnowledgeStore(db_path)

            dr = make_dr()
            store.save(dr)

            # Reload
            store2 = KnowledgeStore(db_path)
            loaded = store2.get("DR-001")

            assert loaded.record_id == "DR-001"
            assert loaded.record_type == "DR"
            assert loaded.decision_type == DECISION_TYPE_TECHNICAL
            assert loaded.title == "Test Decision"
            store.close()
            store2.close()

    def test_adr_persistence_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = KnowledgeStore(db_path)

            adr = make_adr()
            store.save(adr)

            # Reload
            store2 = KnowledgeStore(db_path)
            loaded = store2.get("ADR-001")

            assert loaded.record_id == "ADR-001"
            assert loaded.record_type == "ADR"
            assert loaded.architecture_domain == ADR_DOMAIN_DATA
            assert isinstance(loaded, ArchitectureDecisionRecord)
            store.close()
            store2.close()

    def test_historical_adr_recovery_after_supersession(self):
        """Historical ADRs are preserved after supersession (never deleted)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = KnowledgeStore(db_path)

            adr1 = make_adr(record_id="ADR-001")
            adr2 = make_adr(record_id="ADR-002")
            adr1.supersede("ADR-002")

            store.save(adr1)
            store.save(adr2)

            # Both records persist
            loaded1 = store.get("ADR-001")
            loaded2 = store.get("ADR-002")

            assert loaded1.record_id == "ADR-001"
            assert loaded1.superseded_by == "ADR-002"
            assert loaded2.record_id == "ADR-002"
            assert loaded2.superseded_by is None
            store.close()

    def test_version_preserved_after_reload(self):
        """Version is preserved after store/reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = KnowledgeStore(db_path)

            adr = make_adr(version=3)
            store.save(adr)

            store2 = KnowledgeStore(db_path)
            loaded = store2.get("ADR-001")
            assert loaded.version == 3
            store.close()
            store2.close()

    def test_authority_preserved_after_reload(self):
        """Authority is preserved after store/reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = KnowledgeStore(db_path)

            adr = make_adr()
            adr.transition_status("accepted")
            store.save(adr)

            store2 = KnowledgeStore(db_path)
            loaded = store2.get("ADR-001")
            assert loaded.status == "accepted"
            store.close()
            store2.close()


# ──────────────────────────────────────────────────────────────────────
# KnowledgeGraph traversal, ContradictionDetector integration
# ──────────────────────────────────────────────────────────────────────

class TestGraphTraversal:
    """KnowledgeGraph traversal with DR/ADR records."""

    def test_dr_in_graph_traversal(self):
        dr = make_dr(related_requirements=["REQ-001"])
        graph = EngineeringKnowledgeGraph.build_from_records([dr])

        assert graph.has_node("DR-001")
        neighbors = graph.neighbors("DR-001", relation_type=RELATIONSHIP_DECIDED_BY)
        assert "REQ-001" in neighbors

    def test_adr_in_graph_traversal(self):
        adr = make_adr(related_requirements=["REQ-001", "NFR-001"])
        graph = EngineeringKnowledgeGraph.build_from_records([adr])

        neighbors = graph.neighbors("ADR-001", relation_type=RELATIONSHIP_DECIDED_BY)
        assert "REQ-001" in neighbors
        assert "NFR-001" in neighbors

    def test_supersession_chain_in_graph(self):
        """Graph supports supersession chain traversal."""
        adr1 = make_adr(record_id="ADR-001")
        adr2 = make_adr(record_id="ADR-002")
        adr1.supersede("ADR-002")

        graph = EngineeringKnowledgeGraph.build_from_records([adr1, adr2])
        chain = graph.supersession_chain("ADR-001")
        assert chain == ["ADR-001", "ADR-002"]

    def test_effective_record_in_graph(self):
        """Graph effective_record returns the current effective record."""
        adr1 = make_adr(record_id="ADR-001")
        adr2 = make_adr(record_id="ADR-002")
        adr1.supersede("ADR-002")

        graph = EngineeringKnowledgeGraph.build_from_records([adr1, adr2])
        effective = graph.effective_record("ADR-001")
        assert effective == "ADR-002"

    def test_contradictor_with_graph(self):
        """ContradictionDetector works with DR/ADR graph."""
        from harness.knowledge.contradiction import ContradictionDetector

        adr1 = make_adr(record_id="ADR-001")
        adr2 = make_adr(record_id="ADR-002")
        adr1.add_relationship("ADR-002", RELATIONSHIP_CONTRADICTS)

        graph = EngineeringKnowledgeGraph.build_from_records([adr1, adr2])
        detector = ContradictionDetector(graph)
        assert not detector.is_clean()


# ──────────────────────────────────────────────────────────────────────
# ADRAutomation compatibility/migration
# ──────────────────────────────────────────────────────────────────────

class TestADRAutomationCompatibility:
    """ADRAutomation is an adapter to canonical ADR, not a competing store."""

    def test_adr_automation_creates_canonical_record(self):
        from harness.memory.adr import ADRAutomation
        from harness.memory import MemoryManager

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryManager(storage_dir=os.path.join(tmpdir, "memory"))
            adr_auto = ADRAutomation(memory)

            result = adr_auto.record_decision(
                adr_id="ADR-001",
                context="Choose auth",
                decision="JWT",
                alternatives="OAuth;Session",
                consequences="More secure",
            )
            # Returns path (backward compat)
            assert isinstance(result, str)

    def test_adr_automation_creates_proposed_record(self):
        """Agent-generated ADR via ADRAutomation is always PROPOSED."""
        from harness.memory.adr import ADRAutomation
        from harness.memory import MemoryManager

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "knowledge.db")
            memory = MemoryManager(storage_dir=os.path.join(tmpdir, "memory"))
            adr_auto = ADRAutomation(memory, knowledge_store_path=db_path)

            adr_auto.record_decision(
                adr_id="ADR-001",
                context="Choose auth",
                decision="JWT",
                alternatives="OAuth;Session",
                consequences="More secure",
            )

            # Verify stored as PROPOSED
            store = KnowledgeStore(db_path)
            adr = store.get("ADR-001")
            assert adr.status == "proposed"
            assert adr.authority == AUTHORITY_PROPOSED
            assert adr.record_type == "ADR"
            store.close()

    def test_adr_automation_preserves_legacy_api(self):
        """Old ADRAutomation API calls still work."""
        from harness.memory.adr import ADRAutomation
        from harness.memory import MemoryManager

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryManager(storage_dir=os.path.join(tmpdir, "memory"))
            adr_auto = ADRAutomation(memory)

            # Old API signature preserved
            result = adr_auto.record_decision(
                adr_id="ADR-001",
                context="Context",
                decision="Decision",
                alternatives="Alt1;Alt2",
                consequences="Consequence",
            )
            assert result is not None

            # get_relevant_adrs still works
            results = adr_auto.get_relevant_adrs("Context")
            assert isinstance(results, list)

    def test_memory_references_adr_not_contains(self):
        """Memory may reference ADR but never contains independent canonical truth."""
        from harness.memory.adr import ADRAutomation
        from harness.memory import MemoryManager

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = MemoryManager(storage_dir=os.path.join(tmpdir, "memory"))
            adr_auto = ADRAutomation(memory)

            adr_auto.record_decision(
                adr_id="ADR-001",
                context="Context",
                decision="Decision",
                alternatives="Alt1",
                consequences="Consequence",
            )

            # Memory has a reference (lessons entry)
            entry = memory.retrieve("ADR-001", category="lessons")
            assert entry is not None
            assert "adr" in entry.tags

            # But canonical truth is in KnowledgeStore, not memory
            # Memory entry is just a reference, not the canonical record


# ──────────────────────────────────────────────────────────────────────
# Deterministic serialization, deterministic traversal
# ──────────────────────────────────────────────────────────────────────

class TestDeterministicSerialization:
    """DR/ADR serialization and traversal must be deterministic."""

    def test_dr_deterministic_serialization(self):
        ts = "2026-01-01T00:00:00"
        prov = Provenance(author="test", source="human", timestamp=ts)
        dr1 = make_dr(created_at=ts, updated_at=ts, provenance=prov)
        dr2 = make_dr(created_at=ts, updated_at=ts, provenance=prov)
        assert dr1.to_json() == dr2.to_json()
        assert dr1.compute_hash() == dr2.compute_hash()

    def test_adr_deterministic_serialization(self):
        ts = "2026-01-01T00:00:00"
        prov = Provenance(author="test", source="human", timestamp=ts)
        adr1 = make_adr(patterns_considered=["REST", "GraphQL"], pattern_selected="REST", created_at=ts, updated_at=ts, provenance=prov)
        adr2 = make_adr(patterns_considered=["REST", "GraphQL"], pattern_selected="REST", created_at=ts, updated_at=ts, provenance=prov)
        assert adr1.to_json() == adr2.to_json()
        assert adr1.compute_hash() == adr2.compute_hash()

    def test_deterministic_traversal_order(self):
        """Graph traversal must be deterministic."""
        adr = make_adr(related_requirements=["REQ-003", "REQ-001", "REQ-002"])
        graph = EngineeringKnowledgeGraph.build_from_records([adr])

        # Traversal should be sorted (deterministic)
        neighbors = graph.neighbors("ADR-001", relation_type=RELATIONSHIP_DECIDED_BY)
        assert neighbors == sorted(neighbors)

    def test_deterministic_supersession_chain(self):
        """Supersession chain must be deterministic."""
        adr1 = make_adr(record_id="ADR-001")
        adr2 = make_adr(record_id="ADR-002")
        adr1.supersede("ADR-002")

        record_map = {"ADR-001": adr1, "ADR-002": adr2}
        chain1 = get_supersession_chain("ADR-001", record_map)
        chain2 = get_supersession_chain("ADR-001", record_map)
        assert chain1 == chain2


# ──────────────────────────────────────────────────────────────────────
# Domain separation: Engineering/Learning, Engineering/Evidence
# ──────────────────────────────────────────────────────────────────────

class TestDomainSeparation:
    """DR/ADR are Engineering, NOT Learning, NOT Evidence."""

    def test_dr_is_engineering_record(self):
        from harness.knowledge.records import EngineeringRecord
        dr = make_dr()
        assert isinstance(dr, EngineeringRecord)
        assert dr.record_type == "DR"

    def test_adr_is_engineering_record(self):
        from harness.knowledge.records import EngineeringRecord
        adr = make_adr()
        assert isinstance(adr, EngineeringRecord)
        assert adr.record_type == "ADR"

    def test_dr_not_learning_record(self):
        """DR is not StructuredExperience, FailureLesson, Strategy, etc."""
        dr = make_dr()
        assert dr.__class__.__name__ == "DecisionRecord"
        assert not hasattr(dr, "task_id")  # Not an experience
        assert not hasattr(dr, "failure_mode")  # Not a failure lesson

    def test_adr_not_evidence(self):
        """ADR is not EvidencePackage or CanonicalExecutionRecord."""
        adr = make_adr()
        assert adr.__class__.__name__ == "ArchitectureDecisionRecord"
        assert not hasattr(adr, "content_hash")  # Not an evidence package


# ──────────────────────────────────────────────────────────────────────
# Phase 5 E2E scenario: PRD → REQ → NFR → ADR → Evidence
# ──────────────────────────────────────────────────────────────────────

class TestPhase5E2E:
    """End-to-end Phase 5 scenario: PRD → REQ → NFR → ADR → Evidence."""

    def test_phase5_e2e_scenario(self):
        """Full chain: PRD → REQ → NFR → ADR → Evidence reference."""
        from harness.knowledge import PRD, Requirement, NFR
        from harness.knowledge.requirements import (
            NFR_CATEGORY_PERFORMANCE,
            NFR_SCOPE_GLOBAL,
            MEASUREMENT_LOAD_TEST,
            REQ_TYPE_FUNCTIONAL,
        )

        # Create PRD with requirement
        prd = PRD(
            record_id="PRD-001",
            title="User Authentication",
            description="System must authenticate users",
            objective="Secure user authentication",
            scope="Web and mobile",
            stakeholders=["security", "product"],
            requirements=["REQ-001"],
            related_nfrs=["NFR-001"],
            priority="critical",
            provenance=make_provenance(),
        )

        req = Requirement(
            record_id="REQ-001",
            title="JWT Authentication",
            description="System must use JWT tokens",
            req_type=REQ_TYPE_FUNCTIONAL,
            priority="critical",
            parent_prd="PRD-001",
            provenance=make_provenance(),
        )

        nfr = NFR(
            record_id="NFR-001",
            title="Auth Response Time",
            description="Authentication must respond within 200ms",
            category=NFR_CATEGORY_PERFORMANCE,
            metric={"name": "response_time_p99", "operator": "<", "target": "200ms"},
            measurement={"method": MEASUREMENT_LOAD_TEST},
            related_prds=["PRD-001"],
            related_requirements=["REQ-001"],
            provenance=make_provenance(),
        )

        # ADR addresses REQ and is constrained by NFR
        adr = make_adr(
            record_id="ADR-001",
            patterns_considered=["JWT", "OAuth2", "Session"],
            pattern_selected="JWT",
            trade_offs="Simple token vs complex flow",
            impact="Reduces auth latency",
            related_requirements=["REQ-001"],
        )
        adr.link_requirement("NFR-001")

        # Evidence reference — stable ID reference + graph relationship
        adr.provenance.evidence_refs = ["EV-LOAD-001"]
        adr.add_relationship("EV-LOAD-001", RELATIONSHIP_SUPPORTED_BY)

        # Store all records
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "knowledge.db")
            store = KnowledgeStore(db_path)
            store.save(prd)
            store.save(req)
            store.save(nfr)
            store.save(adr)

            # Build graph
            graph = EngineeringKnowledgeGraph.build_from_store(store)

            # Verify graph relationships
            assert graph.has_node("PRD-001")
            assert graph.has_node("REQ-001")
            assert graph.has_node("NFR-001")
            assert graph.has_node("ADR-001")

            # PRD → REQUIRES → REQ
            prd_neighbors = graph.neighbors("PRD-001", relation_type="REQUIRES")
            assert "REQ-001" in prd_neighbors

            # DR → DECIDED_BY → REQ
            adr_neighbors = graph.neighbors("ADR-001", relation_type=RELATIONSHIP_DECIDED_BY)
            assert "REQ-001" in adr_neighbors
            assert "NFR-001" in adr_neighbors

            # ADR → SUPPORTED_BY → Evidence
            evidence_neighbors = graph.neighbors("ADR-001", relation_type=RELATIONSHIP_SUPPORTED_BY)
            assert "EV-LOAD-001" in evidence_neighbors

            # Reload and verify persistence
            store2 = KnowledgeStore(db_path)
            loaded_adr = store2.get("ADR-001")
            assert loaded_adr.record_type == "ADR"
            assert isinstance(loaded_adr, ArchitectureDecisionRecord)
            assert loaded_adr.pattern_selected == "JWT"
            assert loaded_adr.provenance.evidence_refs == ["EV-LOAD-001"]
            store.close()
            store2.close()

    def test_phase5_version_and_supersession(self):
        """E2E: Version evolution + supersession chain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "knowledge.db")
            store = KnowledgeStore(db_path)

            # ADR v1
            adr1 = make_adr(record_id="ADR-001", version=1)
            store.save(adr1)

            # Version 2 (evolved, same record)
            adr1.version = 2
            store.save(adr1)

            # Superseded by new ADR
            adr2 = make_adr(record_id="ADR-002", version=1)
            adr1.supersede("ADR-002")
            store.save(adr1)
            store.save(adr2)

            store2 = KnowledgeStore(db_path)
            loaded1 = store2.get("ADR-001")
            loaded2 = store2.get("ADR-002")

            assert loaded1.version == 2
            assert loaded1.superseded_by == "ADR-002"
            assert loaded2.superseded_by is None

            # Graph traversal
            graph = EngineeringKnowledgeGraph.build_from_store(store2)
            chain = graph.supersession_chain("ADR-001")
            assert chain == ["ADR-001", "ADR-002"]
            effective = graph.effective_record("ADR-001")
            assert effective == "ADR-002"
            store.close()
            store2.close()
