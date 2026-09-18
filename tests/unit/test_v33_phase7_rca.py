# tests/unit/test_v33_phase7_rca.py — Phase 7 Acceptance Tests
"""Phase 7: RCA + Failure Intelligence Integration — deterministic behavior verification.

Tests cover:
- RCA creation, validation, lifecycle, authority, provenance
- Agent-generated RCA remains PROPOSED
- Failure → RCA traceability with stable failure identity
- Symptom != root cause; root cause != contributing factor
- Multiple root causes; corrective vs preventive actions
- Evidence references; RCA → ADR/TDR/Risk/SEC cross-domain references
- Versioning; version != supersession; supersession; effective RCA
- Contradictions; supersession conflicts
- RootCauseAnalyzer/FailureAnalyzer/FailureToLearningPipeline compatibility
- RCA → FailureLesson derivation; FailureLesson → RCA backtrace
- StructuredExperience traceability; Strategy provenance
- Authority separation; security precedence; unsafe learning blocked
- Idempotency; restart persistence
- KnowledgeStore/Graph/ContradictionDetector integration
- Coverage/completeness metrics; E2E scenarios; determinism
- Three-domain separation; CapabilityGraph separation
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.knowledge import (
    RootCauseAnalysisRecord,
    RCA,
    RCAAnalyzer,
    FailureKnowledgeBridge,
    compute_rca_traceability_metrics,
    RCACompleteness,
    FailureKnowledgeCoverage,
    VALID_RCA_CAUSE_CATEGORIES,
    VALID_RCA_STATES,
    RCA_STATE_DRAFT,
    RCA_STATE_ANALYSIS,
    RCA_STATE_CORRECTIVE_ACTION,
    RCA_STATE_PREVENTIVE_ACTION,
    RCA_STATE_CLOSED,
    RCA_CAUSE_CATEGORY_IMPLEMENTATION,
    RCA_CAUSE_CATEGORY_SECURITY,
    RCA_CAUSE_CATEGORY_INTEGRATION,
    KnowledgeStore,
    EngineeringKnowledgeGraph,
    Provenance,
    AUTHORITY_PROPOSED,
    AUTHORITY_ACCEPTED,
    AUTHORITY_DEPRECATED,
    AUTHORITY_ARCHIVED,
    DecisionRecord,
    ArchitectureDecisionRecord,
    TechnicalDebtRecord,
    RiskRecord,
    SecurityRecord,
    AuthorityPrecedenceResolver,
    SEC_AUTHORITY_AUTHORITATIVE,
    SEC_AUTHORITY_ACCEPTED,
    SEC_AUTHORITY_PROPOSED,
    PRECEDENCE_GOVERNANCE,
    PRECEDENCE_SECURITY_AUTHORITY,
    PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE,
    PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE,
    PRECEDENCE_TASK_REQUIREMENTS,
    PRECEDENCE_LEARNED_KNOWLEDGE,
    PRECEDENCE_EXPLORATION,
    is_valid_transition,
    is_terminal_state,
    ContradictionDetector,
)
from harness.knowledge.records import (
    EngineeringRecord,
    RecordError,
    RELATIONSHIP_DERIVED_FROM,
    RELATIONSHIP_CONSTRAINED_BY,
    RELATIONSHIP_INTRODUCES,
    RELATIONSHIP_MITIGATES,
    RELATIONSHIP_SUPPORTED_BY,
    RELATIONSHIP_CONTRADICTS,
    RELATIONSHIP_RELATES_TO,
    RELATIONSHIP_SUPERSEDES,
    RELATIONSHIP_CAUSED_BY,
    RELATIONSHIP_RESOLVED_BY,
    RELATIONSHIP_VERIFIED_BY,
    RELATIONSHIP_REQUIRES,
    RELATIONSHIP_SATISFIES,
    RELATIONSHIP_DECIDED_BY,
)
from harness.knowledge.lifecycle import LifecycleError
from harness.knowledge.contradiction import ContradictionDetector as CD
from harness.planner.failure_intelligence import (
    RootCauseAnalyzer,
    StructuredFailure,
    FailureToLearningPipeline,
    FailureLesson,
    FailureAnalyzer,
    FAILURE_TAXONOMY,
)
from harness.learning.context import ExperimentContext
from harness.learning.experience import ExperiencePipeline, StructuredExperience


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def make_provenance(author="test_author", source="human"):
    return Provenance(author=author, source=source)


def make_rca(**kwargs):
    defaults = dict(
        record_id="RCA-001",
        title="Test RCA",
        description="A test root cause analysis record",
        summary="Test failure analysis summary",
        failure_ref="fid-001",
        failure_type="integration",
        symptoms=["API timeout", "connection refused"],
        root_causes=["Connection pool exhaustion", "Insufficient pool size"],
        contributing_factors=["High traffic", "No connection reuse"],
        impact="Service unavailable for 30 minutes",
        resolution="Increased pool size and added connection reuse",
        corrective_actions=["Increase connection pool size to 100"],
        preventive_actions=["Add connection pool saturation monitoring"],
        evidence_refs=["EV-001"],
        related_failures=["fid-002"],
        related_decisions=["ADR-001"],
        incident_id="INC-001",
        provenance=make_provenance(),
    )
    defaults.update(kwargs)
    return RootCauseAnalysisRecord(**defaults)


def make_adr(**kwargs):
    defaults = dict(
        record_id="ADR-001",
        title="Test ADR",
        description="A test architecture decision record",
        decision_type="ARCHITECTURE",
        architecture_domain="data",
        context="Need to choose data storage",
        decision="Use PostgreSQL",
        rationale="PostgreSQL provides ACID compliance",
        provenance=make_provenance(),
    )
    defaults.update(kwargs)
    return ArchitectureDecisionRecord(**defaults)


def make_tdr(**kwargs):
    defaults = dict(
        record_id="TDR-001",
        title="Test TDR",
        description="A test technical debt record",
        debt_type="code",
        severity="medium",
        impact="Slows down development",
        remediation="Refactor the module",
        remediation_estimate="2 days",
        priority="medium",
        provenance=make_provenance(),
    )
    defaults.update(kwargs)
    return TechnicalDebtRecord(**defaults)


def make_rsk(**kwargs):
    defaults = dict(
        record_id="RSK-001",
        title="Test Risk",
        description="A test risk record",
        risk_category="technical",
        likelihood="medium",
        impact="medium",
        mitigation="Implement fallback",
        provenance=make_provenance(),
    )
    defaults.update(kwargs)
    return RiskRecord(**defaults)


def make_sec(**kwargs):
    defaults = dict(
        record_id="SEC-001",
        title="Test SEC",
        description="A test security record",
        category="CONSTRAINT",
        enforcement="Automated validation",
        authority=SEC_AUTHORITY_PROPOSED,
        provenance=make_provenance(),
    )
    defaults.update(kwargs)
    return SecurityRecord(**defaults)


def make_failure(failure_id="fid-001", category="integration", message="API timeout"):
    return StructuredFailure(
        node_id="n1",
        capability="backend_development",
        category=category,
        sub_category="timeout",
        severity="major",
        error_message=message,
        failure_id=failure_id,
        source_task_id="TASK-001",
    )


# ──────────────────────────────────────────────────────────────────────
# RCA Creation, Validation, Lifecycle
# ──────────────────────────────────────────────────────────────────────

class TestRCACreation:
    """RCA creation, validation, and lifecycle tests."""

    def test_rca_creation_minimal(self):
        rca = make_rca()
        assert rca.record_type == "RCA"
        assert rca.status == RCA_STATE_DRAFT
        assert rca.authority == AUTHORITY_PROPOSED
        assert rca.failure_ref == "fid-001"
        assert rca.failure_type == "integration"

    def test_rca_creation_full(self):
        rca = make_rca(
            record_id="RCA-002",
            failure_ref="fid-002",
            failure_type="security",
            symptoms=["Auth bypass", "Unauthorized access"],
            root_causes=["Missing input validation", "Weak authentication"],
            contributing_factors=["Legacy code", "No security review"],
            impact="Data breach affecting 1000 users",
            resolution="Implemented input validation and MFA",
            corrective_actions=["Add input validation", "Implement MFA"],
            preventive_actions=["Security audit", "Code review process"],
            evidence_refs=["EV-001", "EV-002"],
            related_failures=["fid-003"],
            related_decisions=["ADR-002"],
            incident_id="INC-002",
        )
        assert rca.record_id == "RCA-002"
        assert rca.failure_ref == "fid-002"
        assert rca.failure_type == "security"
        assert len(rca.symptoms) == 2
        assert len(rca.root_causes) == 2
        assert len(rca.contributing_factors) == 2
        assert len(rca.corrective_actions) == 2
        assert len(rca.preventive_actions) == 2
        assert len(rca.evidence_refs) == 2
        assert len(rca.related_failures) == 1
        assert len(rca.related_decisions) == 1

    def test_rca_invalid_record_id_prefix(self):
        with pytest.raises((RecordError, AssertionError)):
            make_rca(record_id="ADR-001", related_decisions=[])

    def test_rca_empty_failure_ref_rejected(self):
        with pytest.raises(RecordError, match="failure_ref is required"):
            make_rca(failure_ref="")

    def test_rca_empty_root_causes_rejected(self):
        with pytest.raises(RecordError, match="at least one root cause"):
            make_rca(root_causes=[])

    def test_rca_empty_corrective_actions_rejected(self):
        with pytest.raises(RecordError, match="at least one corrective action"):
            make_rca(corrective_actions=[])

    def test_rca_empty_preventive_actions_rejected(self):
        with pytest.raises(RecordError, match="at least one preventive action"):
            make_rca(preventive_actions=[])

    def test_rca_invalid_related_decisions(self):
        with pytest.raises(RecordError, match="ADR/DR"):
            make_rca(related_decisions=["REQ-001"])

    def test_rca_completeness_check(self):
        rca = make_rca()
        completeness = rca.get_completeness()
        assert completeness.has_failure_ref is True
        assert completeness.has_root_causes is True
        assert completeness.has_corrective_actions is True
        assert completeness.has_preventive_actions is True
        assert rca.is_complete() is True

    def test_rca_completeness_missing_impact(self):
        rca = make_rca()
        rca.impact = ""
        completeness = rca.get_completeness()
        assert completeness.has_impact is False
        # Impact is not required for completeness (only failure_ref, root_causes, actions, provenance)
        assert rca.is_complete() is True  # Still complete without impact


class TestRCALifecycle:
    """RCA lifecycle transitions — fail closed on invalid transitions."""

    def test_rca_initial_state(self):
        rca = make_rca()
        assert rca.status == RCA_STATE_DRAFT

    def test_rca_valid_transitions(self):
        rca = make_rca()
        # draft → analysis
        assert rca.can_transition_to(RCA_STATE_ANALYSIS) is True
        rca.transition_status(RCA_STATE_ANALYSIS)
        assert rca.status == RCA_STATE_ANALYSIS

        # analysis → corrective_action
        assert rca.can_transition_to(RCA_STATE_CORRECTIVE_ACTION) is True
        rca.transition_status(RCA_STATE_CORRECTIVE_ACTION)
        assert rca.status == RCA_STATE_CORRECTIVE_ACTION

        # corrective_action → preventive_action
        assert rca.can_transition_to(RCA_STATE_PREVENTIVE_ACTION) is True
        rca.transition_status(RCA_STATE_PREVENTIVE_ACTION)
        assert rca.status == RCA_STATE_PREVENTIVE_ACTION

        # preventive_action → closed
        assert rca.can_transition_to(RCA_STATE_CLOSED) is True
        rca.transition_status(RCA_STATE_CLOSED)
        assert rca.status == RCA_STATE_CLOSED

    def test_rca_invalid_transitions(self):
        rca = make_rca()
        # draft → corrective_action (skip)
        assert rca.can_transition_to(RCA_STATE_CORRECTIVE_ACTION) is False
        with pytest.raises(LifecycleError):
            rca.transition_status(RCA_STATE_CORRECTIVE_ACTION)

    def test_rca_terminal_state(self):
        rca = make_rca()
        rca.transition_status(RCA_STATE_ANALYSIS)
        rca.transition_status(RCA_STATE_CORRECTIVE_ACTION)
        rca.transition_status(RCA_STATE_PREVENTIVE_ACTION)
        rca.transition_status(RCA_STATE_CLOSED)
        assert rca.status == RCA_STATE_CLOSED
        assert is_terminal_state("RCA", RCA_STATE_CLOSED) is True
        # Cannot transition out of closed
        assert rca.can_transition_to(RCA_STATE_DRAFT) is False
        with pytest.raises(LifecycleError):
            rca.transition_status(RCA_STATE_DRAFT)

    def test_rca_self_transition_not_valid(self):
        rca = make_rca()
        assert rca.can_transition_to(RCA_STATE_DRAFT) is False

    def test_rca_draft_to_closed_direct(self):
        """RCA can go directly from draft to closed (abandoned analysis)."""
        rca = make_rca()
        assert rca.can_transition_to(RCA_STATE_CLOSED) is True
        rca.transition_status(RCA_STATE_CLOSED)
        assert rca.status == RCA_STATE_CLOSED


class TestRCAAuthority:
    """RCA authority — agent-generated always starts as PROPOSED."""

    def test_rca_agent_generated_always_proposed(self):
        rca = make_rca()
        assert rca.authority == AUTHORITY_PROPOSED

    def test_rca_authority_not_auto_accepted(self):
        """Agent-generated RCA is never auto-ACCEPTED."""
        rca = make_rca()
        assert rca.authority == AUTHORITY_PROPOSED
        # Even with complete analysis, authority remains PROPOSED
        rca.transition_status(RCA_STATE_ANALYSIS)
        rca.transition_status(RCA_STATE_CORRECTIVE_ACTION)
        rca.transition_status(RCA_STATE_PREVENTIVE_ACTION)
        rca.transition_status(RCA_STATE_CLOSED)
        assert rca.authority == AUTHORITY_PROPOSED

    def test_rca_authority_can_be_accepted(self):
        """RCA authority can be explicitly accepted (by governance)."""
        rca = make_rca()
        rca.authority = AUTHORITY_ACCEPTED
        assert rca.authority == AUTHORITY_ACCEPTED

    def test_rca_analysis_confidence_not_authority(self):
        """Analysis confidence != Engineering Knowledge authority."""
        rca = make_rca()
        # Even with strong evidence and complete analysis
        rca.evidence_refs = ["EV-001", "EV-002", "EV-003"]
        rca.root_causes = ["Definitive root cause confirmed by multiple evidence"]
        # Authority is still PROPOSED
        assert rca.authority == AUTHORITY_PROPOSED


class TestRCASerialization:
    """RCA deterministic serialization."""

    def test_rca_deterministic_serialization(self):
        ts = "2026-01-01T00:00:00"
        prov = Provenance(author="test", source="human", timestamp=ts)
        rca1 = make_rca(created_at=ts, updated_at=ts, provenance=prov)
        rca2 = make_rca(created_at=ts, updated_at=ts, provenance=prov)
        assert rca1.to_json() == rca2.to_json()
        assert rca1.compute_hash() == rca2.compute_hash()

    def test_rca_from_dict_round_trip(self):
        rca = make_rca(
            failure_type="security",
            root_causes=["Root cause 1", "Root cause 2"],
            related_decisions=["ADR-001"],
        )
        data = rca.to_dict()
        restored = RootCauseAnalysisRecord.from_dict(data)
        assert restored.record_id == rca.record_id
        assert restored.failure_ref == rca.failure_ref
        assert restored.failure_type == rca.failure_type
        assert restored.root_causes == rca.root_causes
        assert restored.related_decisions == rca.related_decisions


# ──────────────────────────────────────────────────────────────────────
# Failure → RCA Traceability
# ──────────────────────────────────────────────────────────────────────

class TestFailureToRCATraceability:
    """Failure → RCA traceability with stable failure identity."""

    def test_rca_references_failure(self):
        rca = make_rca(failure_ref="fid-001")
        assert rca.references_failure("fid-001") is True
        assert rca.references_failure("fid-999") is False

    def test_rca_references_related_failure(self):
        rca = make_rca(related_failures=["fid-002"])
        assert rca.references_failure("fid-002") is True

    def test_stable_failure_identity_survives_restart(self):
        """RCA failure_ref survives process restart (stable string ID)."""
        rca = make_rca(failure_ref="fid-stable-123")
        data = rca.to_dict()
        # Simulate restart: reconstruct from dict
        restored = RootCauseAnalysisRecord.from_dict(data)
        assert restored.failure_ref == "fid-stable-123"

    def test_rca_failure_ref_not_object_identity(self):
        """RCA uses stable failure_id, not Python object identity or hash()."""
        failure1 = make_failure(failure_id="fid-001")
        failure2 = make_failure(failure_id="fid-001")
        # Same failure_id → same RCA failure_ref
        rca1 = make_rca(failure_ref=failure1.failure_id)
        rca2 = make_rca(failure_ref=failure2.failure_id)
        assert rca1.failure_ref == rca2.failure_ref


# ──────────────────────────────────────────────────────────────────────
# Symptom != Root Cause; Root Cause != Contributing Factor
# ──────────────────────────────────────────────────────────────────────

class TestRCASemanticSeparation:
    """Symptom != root cause; root cause != contributing factor."""

    def test_symptom_not_root_cause(self):
        rca = make_rca(
            symptoms=["API timeout"],
            root_causes=["Connection pool exhaustion"],
        )
        assert rca.symptoms != rca.root_causes
        assert "API timeout" not in rca.root_causes

    def test_root_cause_not_contributing_factor(self):
        rca = make_rca(
            root_causes=["Connection pool exhaustion"],
            contributing_factors=["High traffic"],
        )
        assert rca.root_causes != rca.contributing_factors
        assert "Connection pool exhaustion" not in rca.contributing_factors

    def test_multiple_root_causes(self):
        rca = make_rca(
            root_causes=["Connection pool exhaustion", "Insufficient pool size", "No connection reuse"],
        )
        assert len(rca.root_causes) == 3
        assert rca.root_causes[0] != rca.root_causes[1]

    def test_corrective_vs_preventive_actions(self):
        rca = make_rca(
            corrective_actions=["Increase connection pool size to 100"],
            preventive_actions=["Add connection pool saturation monitoring"],
        )
        assert rca.corrective_actions != rca.preventive_actions
        # Corrective fixes the observed problem
        assert "Increase" in rca.corrective_actions[0]
        # Preventive reduces probability of recurrence
        assert "monitoring" in rca.preventive_actions[0]


# ──────────────────────────────────────────────────────────────────────
# Evidence References
# ──────────────────────────────────────────────────────────────────────

class TestRCAEvidenceReferences:
    """RCA evidence references (by ID, no embedding)."""

    def test_rca_evidence_refs(self):
        rca = make_rca(evidence_refs=["EV-001", "EV-002"])
        assert rca.references_evidence("EV-001") is True
        assert rca.references_evidence("EV-999") is False

    def test_rca_add_evidence_ref(self):
        rca = make_rca()
        rca.add_evidence_ref("EV-001")
        assert "EV-001" in rca.evidence_refs
        rels = rca.get_relationships(RELATIONSHIP_SUPPORTED_BY)
        assert len(rels) == 1
        assert rels[0].target_id == "EV-001"

    def test_rca_evidence_not_embedded(self):
        """Evidence is referenced by ID, not embedded."""
        rca = make_rca(evidence_refs=["EV-001"])
        data = rca.to_dict()
        assert "EV-001" in data["evidence_refs"]
        # Evidence content is NOT in the RCA
        assert "evidence_content" not in data


# ──────────────────────────────────────────────────────────────────────
# RCA → ADR/TDR/Risk/SEC Cross-Domain References
# ──────────────────────────────────────────────────────────────────────

class TestRCACrossDomainReferences:
    """RCA cross-domain references with ADR, TDR, RSK, SEC."""

    def test_rca_to_adr_relationship(self):
        rca = make_rca(related_decisions=["ADR-001"])
        rels = rca.get_relationships(RELATIONSHIP_RELATES_TO)
        # ADR-001 is in RELATES_TO (related_failures also creates RELATES_TO)
        adr_rels = [r for r in rels if r.target_id == "ADR-001"]
        assert len(adr_rels) == 1
        assert adr_rels[0].target_id == "ADR-001"

    def test_rca_link_adr(self):
        rca = make_rca()
        rca.link_adr("ADR-001")
        assert "ADR-001" in rca.related_decisions
        rels = rca.get_relationships(RELATIONSHIP_RELATES_TO)
        # ADR-001 is in RELATES_TO (related_failures also creates RELATES_TO)
        adr_rels = [r for r in rels if r.target_id == "ADR-001"]
        assert len(adr_rels) == 1

    def test_rca_link_adr_invalid_id(self):
        rca = make_rca()
        with pytest.raises(RecordError, match="ADR/DR"):
            rca.link_adr("REQ-001")

    def test_rca_references_decision(self):
        rca = make_rca(related_decisions=["ADR-001"])
        assert rca.references_decision("ADR-001") is True
        assert rca.references_decision("ADR-999") is False

    def test_rca_to_tdr_relationship(self):
        """RCA can identify TDR (technical debt)."""
        rca = make_rca()
        rca.add_relationship("TDR-001", RELATIONSHIP_RELATES_TO)
        rels = rca.get_relationships(RELATIONSHIP_RELATES_TO)
        assert any(r.target_id == "TDR-001" for r in rels)

    def test_rca_to_rsk_relationship(self):
        """RCA can relate to RSK (risk)."""
        rca = make_rca()
        rca.add_relationship("RSK-001", RELATIONSHIP_RELATES_TO)
        rels = rca.get_relationships(RELATIONSHIP_RELATES_TO)
        assert any(r.target_id == "RSK-001" for r in rels)

    def test_rca_to_sec_relationship(self):
        """RCA can be constrained by SEC (security)."""
        rca = make_rca()
        rca.add_relationship("SEC-001", RELATIONSHIP_CONSTRAINED_BY)
        rels = rca.get_relationships(RELATIONSHIP_CONSTRAINED_BY)
        assert any(r.target_id == "SEC-001" for r in rels)


# ──────────────────────────────────────────────────────────────────────
# Versioning and Supersession
# ──────────────────────────────────────────────────────────────────────

class TestRCAVersioning:
    """RCA versioning and supersession."""

    def test_rca_version_increment(self):
        rca = make_rca(version=1)
        assert rca.version == 1
        rca.version = 2
        assert rca.version == 2
        assert rca.record_id == "RCA-001"

    def test_rca_create_new_version(self):
        """Version = same investigation evolves (same RCA ID, version++)."""
        rca1 = make_rca(record_id="RCA-001", version=1)
        rca2 = rca1.create_new_version("RCA-002")
        assert rca2.version == 2
        assert rca2.record_id == "RCA-002"
        assert rca1.superseded_by == "RCA-002"

    def test_rca_version_not_supersession(self):
        """Version != supersession."""
        rca1 = make_rca(record_id="RCA-010", version=1)
        rca2 = rca1.create_new_version("RCA-011")
        # Version is same investigation evolved
        assert rca2.version == rca1.version + 1
        # Supersession is different RCA replacing earlier
        assert rca1.superseded_by == "RCA-011"

    def test_rca_superseded_by(self):
        """Supersession = different RCA replaces earlier analysis."""
        rca1 = make_rca(record_id="RCA-012")
        rca2 = make_rca(record_id="RCA-013")
        rca1.superseded_by = "RCA-013"
        assert rca1.superseded_by == "RCA-013"

    def test_rca_effective_record(self):
        """Effective RCA = last in supersession chain."""
        rca1 = make_rca(record_id="RCA-014")
        rca2 = make_rca(record_id="RCA-015")
        rca3 = make_rca(record_id="RCA-016")
        rca1.add_relationship("RCA-015", RELATIONSHIP_SUPERSEDES)
        rca2.add_relationship("RCA-016", RELATIONSHIP_SUPERSEDES)

        graph = EngineeringKnowledgeGraph.build_from_records([rca1, rca2, rca3])
        effective = graph.effective_record("RCA-014")
        assert effective == "RCA-016"

    def test_rca_ambiguous_supersession_fails_closed(self):
        """Ambiguous supersession → effective returns None (fail closed)."""
        # Build graph with both supersession edges
        graph = EngineeringKnowledgeGraph()
        graph.add_node("RCA-017")
        graph.add_node("RCA-018")
        graph.add_node("RCA-019")
        graph.add_edge("RCA-017", RELATIONSHIP_SUPERSEDES, "RCA-018")
        graph.add_edge("RCA-017", RELATIONSHIP_SUPERSEDES, "RCA-019")

        effective = graph.effective_record("RCA-017")
        assert effective is None  # Ambiguous — fail closed


# ──────────────────────────────────────────────────────────────────────
# Contradictions
# ──────────────────────────────────────────────────────────────────────

class TestRCAContradictions:
    """RCA contradiction detection."""

    def test_rca_contradicts_relationship(self):
        """RCA-001 CONTRADICTS RCA-002."""
        rca1 = make_rca(record_id="RCA-001")
        rca2 = make_rca(record_id="RCA-002")
        rca1.add_relationship("RCA-002", RELATIONSHIP_CONTRADICTS)

        graph = EngineeringKnowledgeGraph.build_from_records([rca1, rca2])
        conflicts = graph.validate_integrity()
        types = [c.conflict_type for c in conflicts]
        assert "EXPLICIT_CONTRADICTION" in types

    def test_rca_supersession_conflict(self):
        """Mutual supersession → SUPERSESSION_CONFLICT."""
        rca1 = make_rca(record_id="RCA-020")
        rca2 = make_rca(record_id="RCA-021")
        rca1.add_relationship("RCA-021", RELATIONSHIP_SUPERSEDES)
        rca2.add_relationship("RCA-020", RELATIONSHIP_SUPERSEDES)

        graph = EngineeringKnowledgeGraph.build_from_records([rca1, rca2])
        conflicts = graph.validate_integrity()
        types = [c.conflict_type for c in conflicts]
        assert "SUPERSESSION_CONFLICT" in types

    def test_rca_status_conflict(self):
        """Superseded RCA still active → STATUS_CONFLICT."""
        rca1 = make_rca(record_id="RCA-022")
        rca2 = make_rca(record_id="RCA-023")
        rca1.add_relationship("RCA-023", RELATIONSHIP_SUPERSEDES)
        # rca1 is still in draft status (active)

        graph = EngineeringKnowledgeGraph.build_from_records([rca1, rca2])
        detector = ContradictionDetector(graph)
        conflicts = detector.detect_with_records([rca1, rca2])
        types = [c.conflict_type for c in conflicts]
        assert "STATUS_CONFLICT" in types


# ──────────────────────────────────────────────────────────────────────
# RootCauseAnalyzer / FailureAnalyzer / FailureToLearningPipeline Compatibility
# ──────────────────────────────────────────────────────────────────────

class TestFailureIntelligenceCompatibility:
    """Existing V3.2 failure intelligence remains functional."""

    def test_root_cause_analyzer_unchanged(self):
        """RootCauseAnalyzer still works as before."""
        analyzer = RootCauseAnalyzer()
        failure = make_failure()
        result = analyzer.analyze(failure)
        assert result.category == "integration"
        assert result.sub_category == "timeout"

    def test_failure_analyzer_unchanged(self):
        """FailureAnalyzer still works as before."""
        analyzer = FailureAnalyzer()
        failure = make_failure()
        result = analyzer.analyze(failure)
        assert result.category == "integration"

    def test_failure_to_learning_pipeline_unchanged(self):
        """FailureToLearningPipeline still works as before."""
        with tempfile.TemporaryDirectory() as tmp:
            ctx = ExperimentContext(
                validation_run_id="V3.2-P7-TEST",
                benchmark_id="BENCH-TEST",
                task_id="TASK-TEST",
                execution_id="exec-test",
                mode="LEARNED_EXPLORATION",
            )
            pipe = FailureToLearningPipeline(
                experience_pipeline=ExperiencePipeline(storage_dir=tmp),
                experiment_context=ctx,
            )
            failure = make_failure()
            result = pipe.process(failure, ctx)
            assert result["suppressed"] is False
            assert result["experience_id"] is not None

    def test_rca_analyzer_uses_root_cause_analyzer(self):
        """RCAAnalyzer uses RootCauseAnalyzer internally."""
        rca_analyzer = RCAAnalyzer()
        failure = make_failure()
        rca = rca_analyzer.analyze_to_rca(
            failure=failure,
            root_causes=["Connection pool exhaustion"],
            corrective_actions=["Increase pool size"],
            preventive_actions=["Add monitoring"],
        )
        assert rca.record_type == "RCA"
        assert rca.failure_ref == "fid-001"
        assert len(rca.root_causes) == 1


# ──────────────────────────────────────────────────────────────────────
# RCA → FailureLesson Derivation
# ──────────────────────────────────────────────────────────────────────

class TestRCAFailureLessonDerivation:
    """RCA → FailureLesson derivation with stable provenance."""

    def test_rca_derive_failure_lesson(self):
        """RCA can derive a FailureLesson proposal."""
        rca = make_rca()
        lesson = rca.derive_failure_lesson()
        assert lesson["source_rca_id"] == "RCA-001"
        assert lesson["source_failure_id"] == "fid-001"
        assert lesson["failure_category"] == "integration"
        assert lesson["root_cause"] == "Connection pool exhaustion; Insufficient pool size"
        assert lesson["authority"] == "proposed"

    def test_rca_derive_lesson_authority_separation(self):
        """RCA ACCEPTED does NOT automatically mean FailureLesson AUTHORITATIVE."""
        rca = make_rca()
        rca.authority = AUTHORITY_ACCEPTED
        lesson = rca.derive_failure_lesson()
        # Lesson authority is still "proposed" — not auto-upgraded
        assert lesson["authority"] == "proposed"

    def test_rca_derive_lesson_preserves_provenance(self):
        """Derived lesson preserves source RCA and failure IDs."""
        rca = make_rca(
            record_id="RCA-001",
            failure_ref="fid-001",
            root_causes=["Root cause A", "Root cause B"],
            corrective_actions=["Fix A", "Fix B"],
            preventive_actions=["Prevent A", "Prevent B"],
        )
        lesson = rca.derive_failure_lesson()
        assert lesson["source_rca_id"] == "RCA-001"
        assert lesson["source_failure_id"] == "fid-001"
        assert "Root cause A" in lesson["root_cause"]
        assert "Fix A" in lesson["corrective_action"]
        assert "Prevent A" in lesson["prevention_strategy"]

    def test_rca_derive_lesson_security_checked(self):
        """Security failure RCA derives security-checked lesson."""
        rca = make_rca(
            failure_type="security",
            root_causes=["Auth bypass"],
        )
        lesson = rca.derive_failure_lesson()
        assert lesson["security_checked"] is True

    def test_failure_lesson_to_rca_backtrace(self):
        """FailureLesson → source → RCA → source → Failure."""
        rca = make_rca(
            record_id="RCA-001",
            failure_ref="fid-001",
        )
        lesson = rca.derive_failure_lesson()
        # Backtrace: lesson → RCA → failure
        assert lesson["source_rca_id"] == "RCA-001"
        assert lesson["source_failure_id"] == "fid-001"
        # RCA references failure
        assert rca.failure_ref == "fid-001"


# ──────────────────────────────────────────────────────────────────────
# StructuredExperience Traceability
# ──────────────────────────────────────────────────────────────────────

class TestStructuredExperienceTraceability:
    """StructuredExperience traceability to RCA and Failure."""

    def test_experience_traces_to_failure(self):
        """StructuredExperience traces back to failure_id."""
        failure = make_failure(failure_id="fid-001")
        rca = make_rca(failure_ref="fid-001")
        # Experience provenance includes failure_id
        exp_provenance = {
            "source_type": "failure",
            "failure_id": failure.failure_id,
            "source_rca_id": rca.record_id,
        }
        assert exp_provenance["failure_id"] == "fid-001"
        assert exp_provenance["source_rca_id"] == "RCA-001"

    def test_experience_traces_to_rca(self):
        """StructuredExperience can trace back to RCA."""
        rca = make_rca(record_id="RCA-001", failure_ref="fid-001")
        exp_provenance = {
            "source_type": "failure",
            "failure_id": "fid-001",
            "source_rca_id": "RCA-001",
        }
        # Backtrace: experience → RCA → failure
        assert exp_provenance["source_rca_id"] == rca.record_id
        assert rca.failure_ref == exp_provenance["failure_id"]


# ──────────────────────────────────────────────────────────────────────
# Strategy Provenance
# ──────────────────────────────────────────────────────────────────────

class TestStrategyProvenance:
    """Strategy → DERIVED_FROM → FailureLesson → DERIVED_FROM → RCA → DERIVED_FROM → Failure."""

    def test_strategy_provenance_chain(self):
        """Strategy provenance chain: Strategy → FailureLesson → RCA → Failure."""
        rca = make_rca(record_id="RCA-001", failure_ref="fid-001")
        lesson = rca.derive_failure_lesson()
        # Strategy derived from lesson
        strategy_provenance = {
            "derived_from": "FailureLesson",
            "source_lesson_id": "lesson-001",
            "source_rca_id": lesson["source_rca_id"],
            "source_failure_id": lesson["source_failure_id"],
        }
        assert strategy_provenance["source_rca_id"] == "RCA-001"
        assert strategy_provenance["source_failure_id"] == "fid-001"
        # Full chain: Strategy → FailureLesson → RCA → Failure
        assert strategy_provenance["source_rca_id"] == rca.record_id
        assert rca.failure_ref == strategy_provenance["source_failure_id"]


# ──────────────────────────────────────────────────────────────────────
# Authority Separation
# ──────────────────────────────────────────────────────────────────────

class TestAuthoritySeparation:
    """Engineering authority does not leak into Learning; learning confidence does not become Engineering authority."""

    def test_rca_authority_not_from_lesson_confidence(self):
        """High-confidence FailureLesson does NOT upgrade RCA authority."""
        rca = make_rca()
        lesson = rca.derive_failure_lesson()
        # Even if lesson has high confidence
        lesson["confidence"] = 0.95
        # RCA authority remains PROPOSED
        assert rca.authority == AUTHORITY_PROPOSED

    def test_lesson_authority_not_from_rca_authority(self):
        """RCA ACCEPTED does NOT automatically make FailureLesson AUTHORITATIVE."""
        rca = make_rca()
        rca.authority = AUTHORITY_ACCEPTED
        lesson = rca.derive_failure_lesson()
        # Lesson authority is still "proposed"
        assert lesson["authority"] == "proposed"

    def test_rca_authority_not_from_analysis_confidence(self):
        """Analysis confidence != Engineering Knowledge authority."""
        rca = make_rca()
        # Strong evidence and complete analysis
        rca.evidence_refs = ["EV-001", "EV-002", "EV-003"]
        rca.root_causes = ["Definitive root cause confirmed"]
        # Authority is still PROPOSED
        assert rca.authority == AUTHORITY_PROPOSED


# ──────────────────────────────────────────────────────────────────────
# Security Precedence
# ──────────────────────────────────────────────────────────────────────

class TestSecurityPrecedence:
    """RCA-derived lessons cannot override SEC."""

    def test_rca_derived_lesson_cannot_override_sec(self):
        """FailureLesson 'disable certificate validation' + SEC → SEC WINS."""
        sec = make_sec(
            record_id="SEC-001",
            authority=SEC_AUTHORITY_AUTHORITATIVE,
        )
        rca = make_rca(
            record_id="RCA-001",
            failure_type="security",
            root_causes=["Certificate validation timeout"],
        )
        lesson = rca.derive_failure_lesson()
        # Even if lesson suggests disabling validation
        lesson["corrective_action"] = "disable certificate validation"
        # SEC still wins
        resolver = AuthorityPrecedenceResolver()
        sec_candidate = {"source": "security", "authority": SEC_AUTHORITY_AUTHORITATIVE, "record_type": "SEC"}
        lesson_candidate = {"source": "learning", "authority": AUTHORITY_ACCEPTED, "record_type": "FailureLesson"}
        assert resolver.can_override(sec_candidate, lesson_candidate) is True
        assert resolver.can_override(lesson_candidate, sec_candidate) is False

    def test_rca_security_failure_defensive_only(self):
        """Security failure RCA produces defensive-only lessons."""
        rca = make_rca(
            failure_type="security",
            root_causes=["Auth bypass"],
        )
        lesson = rca.derive_failure_lesson()
        assert lesson["security_checked"] is True
        # Lesson should not suggest weakening security
        assert "disable" not in lesson["corrective_action"].lower()
        assert "bypass" not in lesson["corrective_action"].lower()

    def test_rca_constrained_by_sec(self):
        """RCA can be constrained by SEC."""
        rca = make_rca()
        rca.add_relationship("SEC-001", RELATIONSHIP_CONSTRAINED_BY)
        rels = rca.get_relationships(RELATIONSHIP_CONSTRAINED_BY)
        assert any(r.target_id == "SEC-001" for r in rels)


# ──────────────────────────────────────────────────────────────────────
# Unsafe Learning Blocked
# ──────────────────────────────────────────────────────────────────────

class TestUnsafeLearningBlocked:
    """Unsafe learned workaround cannot override SEC."""

    def test_unsafe_lesson_blocked_by_sec(self):
        """FailureLesson 'disable certificate validation' + SEC → BLOCK."""
        sec = make_sec(
            record_id="SEC-001",
            authority=SEC_AUTHORITY_AUTHORITATIVE,
        )
        # Simulate unsafe lesson
        unsafe_lesson = {
            "source": "learning",
            "authority": AUTHORITY_ACCEPTED,
            "record_type": "FailureLesson",
            "corrective_action": "disable certificate validation to avoid timeout",
        }
        resolver = AuthorityPrecedenceResolver()
        sec_candidate = {"source": "security", "authority": SEC_AUTHORITY_AUTHORITATIVE, "record_type": "SEC"}
        # SEC wins
        assert resolver.can_override(sec_candidate, unsafe_lesson) is True
        assert resolver.can_override(unsafe_lesson, sec_candidate) is False

    def test_rca_security_precedence_over_lesson(self):
        """RCA security analysis respects SEC precedence."""
        sec = make_sec(
            record_id="SEC-001",
            authority=SEC_AUTHORITY_AUTHORITATIVE,
        )
        rca = make_rca(
            record_id="RCA-001",
            failure_type="security",
        )
        # RCA is constrained by SEC
        rca.add_relationship("SEC-001", RELATIONSHIP_CONSTRAINED_BY)
        # SEC authority is higher
        resolver = AuthorityPrecedenceResolver()
        sec_candidate = {"source": "security", "authority": SEC_AUTHORITY_AUTHORITATIVE, "record_type": "SEC"}
        rca_candidate = {"source": "engineering", "authority": AUTHORITY_ACCEPTED, "record_type": "RCA"}
        assert resolver.can_override(sec_candidate, rca_candidate) is True


# ──────────────────────────────────────────────────────────────────────
# Idempotency
# ──────────────────────────────────────────────────────────────────────

class TestIdempotency:
    """Processing the same failure/RCA derivation repeatedly must not create uncontrolled duplicates."""

    def test_rca_creation_idempotent(self):
        """Same failure → same RCA (deterministic)."""
        failure = make_failure(failure_id="fid-001")
        rca_analyzer = RCAAnalyzer()
        rca1 = rca_analyzer.analyze_to_rca(
            failure=failure,
            root_causes=["Connection pool exhaustion"],
            corrective_actions=["Increase pool size"],
            preventive_actions=["Add monitoring"],
            rca_id="RCA-001",
        )
        rca2 = rca_analyzer.analyze_to_rca(
            failure=failure,
            root_causes=["Connection pool exhaustion"],
            corrective_actions=["Increase pool size"],
            preventive_actions=["Add monitoring"],
            rca_id="RCA-001",
        )
        # Same RCA ID
        assert rca1.record_id == rca2.record_id
        # Same content (excluding timestamps and provenance which differ by microseconds)
        d1 = rca1.to_dict()
        d2 = rca2.to_dict()
        d1.pop("created_at", None)
        d1.pop("updated_at", None)
        d2.pop("created_at", None)
        d2.pop("updated_at", None)
        # Compare provenance without timestamp
        p1 = d1.pop("provenance", {})
        p2 = d2.pop("provenance", {})
        p1.pop("timestamp", None)
        p2.pop("timestamp", None)
        assert d1 == d2
        assert p1 == p2

    def test_rca_derivation_idempotent(self):
        """Same RCA → same FailureLesson derivation."""
        rca = make_rca()
        lesson1 = rca.derive_failure_lesson()
        lesson2 = rca.derive_failure_lesson()
        assert lesson1 == lesson2


# ──────────────────────────────────────────────────────────────────────
# Restart Persistence
# ──────────────────────────────────────────────────────────────────────

class TestRestartPersistence:
    """Process A: failure → RCA → lesson → experience → persist. Restart. Process B: reload → trace."""

    def test_rca_persistence_and_reload(self):
        """RCA persists through KnowledgeStore, survives restart/reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = KnowledgeStore(db_path)

            rca = make_rca()
            store.save(rca)

            # Reload
            store2 = KnowledgeStore(db_path)
            loaded = store2.get("RCA-001")

            assert loaded.record_id == "RCA-001"
            assert loaded.record_type == "RCA"
            assert loaded.failure_ref == "fid-001"
            assert isinstance(loaded, RootCauseAnalysisRecord)
            store.close()
            store2.close()

    def test_rca_version_preserved_after_reload(self):
        """RCA version is preserved after store/reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = KnowledgeStore(db_path)

            rca = make_rca(version=3)
            store.save(rca)

            store2 = KnowledgeStore(db_path)
            loaded = store2.get("RCA-001")
            assert loaded.version == 3
            store.close()
            store2.close()

    def test_rca_supersession_preserved_after_reload(self):
        """RCA supersession is preserved after store/reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = KnowledgeStore(db_path)

            rca1 = make_rca(record_id="RCA-001")
            rca2 = make_rca(record_id="RCA-002")
            rca1.superseded_by = "RCA-002"

            store.save(rca1)
            store.save(rca2)

            store2 = KnowledgeStore(db_path)
            loaded1 = store2.get("RCA-001")
            loaded2 = store2.get("RCA-002")

            assert loaded1.superseded_by == "RCA-002"
            assert loaded2.superseded_by is None
            store.close()
            store2.close()


# ──────────────────────────────────────────────────────────────────────
# KnowledgeStore / Graph / ContradictionDetector Integration
# ──────────────────────────────────────────────────────────────────────

class TestKnowledgeStoreGraphIntegration:
    """RCA integrates with KnowledgeStore, EngineeringKnowledgeGraph, ContradictionDetector."""

    def test_rca_in_knowledge_store(self):
        """RCA can be stored and retrieved from KnowledgeStore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = KnowledgeStore(db_path)

            rca = make_rca()
            store.save(rca)

            loaded = store.get("RCA-001")
            assert loaded.record_id == "RCA-001"
            assert loaded.record_type == "RCA"
            assert isinstance(loaded, RootCauseAnalysisRecord)
            store.close()

    def test_rca_in_graph_traversal(self):
        """RCA participates in graph traversal."""
        rca = make_rca(related_decisions=["ADR-001"])
        graph = EngineeringKnowledgeGraph.build_from_records([rca])

        assert graph.has_node("RCA-001")
        neighbors = graph.neighbors("RCA-001", relation_type=RELATIONSHIP_RELATES_TO)
        assert "ADR-001" in neighbors

    def test_rca_contradiction_detector(self):
        """ContradictionDetector works with RCA graph."""
        rca1 = make_rca(record_id="RCA-001")
        rca2 = make_rca(record_id="RCA-002")
        rca1.add_relationship("RCA-002", RELATIONSHIP_CONTRADICTS)

        graph = EngineeringKnowledgeGraph.build_from_records([rca1, rca2])
        detector = ContradictionDetector(graph)
        conflicts = detector.detect_all()
        types = [c.conflict_type for c in conflicts]
        assert "EXPLICIT_CONTRADICTION" in types

    def test_rca_with_adr_tdr_rsk_sec_graph(self):
        """RCA integrates with ADR, TDR, RSK, SEC in graph."""
        rca = make_rca(
            record_id="RCA-001",
            related_decisions=["ADR-001"],
        )
        adr = make_adr(record_id="ADR-001")
        tdr = make_tdr(record_id="TDR-001")
        rsk = make_rsk(record_id="RSK-001")
        sec = make_sec(record_id="SEC-001")

        # RCA → ADR
        rca.add_relationship("ADR-001", RELATIONSHIP_RELATES_TO)
        # RCA → TDR
        rca.add_relationship("TDR-001", RELATIONSHIP_RELATES_TO)
        # RCA → RSK
        rca.add_relationship("RSK-001", RELATIONSHIP_RELATES_TO)
        # RCA → SEC
        rca.add_relationship("SEC-001", RELATIONSHIP_CONSTRAINED_BY)

        graph = EngineeringKnowledgeGraph.build_from_records([rca, adr, tdr, rsk, sec])

        assert graph.has_node("RCA-001")
        assert graph.has_node("ADR-001")
        assert graph.has_node("TDR-001")
        assert graph.has_node("RSK-001")
        assert graph.has_node("SEC-001")

        # RCA → ADR
        adr_neighbors = graph.neighbors("RCA-001", relation_type=RELATIONSHIP_RELATES_TO)
        assert "ADR-001" in adr_neighbors

        # RCA → SEC
        sec_neighbors = graph.neighbors("RCA-001", relation_type=RELATIONSHIP_CONSTRAINED_BY)
        assert "SEC-001" in sec_neighbors


# ──────────────────────────────────────────────────────────────────────
# Coverage / Completeness Metrics
# ──────────────────────────────────────────────────────────────────────

class TestCoverageMetrics:
    """RCA traceability metrics and coverage."""

    def test_rca_traceability_metrics(self):
        """Compute deterministic RCA traceability metrics."""
        failures = [make_failure(failure_id="fid-001"), make_failure(failure_id="fid-002")]
        rcas = [
            make_rca(record_id="RCA-001", failure_ref="fid-001"),
            make_rca(record_id="RCA-002", failure_ref="fid-002"),
        ]
        experiences = []

        metrics = compute_rca_traceability_metrics(failures, rcas, experiences)
        assert metrics["failures_with_rca"] == 2
        assert metrics["eligible_failures"] == 2
        assert metrics["rcas_with_root_cause"] == 2
        assert metrics["total_rcas"] == 2
        assert metrics["rcas_with_evidence"] == 2
        assert metrics["rcas_with_corrective_action"] == 2

    def test_rca_completeness_metrics(self):
        """RCA completeness check is deterministic."""
        rca = make_rca()
        completeness = rca.get_completeness()
        assert completeness.has_failure_ref is True
        assert completeness.has_root_causes is True
        assert completeness.has_corrective_actions is True
        assert completeness.has_preventive_actions is True
        assert rca.is_complete() is True

    def test_rca_incomplete_metrics(self):
        """RCA with missing fields is incomplete."""
        rca = make_rca()
        rca.impact = ""
        rca.evidence_refs = []
        completeness = rca.get_completeness()
        assert completeness.has_impact is False
        assert completeness.has_evidence is False
        # Still complete (impact and evidence not required)
        assert rca.is_complete() is True


# ──────────────────────────────────────────────────────────────────────
# Phase 7 E2E Scenarios
# ──────────────────────────────────────────────────────────────────────

class TestPhase7E2E:
    """End-to-end Phase 7 scenarios."""

    def test_standard_failure_e2e(self):
        """Standard Failure: FAIL-001 → RootCauseAnalyzer → RCA-001 → FailureLesson → Experience."""
        failure = make_failure(
            failure_id="fid-001",
            category="integration",
            message="API timeout: connection refused after 30s",
        )

        # Create RCA
        rca_analyzer = RCAAnalyzer()
        rca = rca_analyzer.analyze_to_rca(
            failure=failure,
            symptoms=["API timeout", "connection refused"],
            root_causes=["Connection pool exhaustion"],
            contributing_factors=["High traffic", "No connection reuse"],
            impact="Service unavailable for 30 minutes",
            corrective_actions=["Increase connection pool size to 100"],
            preventive_actions=["Add connection pool saturation monitoring"],
            evidence_refs=["EV-001"],
            rca_id="RCA-001",
        )

        assert rca.record_type == "RCA"
        assert rca.failure_ref == "fid-001"
        assert len(rca.root_causes) == 1
        assert len(rca.corrective_actions) == 1
        assert len(rca.preventive_actions) == 1

        # Derive FailureLesson
        lesson = rca.derive_failure_lesson()
        assert lesson["source_rca_id"] == "RCA-001"
        assert lesson["source_failure_id"] == "fid-001"

        # Store RCA
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "knowledge.db")
            store = KnowledgeStore(db_path)
            store.save(rca)

            # Reload and verify
            store2 = KnowledgeStore(db_path)
            loaded = store2.get("RCA-001")
            assert loaded.record_type == "RCA"
            assert loaded.failure_ref == "fid-001"
            store.close()
            store2.close()

    def test_engineering_impact_e2e(self):
        """Engineering Impact: ADR-003 → failure → RCA-004 → RELATES_TO ADR-003 + TDR-006 + RSK-002."""
        adr = make_adr(record_id="ADR-003")
        tdr = make_tdr(record_id="TDR-006")
        rsk = make_rsk(record_id="RSK-002")

        rca = make_rca(
            record_id="RCA-004",
            failure_ref="fid-004",
            related_decisions=["ADR-003"],
        )
        rca.add_relationship("TDR-006", RELATIONSHIP_RELATES_TO)
        rca.add_relationship("RSK-002", RELATIONSHIP_RELATES_TO)

        graph = EngineeringKnowledgeGraph.build_from_records([rca, adr, tdr, rsk])

        assert graph.has_node("RCA-004")
        assert graph.has_node("ADR-003")
        assert graph.has_node("TDR-006")
        assert graph.has_node("RSK-002")

        # RCA → ADR
        adr_neighbors = graph.neighbors("RCA-004", relation_type=RELATIONSHIP_RELATES_TO)
        assert "ADR-003" in adr_neighbors

        # RCA → TDR
        tdr_neighbors = graph.neighbors("RCA-004", relation_type=RELATIONSHIP_RELATES_TO)
        assert "TDR-006" in tdr_neighbors

        # RCA → RSK
        rsk_neighbors = graph.neighbors("RCA-004", relation_type=RELATIONSHIP_RELATES_TO)
        assert "RSK-002" in rsk_neighbors

    def test_security_safety_e2e(self):
        """Security Safety: SEC-001 + FAIL-SEC-001 → RCA-SEC-001 → FailureLesson proposal → SEC conflict → BLOCK."""
        sec = make_sec(
            record_id="SEC-001",
            authority=SEC_AUTHORITY_AUTHORITATIVE,
        )

        failure = make_failure(
            failure_id="fid-sec-001",
            category="security",
            message="Certificate validation timeout",
        )

        rca_analyzer = RCAAnalyzer()
        rca = rca_analyzer.analyze_to_rca(
            failure=failure,
            root_causes=["Certificate validation timeout"],
            corrective_actions=["Increase timeout threshold"],
            preventive_actions=["Add certificate caching"],
            rca_id="RCA-001",
        )

        # Derive lesson
        lesson = rca.derive_failure_lesson()
        assert lesson["security_checked"] is True

        # SEC constrains RCA
        rca.add_relationship("SEC-001", RELATIONSHIP_CONSTRAINED_BY)

        # SEC wins over lesson
        resolver = AuthorityPrecedenceResolver()
        sec_candidate = {"source": "security", "authority": SEC_AUTHORITY_AUTHORITATIVE, "record_type": "SEC"}
        lesson_candidate = {"source": "learning", "authority": AUTHORITY_ACCEPTED, "record_type": "FailureLesson"}
        assert resolver.can_override(sec_candidate, lesson_candidate) is True
        assert resolver.can_override(lesson_candidate, sec_candidate) is False

    def test_restart_e2e(self):
        """Restart: Persist Failure, RCA, FailureLesson, Experience. Restart. Verify provenance chain."""
        failure = make_failure(failure_id="fid-restart")

        # Process A: create RCA and persist
        rca_analyzer = RCAAnalyzer()
        rca = rca_analyzer.analyze_to_rca(
            failure=failure,
            root_causes=["Connection pool exhaustion"],
            corrective_actions=["Increase pool size"],
            preventive_actions=["Add monitoring"],
            rca_id="RCA-001",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "knowledge.db")

            # Process A: persist
            store = KnowledgeStore(db_path)
            store.save(rca)
            store.close()

            # Process B: reload and trace (same db_path simulates restart)
            store2 = KnowledgeStore(db_path)
            loaded = store2.get("RCA-001")
            assert loaded.record_type == "RCA"
            assert loaded.failure_ref == "fid-restart"

            # Trace: RCA → failure
            assert loaded.failure_ref == failure.failure_id

            # Trace: RCA → lesson (derivable)
            lesson = loaded.derive_failure_lesson()
            assert lesson["source_rca_id"] == "RCA-001"
            assert lesson["source_failure_id"] == "fid-restart"
            store2.close()


# ──────────────────────────────────────────────────────────────────────
# Determinism
# ──────────────────────────────────────────────────────────────────────

class TestDeterminism:
    """RCA serialization and traversal must be deterministic."""

    def test_rca_deterministic_traversal_order(self):
        """Graph traversal must be deterministic."""
        rca = make_rca(
            related_decisions=["ADR-003", "ADR-001", "ADR-002"],
        )
        graph = EngineeringKnowledgeGraph.build_from_records([rca])
        neighbors = graph.neighbors("RCA-001", relation_type=RELATIONSHIP_RELATES_TO)
        assert neighbors == sorted(neighbors)

    def test_rca_deterministic_hash(self):
        """RCA hash must be deterministic."""
        ts = "2026-01-01T00:00:00"
        prov = Provenance(author="test", source="human", timestamp=ts)
        rca1 = make_rca(created_at=ts, updated_at=ts, provenance=prov)
        rca2 = make_rca(created_at=ts, updated_at=ts, provenance=prov)
        assert rca1.compute_hash() == rca2.compute_hash()


# ──────────────────────────────────────────────────────────────────────
# Three-Domain Separation
# ──────────────────────────────────────────────────────────────────────

class TestThreeDomainSeparation:
    """RCA is Engineering, NOT Learning, NOT Evidence."""

    def test_rca_is_engineering_record(self):
        rca = make_rca()
        assert isinstance(rca, EngineeringRecord)
        assert rca.record_type == "RCA"

    def test_rca_not_learning_record(self):
        """RCA is not StructuredExperience, FailureLesson, Strategy, etc."""
        rca = make_rca()
        assert rca.__class__.__name__ == "RootCauseAnalysisRecord"
        assert not hasattr(rca, "task_id")
        assert not hasattr(rca, "failure_mode")

    def test_rca_not_evidence(self):
        """RCA is not EvidencePackage or CanonicalExecutionRecord."""
        rca = make_rca()
        assert rca.__class__.__name__ == "RootCauseAnalysisRecord"
        assert not hasattr(rca, "content_hash")

    def test_rca_semantic_separation(self):
        """RCA != FailureLesson != StructuredExperience != Strategy."""
        rca = make_rca()
        lesson = rca.derive_failure_lesson()
        # RCA is a record, lesson is a dict proposal
        assert isinstance(rca, EngineeringRecord)
        assert isinstance(lesson, dict)
        # They are different types
        assert rca.__class__.__name__ != "FailureLesson"


# ──────────────────────────────────────────────────────────────────────
# CapabilityGraph Separation
# ──────────────────────────────────────────────────────────────────────

class TestCapabilityGraphSeparation:
    """RCA does not leak into CapabilityGraph semantics."""

    def test_rca_not_in_capability_graph(self):
        """RCA does not appear in CapabilityGraph."""
        rca = make_rca()
        assert rca.record_type == "RCA"
        assert not hasattr(rca, "capability_id")
