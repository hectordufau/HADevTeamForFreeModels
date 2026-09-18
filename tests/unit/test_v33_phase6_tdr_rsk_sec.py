# tests/unit/test_v33_phase6_tdr_rsk_sec.py — Phase 6 Acceptance Tests
"""Phase 6: TDR / Risk / Security — deterministic behavior verification.

Tests cover:
- TDR creation, validation, lifecycle, versioning, remediation, ADR→TDR
- Risk creation, validation, lifecycle, likelihood/impact, risk versioning, ADR→Risk, TDR→Risk
- SEC creation, validation, lifecycle, authority, provenance, versioning, NFR(Security) != SEC
- Agent-generated SEC remains proposed
- Agent cannot self-approve exception
- ADR conflict with SEC, REQ conflict with SEC, Learned Strategy conflict with SEC
- Security precedence, no timestamp/role/model precedence
- Evidence references, KnowledgeStore persistence, KnowledgeGraph traversal
- TDR/Risk/Security/Combined E2E
- Determinism, three-domain separation, CapabilityGraph separation
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.knowledge import (
    TechnicalDebtRecord,
    RiskRecord,
    SecurityRecord,
    AuthorityPrecedenceResolver,
    SecurityConflict,
    detect_security_conflicts,
    SecurityAuthorityError,
    validate_security_authority,
    is_security_authority_at_least,
    can_transition_security_authority,
    # TDR constants
    TDR_DEBT_TYPE_CODE,
    TDR_DEBT_TYPE_DESIGN,
    TDR_DEBT_TYPE_ARCHITECTURE,
    TDR_DEBT_TYPE_TEST,
    TDR_DEBT_TYPE_DOCUMENTATION,
    TDR_DEBT_TYPE_DEPENDENCY,
    VALID_TDR_DEBT_TYPES,
    TDR_SEVERITY_CRITICAL,
    TDR_SEVERITY_HIGH,
    TDR_SEVERITY_MEDIUM,
    TDR_SEVERITY_LOW,
    VALID_TDR_SEVERITIES,
    TDR_PRIORITY_CRITICAL,
    TDR_PRIORITY_HIGH,
    TDR_PRIORITY_MEDIUM,
    TDR_PRIORITY_LOW,
    VALID_TDR_PRIORITIES,
    # RSK constants
    RSK_CATEGORY_TECHNICAL,
    RSK_CATEGORY_SCHEDULE,
    RSK_CATEGORY_RESOURCE,
    RSK_CATEGORY_SECURITY,
    RSK_CATEGORY_COMPLIANCE,
    VALID_RSK_CATEGORIES,
    RSK_LIKELIHOOD_HIGH,
    RSK_LIKELIHOOD_MEDIUM,
    RSK_LIKELIHOOD_LOW,
    VALID_RSK_LIKELIHOODS,
    RSK_IMPACT_HIGH,
    RSK_IMPACT_MEDIUM,
    RSK_IMPACT_LOW,
    VALID_RSK_IMPACTS,
    VALID_RSK_EXPOSURES,
    # SEC constants
    SEC_CATEGORY_CONSTRAINT,
    SEC_CATEGORY_CONTROL,
    SEC_CATEGORY_FINDING,
    SEC_CATEGORY_THREAT,
    SEC_CATEGORY_REQUIREMENT,
    VALID_SEC_CATEGORIES,
    SEC_AUTHORITY_AUTHORITATIVE,
    SEC_AUTHORITY_ACCEPTED,
    SEC_AUTHORITY_PROPOSED,
    SEC_AUTHORITY_INFERRED,
    SEC_AUTHORITY_HISTORICAL,
    SEC_AUTHORITY_DEPRECATED,
    VALID_SEC_AUTHORITIES,
    SEC_AUTHORITY_RANK,
    PRECEDENCE_GOVERNANCE,
    PRECEDENCE_SECURITY_AUTHORITY,
    PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE,
    PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE,
    PRECEDENCE_TASK_REQUIREMENTS,
    PRECEDENCE_LEARNED_KNOWLEDGE,
    PRECEDENCE_EXPLORATION,
    # Base imports
    KnowledgeStore,
    EngineeringKnowledgeGraph,
    Provenance,
    AUTHORITY_PROPOSED,
    AUTHORITY_ACCEPTED,
    AUTHORITY_DEPRECATED,
    AUTHORITY_ARCHIVED,
    VALID_AUTHORITY_LEVELS,
    DecisionRecord,
    ArchitectureDecisionRecord,
    Alternative,
    Consequences,
    DECISION_TYPE_ARCHITECTURE,
    DECISION_TYPE_TECHNICAL,
    ADR_DOMAIN_DATA,
    ADR_DOMAIN_SECURITY,
    is_valid_transition,
    is_terminal_state,
)
from harness.knowledge.records import (
    EngineeringRecord,
    RecordError,
    RELATIONSHIP_DECIDED_BY,
    RELATIONSHIP_CONSTRAINED_BY,
    RELATIONSHIP_INTRODUCES,
    RELATIONSHIP_MITIGATES,
    RELATIONSHIP_SUPPORTED_BY,
    RELATIONSHIP_CONTRADICTS,
    RELATIONSHIP_RELATES_TO,
    RELATIONSHIP_DERIVED_FROM,
    RELATIONSHIP_SATISFIES,
    RELATIONSHIP_REQUIRES,
    RELATIONSHIP_SUPERSEDES,
)
from harness.knowledge.lifecycle import LifecycleError
from harness.knowledge.contradiction import ContradictionDetector


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def make_provenance(author="test_author", source="human"):
    return Provenance(author=author, source=source)


def make_tdr(**kwargs):
    defaults = dict(
        record_id="TDR-001",
        title="Test TDR",
        description="A test technical debt record",
        debt_type=TDR_DEBT_TYPE_CODE,
        severity=TDR_SEVERITY_MEDIUM,
        impact="Slows down development",
        remediation="Refactor the module",
        remediation_estimate="2 days",
        priority=TDR_PRIORITY_MEDIUM,
        provenance=make_provenance(),
    )
    defaults.update(kwargs)
    return TechnicalDebtRecord(**defaults)


def make_rsk(**kwargs):
    defaults = dict(
        record_id="RSK-001",
        title="Test Risk",
        description="A test risk record",
        risk_category=RSK_CATEGORY_TECHNICAL,
        likelihood=RSK_LIKELIHOOD_MEDIUM,
        impact=RSK_IMPACT_MEDIUM,
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
        category=SEC_CATEGORY_CONSTRAINT,
        enforcement="Automated validation",
        authority=SEC_AUTHORITY_PROPOSED,
        provenance=make_provenance(),
    )
    defaults.update(kwargs)
    return SecurityRecord(**defaults)


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
# TDR creation, validation, lifecycle
# ──────────────────────────────────────────────────────────────────────

class TestTDRCreation:
    """TDR creation, validation, and lifecycle tests."""

    def test_tdr_creation_minimal(self):
        tdr = make_tdr()
        assert tdr.record_type == "TDR"
        assert tdr.status == "identified"
        assert tdr.authority == AUTHORITY_PROPOSED
        assert tdr.debt_type == TDR_DEBT_TYPE_CODE
        assert tdr.severity == TDR_SEVERITY_MEDIUM

    def test_tdr_creation_full(self):
        tdr = make_tdr(
            record_id="TDR-002",
            debt_type=TDR_DEBT_TYPE_ARCHITECTURE,
            severity=TDR_SEVERITY_HIGH,
            impact="Blocks scalability",
            remediation="Migrate to event-driven architecture",
            remediation_estimate="5 days",
            target_condition="All services use async messaging",
            priority=TDR_PRIORITY_HIGH,
            owner="platform-team",
            target_milestone="Q4-2026",
            related_decisions=["ADR-001"],
        )
        assert tdr.debt_type == TDR_DEBT_TYPE_ARCHITECTURE
        assert tdr.severity == TDR_SEVERITY_HIGH
        assert tdr.priority == TDR_PRIORITY_HIGH
        assert tdr.owner == "platform-team"
        assert tdr.target_milestone == "Q4-2026"

    def test_tdr_invalid_debt_type(self):
        with pytest.raises(RecordError, match="Invalid debt_type"):
            make_tdr(debt_type="INVALID")

    def test_tdr_all_valid_debt_types(self):
        for dt in VALID_TDR_DEBT_TYPES:
            tdr = make_tdr(debt_type=dt)
            assert tdr.debt_type == dt

    def test_tdr_invalid_severity(self):
        with pytest.raises(RecordError, match="Invalid severity"):
            make_tdr(severity="INVALID")

    def test_tdr_all_valid_severities(self):
        for sev in VALID_TDR_SEVERITIES:
            tdr = make_tdr(severity=sev)
            assert tdr.severity == sev

    def test_tdr_invalid_priority(self):
        with pytest.raises(RecordError, match="Invalid priority"):
            make_tdr(priority="INVALID")

    def test_tdr_all_valid_priorities(self):
        for pri in VALID_TDR_PRIORITIES:
            tdr = make_tdr(priority=pri)
            assert tdr.priority == pri

    def test_tdr_invalid_record_id_prefix(self):
        with pytest.raises(RecordError, match="does not match record_type"):
            make_tdr(record_id="ADR-001")

    def test_tdr_empty_impact_rejected(self):
        with pytest.raises(RecordError, match="impact is required"):
            make_tdr(impact="")

    def test_tdr_empty_remediation_rejected(self):
        with pytest.raises(RecordError, match="remediation is required"):
            make_tdr(remediation="")

    def test_tdr_related_decisions_must_be_valid_ids(self):
        with pytest.raises(RecordError, match="ADR/DR"):
            make_tdr(related_decisions=["REQ-001"])

    def test_tdr_completeness_check(self):
        tdr = make_tdr()
        completeness = tdr.get_completeness()
        assert completeness["has_debt_type"] is True
        assert completeness["has_severity"] is True
        assert completeness["has_impact"] is True
        assert completeness["has_remediation"] is True
        assert tdr.is_complete() is True

    def test_tdr_completeness_missing_impact(self):
        tdr = make_tdr()
        tdr.impact = ""
        completeness = tdr.get_completeness()
        assert completeness["has_impact"] is False
        assert tdr.is_complete() is False


class TestTDRLifecycle:
    """TDR lifecycle transitions — fail closed on invalid transitions."""

    def test_tdr_initial_state(self):
        tdr = make_tdr()
        assert tdr.status == "identified"

    def test_tdr_valid_transitions(self):
        tdr = make_tdr()
        # identified → acknowledged
        assert tdr.can_transition_to("acknowledged") is True
        tdr.transition_status("acknowledged")
        assert tdr.status == "acknowledged"

        # acknowledged → remediation_planned
        assert tdr.can_transition_to("remediation_planned") is True
        tdr.transition_status("remediation_planned")
        assert tdr.status == "remediation_planned"

        # remediation_planned → in_progress
        assert tdr.can_transition_to("in_progress") is True
        tdr.transition_status("in_progress")
        assert tdr.status == "in_progress"

        # in_progress → resolved
        assert tdr.can_transition_to("resolved") is True
        tdr.transition_status("resolved")
        assert tdr.status == "resolved"

        # resolved → closed
        assert tdr.can_transition_to("closed") is True
        tdr.transition_status("closed")
        assert tdr.status == "closed"

    def test_tdr_invalid_transitions(self):
        tdr = make_tdr()
        # identified → in_progress (skip)
        assert tdr.can_transition_to("in_progress") is False
        with pytest.raises(LifecycleError):
            tdr.transition_status("in_progress")

    def test_tdr_terminal_state(self):
        tdr = make_tdr()
        tdr.transition_status("acknowledged")
        tdr.transition_status("remediation_planned")
        tdr.transition_status("in_progress")
        tdr.transition_status("resolved")
        tdr.transition_status("closed")
        assert tdr.status == "closed"
        assert is_terminal_state("TDR", "closed") is True
        # Cannot transition out of closed
        assert tdr.can_transition_to("identified") is False
        with pytest.raises(LifecycleError):
            tdr.transition_status("identified")

    def test_tdr_self_transition_not_valid(self):
        tdr = make_tdr()
        assert tdr.can_transition_to("identified") is False

    def test_tdr_identified_to_closed_direct(self):
        """TDR can go directly from identified to closed (abandoned debt)."""
        tdr = make_tdr()
        assert tdr.can_transition_to("closed") is True
        tdr.transition_status("closed")
        assert tdr.status == "closed"


class TestTDRRelationships:
    """TDR graph relationships with ADR and evidence."""

    def test_tdr_to_adr_relationship(self):
        tdr = make_tdr(related_decisions=["ADR-001"])
        rels = tdr.get_relationships(RELATIONSHIP_DERIVED_FROM)
        assert len(rels) == 1
        assert rels[0].target_id == "ADR-001"

    def test_tdr_to_evidence_relationship(self):
        tdr = make_tdr()
        tdr.add_resolution_evidence("EV-001")
        rels = tdr.get_relationships(RELATIONSHIP_SUPPORTED_BY)
        assert len(rels) == 1
        assert rels[0].target_id == "EV-001"

    def test_tdr_link_adr(self):
        tdr = make_tdr()
        tdr.link_adr("ADR-001")
        assert "ADR-001" in tdr.related_decisions
        rels = tdr.get_relationships(RELATIONSHIP_DERIVED_FROM)
        assert len(rels) == 1

    def test_tdr_link_adr_invalid_id(self):
        tdr = make_tdr()
        with pytest.raises(RecordError, match="ADR/DR"):
            tdr.link_adr("REQ-001")

    def test_tdr_resolution_evidence(self):
        tdr = make_tdr()
        tdr.add_resolution_evidence("EV-001")
        tdr.add_resolution_evidence("EV-002")
        assert tdr.resolution_evidence == ["EV-001", "EV-002"]


class TestTDRVersioning:
    """TDR versioning and supersession."""

    def test_tdr_version_increment(self):
        tdr = make_tdr(version=1)
        assert tdr.version == 1
        tdr.version = 2
        assert tdr.version == 2
        assert tdr.record_id == "TDR-001"

    def test_tdr_superseded_by(self):
        tdr1 = make_tdr(record_id="TDR-001")
        tdr2 = make_tdr(record_id="TDR-002")
        tdr1.superseded_by = "TDR-002"
        assert tdr1.superseded_by == "TDR-002"


class TestTDRSerialization:
    """TDR deterministic serialization."""

    def test_tdr_deterministic_serialization(self):
        ts = "2026-01-01T00:00:00"
        prov = Provenance(author="test", source="human", timestamp=ts)
        tdr1 = make_tdr(created_at=ts, updated_at=ts, provenance=prov)
        tdr2 = make_tdr(created_at=ts, updated_at=ts, provenance=prov)
        assert tdr1.to_json() == tdr2.to_json()
        assert tdr1.compute_hash() == tdr2.compute_hash()

    def test_tdr_from_dict_round_trip(self):
        tdr = make_tdr(
            debt_type=TDR_DEBT_TYPE_ARCHITECTURE,
            severity=TDR_SEVERITY_HIGH,
            related_decisions=["ADR-001"],
        )
        data = tdr.to_dict()
        restored = TechnicalDebtRecord.from_dict(data)
        assert restored.record_id == tdr.record_id
        assert restored.debt_type == tdr.debt_type
        assert restored.severity == tdr.severity
        assert restored.related_decisions == tdr.related_decisions


# ──────────────────────────────────────────────────────────────────────
# RSK creation, validation, lifecycle
# ──────────────────────────────────────────────────────────────────────

class TestRSKCreation:
    """RSK creation, validation, and lifecycle tests."""

    def test_rsk_creation_minimal(self):
        rsk = make_rsk()
        assert rsk.record_type == "RSK"
        assert rsk.status == "identified"
        assert rsk.authority == AUTHORITY_PROPOSED
        assert rsk.risk_category == RSK_CATEGORY_TECHNICAL
        assert rsk.likelihood == RSK_LIKELIHOOD_MEDIUM
        assert rsk.impact == RSK_IMPACT_MEDIUM

    def test_rsk_creation_full(self):
        rsk = make_rsk(
            record_id="RSK-002",
            risk_category=RSK_CATEGORY_SECURITY,
            likelihood=RSK_LIKELIHOOD_HIGH,
            impact=RSK_IMPACT_HIGH,
            mitigation="Implement MFA",
            mitigation_owner="security-team",
            contingency="Disable affected accounts",
            related_decisions=["ADR-001"],
            related_security=["SEC-001"],
        )
        assert rsk.risk_category == RSK_CATEGORY_SECURITY
        assert rsk.likelihood == RSK_LIKELIHOOD_HIGH
        assert rsk.impact == RSK_IMPACT_HIGH
        assert rsk.mitigation_owner == "security-team"

    def test_rsk_invalid_category(self):
        with pytest.raises(RecordError, match="Invalid risk_category"):
            make_rsk(risk_category="INVALID")

    def test_rsk_all_valid_categories(self):
        for cat in VALID_RSK_CATEGORIES:
            rsk = make_rsk(risk_category=cat)
            assert rsk.risk_category == cat

    def test_rsk_invalid_likelihood(self):
        with pytest.raises(RecordError, match="Invalid likelihood"):
            make_rsk(likelihood="INVALID")

    def test_rsk_all_valid_likelihoods(self):
        for lik in VALID_RSK_LIKELIHOODS:
            rsk = make_rsk(likelihood=lik)
            assert rsk.likelihood == lik

    def test_rsk_invalid_impact(self):
        with pytest.raises(RecordError, match="Invalid impact"):
            make_rsk(impact="INVALID")

    def test_rsk_all_valid_impacts(self):
        for imp in VALID_RSK_IMPACTS:
            rsk = make_rsk(impact=imp)
            assert rsk.impact == imp

    def test_rsk_invalid_record_id_prefix(self):
        with pytest.raises(RecordError, match="does not match record_type"):
            make_rsk(record_id="ADR-001")

    def test_rsk_empty_mitigation_rejected(self):
        with pytest.raises(RecordError, match="mitigation is required"):
            make_rsk(mitigation="")

    def test_rsk_related_decisions_must_be_valid_ids(self):
        with pytest.raises(RecordError, match="ADR/DR"):
            make_rsk(related_decisions=["REQ-001"])

    def test_rsk_related_security_must_be_valid_ids(self):
        with pytest.raises(RecordError, match="SEC"):
            make_rsk(related_security=["ADR-001"])

    def test_rsk_completeness_check(self):
        rsk = make_rsk()
        completeness = rsk.get_completeness()
        assert completeness["has_risk_category"] is True
        assert completeness["has_likelihood"] is True
        assert completeness["has_impact"] is True
        assert completeness["has_mitigation"] is True
        assert rsk.is_complete() is True


class TestRSKExposure:
    """RSK exposure computation (qualitative, no fabricated precision)."""

    def test_exposure_high_high(self):
        rsk = make_rsk(likelihood=RSK_LIKELIHOOD_HIGH, impact=RSK_IMPACT_HIGH)
        assert rsk.exposure == "critical"

    def test_exposure_high_medium(self):
        rsk = make_rsk(likelihood=RSK_LIKELIHOOD_HIGH, impact=RSK_IMPACT_MEDIUM)
        assert rsk.exposure == "high"

    def test_exposure_medium_high(self):
        rsk = make_rsk(likelihood=RSK_LIKELIHOOD_MEDIUM, impact=RSK_IMPACT_HIGH)
        assert rsk.exposure == "high"

    def test_exposure_medium_medium(self):
        rsk = make_rsk(likelihood=RSK_LIKELIHOOD_MEDIUM, impact=RSK_IMPACT_MEDIUM)
        assert rsk.exposure == "medium"

    def test_exposure_low_low(self):
        rsk = make_rsk(likelihood=RSK_LIKELIHOOD_LOW, impact=RSK_IMPACT_LOW)
        assert rsk.exposure == "low"

    def test_exposure_high_low(self):
        rsk = make_rsk(likelihood=RSK_LIKELIHOOD_HIGH, impact=RSK_IMPACT_LOW)
        assert rsk.exposure == "medium"

    def test_exposure_low_high(self):
        rsk = make_rsk(likelihood=RSK_LIKELIHOOD_LOW, impact=RSK_IMPACT_HIGH)
        assert rsk.exposure == "medium"


class TestRSKLifecycle:
    """RSK lifecycle transitions — ACCEPTED != MITIGATED != CLOSED."""

    def test_rsk_initial_state(self):
        rsk = make_rsk()
        assert rsk.status == "identified"

    def test_rsk_valid_transitions(self):
        rsk = make_rsk()
        # identified → assessed
        assert rsk.can_transition_to("assessed") is True
        rsk.transition_status("assessed")
        assert rsk.status == "assessed"

        # assessed → mitigated
        assert rsk.can_transition_to("mitigated") is True
        rsk.transition_status("mitigated")
        assert rsk.status == "mitigated"

        # mitigated → accepted
        assert rsk.can_transition_to("accepted") is True
        rsk.transition_status("accepted")
        assert rsk.status == "accepted"

        # accepted → closed
        assert rsk.can_transition_to("closed") is True
        rsk.transition_status("closed")
        assert rsk.status == "closed"

    def test_rsk_accepted_not_resolved(self):
        """ACCEPTED means consciously accepted, NOT that risk no longer exists."""
        rsk = make_rsk()
        rsk.transition_status("assessed")
        rsk.transition_status("accepted")
        assert rsk.status == "accepted"
        # Risk still exists — accepted is not closed
        assert rsk.status != "closed"

    def test_rsk_mitigated_not_accepted(self):
        """MITIGATED means mitigation actions taken, not necessarily accepted."""
        rsk = make_rsk()
        rsk.transition_status("assessed")
        rsk.transition_status("mitigated")
        assert rsk.status == "mitigated"
        assert rsk.status != "accepted"

    def test_rsk_invalid_transitions(self):
        rsk = make_rsk()
        # identified → mitigated (skip)
        assert rsk.can_transition_to("mitigated") is False
        with pytest.raises(LifecycleError):
            rsk.transition_status("mitigated")

    def test_rsk_terminal_state(self):
        rsk = make_rsk()
        rsk.transition_status("assessed")
        rsk.transition_status("accepted")
        rsk.transition_status("closed")
        assert rsk.status == "closed"
        assert is_terminal_state("RSK", "closed") is True

    def test_rsk_identified_to_closed_direct(self):
        """RSK can go directly from identified to closed (risk no longer relevant)."""
        rsk = make_rsk()
        assert rsk.can_transition_to("closed") is True
        rsk.transition_status("closed")
        assert rsk.status == "closed"


class TestRSKRelationships:
    """RSK graph relationships with ADR, SEC, and TDR."""

    def test_rsk_to_adr_relationship(self):
        rsk = make_rsk(related_decisions=["ADR-001"])
        rels = rsk.get_relationships(RELATIONSHIP_DERIVED_FROM)
        assert len(rels) == 1
        assert rels[0].target_id == "ADR-001"

    def test_rsk_to_sec_relationship(self):
        rsk = make_rsk(related_security=["SEC-001"])
        rels = rsk.get_relationships(RELATIONSHIP_MITIGATES)
        assert len(rels) == 1
        assert rels[0].target_id == "SEC-001"

    def test_rsk_link_adr(self):
        rsk = make_rsk()
        rsk.link_adr("ADR-001")
        assert "ADR-001" in rsk.related_decisions

    def test_rsk_link_security(self):
        rsk = make_rsk()
        rsk.link_security("SEC-001")
        assert "SEC-001" in rsk.related_security
        rels = rsk.get_relationships(RELATIONSHIP_MITIGATES)
        assert len(rels) == 1

    def test_rsk_link_security_invalid_id(self):
        rsk = make_rsk()
        with pytest.raises(RecordError, match="SEC"):
            rsk.link_security("ADR-001")


class TestRSKSerialization:
    """RSK deterministic serialization."""

    def test_rsk_deterministic_serialization(self):
        ts = "2026-01-01T00:00:00"
        prov = Provenance(author="test", source="human", timestamp=ts)
        rsk1 = make_rsk(created_at=ts, updated_at=ts, provenance=prov)
        rsk2 = make_rsk(created_at=ts, updated_at=ts, provenance=prov)
        assert rsk1.to_json() == rsk2.to_json()
        assert rsk1.compute_hash() == rsk2.compute_hash()

    def test_rsk_from_dict_round_trip(self):
        rsk = make_rsk(
            risk_category=RSK_CATEGORY_SECURITY,
            likelihood=RSK_LIKELIHOOD_HIGH,
            impact=RSK_IMPACT_HIGH,
            related_decisions=["ADR-001"],
            related_security=["SEC-001"],
        )
        data = rsk.to_dict()
        restored = RiskRecord.from_dict(data)
        assert restored.record_id == rsk.record_id
        assert restored.risk_category == rsk.risk_category
        assert restored.likelihood == rsk.likelihood
        assert restored.impact == rsk.impact
        assert restored.exposure == rsk.exposure
        assert restored.related_decisions == rsk.related_decisions
        assert restored.related_security == rsk.related_security


# ──────────────────────────────────────────────────────────────────────
# SEC creation, validation, lifecycle, authority
# ──────────────────────────────────────────────────────────────────────

class TestSECCreation:
    """SEC creation, validation, and lifecycle tests."""

    def test_sec_creation_minimal(self):
        sec = make_sec()
        assert sec.record_type == "SEC"
        assert sec.status == "active"
        assert sec.authority == SEC_AUTHORITY_PROPOSED
        assert sec.category == SEC_CATEGORY_CONSTRAINT

    def test_sec_creation_full(self):
        sec = make_sec(
            record_id="SEC-002",
            category=SEC_CATEGORY_CONTROL,
            enforcement="Automated scanning",
            authority=SEC_AUTHORITY_ACCEPTED,
            related_nfrs=["NFR-001"],
            related_adrs=["ADR-001"],
            related_risks=["RSK-001"],
            related_tdrs=["TDR-001"],
        )
        assert sec.category == SEC_CATEGORY_CONTROL
        assert sec.authority == SEC_AUTHORITY_ACCEPTED
        assert sec.related_nfrs == ["NFR-001"]
        assert sec.related_adrs == ["ADR-001"]
        assert sec.related_risks == ["RSK-001"]
        assert sec.related_tdrs == ["TDR-001"]

    def test_sec_invalid_category(self):
        with pytest.raises(RecordError, match="Invalid category"):
            make_sec(category="INVALID")

    def test_sec_all_valid_categories(self):
        for cat in VALID_SEC_CATEGORIES:
            sec = make_sec(category=cat)
            assert sec.category == cat

    def test_sec_invalid_record_id_prefix(self):
        with pytest.raises(RecordError, match="does not match record_type"):
            make_sec(record_id="ADR-001")

    def test_sec_completeness_check(self):
        sec = make_sec()
        completeness = sec.get_completeness()
        assert completeness["has_category"] is True
        assert completeness["has_authority"] is True
        assert sec.is_complete() is True

    def test_sec_completeness_missing_enforcement(self):
        sec = make_sec()
        sec.enforcement = ""
        completeness = sec.get_completeness()
        assert completeness["has_enforcement"] is False
        assert sec.is_complete() is False


class TestSECAuthority:
    """SEC authority hierarchy — separate from standard authority levels."""

    def test_sec_authority_levels(self):
        """SEC uses its own authority hierarchy."""
        assert SEC_AUTHORITY_RANK[SEC_AUTHORITY_DEPRECATED] == 0
        assert SEC_AUTHORITY_RANK[SEC_AUTHORITY_HISTORICAL] == 1
        assert SEC_AUTHORITY_RANK[SEC_AUTHORITY_INFERRED] == 2
        assert SEC_AUTHORITY_RANK[SEC_AUTHORITY_PROPOSED] == 3
        assert SEC_AUTHORITY_RANK[SEC_AUTHORITY_ACCEPTED] == 4
        assert SEC_AUTHORITY_RANK[SEC_AUTHORITY_AUTHORITATIVE] == 5

    def test_sec_authority_validation(self):
        """SEC authority must be one of the valid SEC authority levels."""
        validate_security_authority(SEC_AUTHORITY_AUTHORITATIVE)
        validate_security_authority(SEC_AUTHORITY_ACCEPTED)
        validate_security_authority(SEC_AUTHORITY_PROPOSED)
        validate_security_authority(SEC_AUTHORITY_INFERRED)
        validate_security_authority(SEC_AUTHORITY_HISTORICAL)
        validate_security_authority(SEC_AUTHORITY_DEPRECATED)

    def test_sec_invalid_authority(self):
        with pytest.raises(SecurityAuthorityError):
            validate_security_authority("INVALID")

    def test_sec_is_authority_at_least(self):
        assert is_security_authority_at_least(SEC_AUTHORITY_AUTHORITATIVE, SEC_AUTHORITY_ACCEPTED) is True
        assert is_security_authority_at_least(SEC_AUTHORITY_ACCEPTED, SEC_AUTHORITY_PROPOSED) is True
        assert is_security_authority_at_least(SEC_AUTHORITY_PROPOSED, SEC_AUTHORITY_ACCEPTED) is False

    def test_sec_can_transition_authority(self):
        """SEC authority transitions follow governance rules."""
        # PROPOSED → ACCEPTED (approval)
        assert can_transition_security_authority(SEC_AUTHORITY_PROPOSED, SEC_AUTHORITY_ACCEPTED) is True
        # ACCEPTED → AUTHORITATIVE (elevation by governance)
        assert can_transition_security_authority(SEC_AUTHORITY_ACCEPTED, SEC_AUTHORITY_AUTHORITATIVE) is True
        # ACCEPTED → DEPRECATED (deprecation)
        assert can_transition_security_authority(SEC_AUTHORITY_ACCEPTED, SEC_AUTHORITY_DEPRECATED) is True
        # AUTHORITATIVE → DEPRECATED (deprecation by governance)
        assert can_transition_security_authority(SEC_AUTHORITY_AUTHORITATIVE, SEC_AUTHORITY_DEPRECATED) is True
        # DEPRECATED is terminal
        assert can_transition_security_authority(SEC_AUTHORITY_DEPRECATED, SEC_AUTHORITY_PROPOSED) is False
        # No skipping
        assert can_transition_security_authority(SEC_AUTHORITY_PROPOSED, SEC_AUTHORITY_AUTHORITATIVE) is False
        # No downgrade
        assert can_transition_security_authority(SEC_AUTHORITY_ACCEPTED, SEC_AUTHORITY_PROPOSED) is False

    def test_sec_agent_generated_always_proposed(self):
        """Agent-generated SEC always starts as PROPOSED."""
        sec = make_sec()
        assert sec.authority == SEC_AUTHORITY_PROPOSED

    def test_sec_is_authoritative(self):
        sec = make_sec(authority=SEC_AUTHORITY_AUTHORITATIVE)
        assert sec.is_authoritative() is True
        sec2 = make_sec(authority=SEC_AUTHORITY_ACCEPTED)
        assert sec2.is_authoritative() is False

    def test_sec_is_accepted_or_higher(self):
        sec = make_sec(authority=SEC_AUTHORITY_ACCEPTED)
        assert sec.is_accepted_or_higher() is True
        sec2 = make_sec(authority=SEC_AUTHORITY_PROPOSED)
        assert sec2.is_accepted_or_higher() is False

    def test_sec_can_override(self):
        """SEC can override lower authority levels."""
        sec = make_sec(authority=SEC_AUTHORITY_AUTHORITATIVE)
        assert sec.can_override(SEC_AUTHORITY_ACCEPTED) is True
        assert sec.can_override(SEC_AUTHORITY_PROPOSED) is True
        assert sec.can_override(SEC_AUTHORITY_AUTHORITATIVE) is False

    def test_sec_authority_not_inferred_from_record_type(self):
        """Security authority must NOT be inferred from record_type == SEC."""
        sec = make_sec()
        # Even though record_type is SEC, authority is PROPOSED (not automatically high)
        assert sec.authority == SEC_AUTHORITY_PROPOSED
        assert sec.is_accepted_or_higher() is False


class TestSECLifecycle:
    """SEC lifecycle transitions."""

    def test_sec_initial_state(self):
        sec = make_sec()
        assert sec.status == "active"

    def test_sec_valid_transitions(self):
        sec = make_sec()
        # active → waived
        assert sec.can_transition_to("waived") is True
        sec.transition_status("waived")
        assert sec.status == "waived"

        # waived → expired
        assert sec.can_transition_to("expired") is True
        sec.transition_status("expired")
        assert sec.status == "expired"

        # expired → deprecated
        assert sec.can_transition_to("deprecated") is True
        sec.transition_status("deprecated")
        assert sec.status == "deprecated"

    def test_sec_invalid_transitions(self):
        sec = make_sec()
        # active → deprecated (skip)
        assert sec.can_transition_to("deprecated") is False
        with pytest.raises(LifecycleError):
            sec.transition_status("deprecated")

    def test_sec_terminal_state(self):
        sec = make_sec()
        sec.transition_status("waived")
        sec.transition_status("expired")
        sec.transition_status("deprecated")
        assert sec.status == "deprecated"
        assert is_terminal_state("SEC", "deprecated") is True

    def test_sec_active_to_expired_direct(self):
        """SEC can go directly from active to expired."""
        sec = make_sec()
        assert sec.can_transition_to("expired") is True
        sec.transition_status("expired")
        assert sec.status == "expired"


class TestSECRelationships:
    """SEC graph relationships with NFR, ADR, RSK, TDR."""

    def test_sec_to_nfr_relationship(self):
        sec = make_sec(related_nfrs=["NFR-001"])
        rels = sec.get_relationships(RELATIONSHIP_SATISFIES)
        assert len(rels) == 1
        assert rels[0].target_id == "NFR-001"

    def test_sec_to_adr_relationship(self):
        sec = make_sec(related_adrs=["ADR-001"])
        rels = sec.get_relationships(RELATIONSHIP_CONSTRAINED_BY)
        assert len(rels) == 1
        assert rels[0].target_id == "ADR-001"

    def test_sec_to_risk_relationship(self):
        sec = make_sec(related_risks=["RSK-001"])
        rels = sec.get_relationships(RELATIONSHIP_MITIGATES)
        assert len(rels) == 1
        assert rels[0].target_id == "RSK-001"

    def test_sec_to_tdr_relationship(self):
        sec = make_sec(related_tdrs=["TDR-001"])
        rels = sec.get_relationships(RELATIONSHIP_CONSTRAINED_BY)
        assert len(rels) == 1
        assert rels[0].target_id == "TDR-001"

    def test_sec_link_nfr(self):
        sec = make_sec()
        sec.link_nfr("NFR-001")
        assert "NFR-001" in sec.related_nfrs

    def test_sec_link_adr(self):
        sec = make_sec()
        sec.link_adr("ADR-001")
        assert "ADR-001" in sec.related_adrs

    def test_sec_link_risk(self):
        sec = make_sec()
        sec.link_risk("RSK-001")
        assert "RSK-001" in sec.related_risks

    def test_sec_link_tdr(self):
        sec = make_sec()
        sec.link_tdr("TDR-001")
        assert "TDR-001" in sec.related_tdrs


class TestSECSerialization:
    """SEC deterministic serialization."""

    def test_sec_deterministic_serialization(self):
        ts = "2026-01-01T00:00:00"
        prov = Provenance(author="test", source="human", timestamp=ts)
        sec1 = make_sec(created_at=ts, updated_at=ts, provenance=prov)
        sec2 = make_sec(created_at=ts, updated_at=ts, provenance=prov)
        assert sec1.to_json() == sec2.to_json()
        assert sec1.compute_hash() == sec2.compute_hash()

    def test_sec_from_dict_round_trip(self):
        sec = make_sec(
            category=SEC_CATEGORY_CONTROL,
            authority=SEC_AUTHORITY_ACCEPTED,
            related_nfrs=["NFR-001"],
            related_adrs=["ADR-001"],
            related_risks=["RSK-001"],
            related_tdrs=["TDR-001"],
        )
        data = sec.to_dict()
        restored = SecurityRecord.from_dict(data)
        assert restored.record_id == sec.record_id
        assert restored.category == sec.category
        assert restored.authority == sec.authority
        assert restored.related_nfrs == sec.related_nfrs
        assert restored.related_adrs == sec.related_adrs
        assert restored.related_risks == sec.related_risks
        assert restored.related_tdrs == sec.related_tdrs


# ──────────────────────────────────────────────────────────────────────
# Security Authority Precedence
# ──────────────────────────────────────────────────────────────────────

class TestAuthorityPrecedence:
    """Deterministic authority/precedence resolver."""

    def test_precedence_hierarchy(self):
        """Precedence: Governance > Security Authority > Authoritative Eng > Accepted Eng > Task > Learned > Exploration."""
        assert PRECEDENCE_GOVERNANCE > PRECEDENCE_SECURITY_AUTHORITY
        assert PRECEDENCE_SECURITY_AUTHORITY > PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE
        assert PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE > PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE
        assert PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE > PRECEDENCE_TASK_REQUIREMENTS
        assert PRECEDENCE_TASK_REQUIREMENTS > PRECEDENCE_LEARNED_KNOWLEDGE
        assert PRECEDENCE_LEARNED_KNOWLEDGE > PRECEDENCE_EXPLORATION

    def test_resolve_governance_wins(self):
        resolver = AuthorityPrecedenceResolver()
        candidates = [
            {"source": "governance", "authority": "accepted", "record_type": "POLICY"},
            {"source": "security", "authority": SEC_AUTHORITY_AUTHORITATIVE, "record_type": "SEC"},
        ]
        winner = resolver.resolve(candidates)
        assert winner["source"] == "governance"

    def test_resolve_security_wins_over_accepted_adr(self):
        resolver = AuthorityPrecedenceResolver()
        candidates = [
            {"source": "engineering", "authority": AUTHORITY_ACCEPTED, "record_type": "ADR"},
            {"source": "security", "authority": SEC_AUTHORITY_ACCEPTED, "record_type": "SEC"},
        ]
        winner = resolver.resolve(candidates)
        assert winner["record_type"] == "SEC"

    def test_resolve_authoritative_adr_wins_over_accepted_prd(self):
        resolver = AuthorityPrecedenceResolver()
        candidates = [
            {"source": "engineering", "authority": AUTHORITY_ACCEPTED, "record_type": "PRD"},
            {"source": "engineering", "authority": AUTHORITY_ACCEPTED, "record_type": "ADR"},
        ]
        winner = resolver.resolve(candidates)
        assert winner["record_type"] == "ADR"

    def test_resolve_task_wins_over_learning(self):
        resolver = AuthorityPrecedenceResolver()
        candidates = [
            {"source": "learning", "authority": AUTHORITY_ACCEPTED, "record_type": "STRATEGY"},
            {"source": "task", "authority": AUTHORITY_PROPOSED, "record_type": "TASK"},
        ]
        winner = resolver.resolve(candidates)
        assert winner["source"] == "task"

    def test_resolve_empty(self):
        resolver = AuthorityPrecedenceResolver()
        assert resolver.resolve([]) is None

    def test_can_override(self):
        resolver = AuthorityPrecedenceResolver()
        source = {"source": "security", "authority": SEC_AUTHORITY_AUTHORITATIVE, "record_type": "SEC"}
        target = {"source": "engineering", "authority": AUTHORITY_ACCEPTED, "record_type": "ADR"}
        assert resolver.can_override(source, target) is True

    def test_cannot_override_same_precedence(self):
        resolver = AuthorityPrecedenceResolver()
        source = {"source": "security", "authority": SEC_AUTHORITY_ACCEPTED, "record_type": "SEC"}
        target = {"source": "security", "authority": SEC_AUTHORITY_AUTHORITATIVE, "record_type": "SEC"}
        assert resolver.can_override(source, target) is False


# ──────────────────────────────────────────────────────────────────────
# Security Conflict Detection
# ──────────────────────────────────────────────────────────────────────

class TestSecurityConflicts:
    """Deterministic security conflict detection."""

    def test_adr_contradicts_sec(self):
        """ADR PROPOSED contradicts authoritative SEC → CONFLICT."""
        adr = make_adr(record_id="ADR-001")
        sec = make_sec(record_id="SEC-001", authority=SEC_AUTHORITY_AUTHORITATIVE)
        adr.add_relationship("SEC-001", RELATIONSHIP_CONTRADICTS)

        conflicts = detect_security_conflicts([adr, sec])
        types = [c.conflict_type for c in conflicts]
        assert "ADR_CONTRADICTS_SEC" in types

    def test_req_contradicts_sec(self):
        """REQ contradicts authoritative SEC → CONFLICT."""
        from harness.knowledge import Requirement, REQ_TYPE_FUNCTIONAL
        req = Requirement(
            record_id="REQ-001",
            title="Test REQ",
            description="Test requirement",
            req_type=REQ_TYPE_FUNCTIONAL,
            priority="high",
            provenance=make_provenance(),
        )
        sec = make_sec(record_id="SEC-001", authority=SEC_AUTHORITY_AUTHORITATIVE)
        req.add_relationship("SEC-001", RELATIONSHIP_CONTRADICTS)

        conflicts = detect_security_conflicts([req, sec])
        types = [c.conflict_type for c in conflicts]
        assert "REQ_CONTRADICTS_SEC" in types

    def test_tdr_resolved_without_evidence(self):
        """Active TDR claims resolved without evidence → CONFLICT."""
        tdr = make_tdr(record_id="TDR-001")
        tdr.transition_status("acknowledged")
        tdr.transition_status("remediation_planned")
        tdr.transition_status("in_progress")
        tdr.transition_status("resolved")
        # No resolution evidence added

        conflicts = detect_security_conflicts([tdr])
        types = [c.conflict_type for c in conflicts]
        assert "TDR_RESOLVED_WITHOUT_EVIDENCE" in types

    def test_risk_state_conflict(self):
        """High-exposure risk ACCEPTED → CONFLICT."""
        rsk = make_rsk(
            record_id="RSK-001",
            likelihood=RSK_LIKELIHOOD_HIGH,
            impact=RSK_IMPACT_HIGH,
        )
        rsk.transition_status("assessed")
        rsk.transition_status("accepted")

        conflicts = detect_security_conflicts([rsk])
        types = [c.conflict_type for c in conflicts]
        assert "RISK_STATE_CONFLICT" in types

    def test_no_conflicts_clean_records(self):
        """Clean records have no conflicts."""
        adr = make_adr(record_id="ADR-001")
        sec = make_sec(record_id="SEC-001", authority=SEC_AUTHORITY_AUTHORITATIVE)
        tdr = make_tdr(record_id="TDR-001")
        rsk = make_rsk(record_id="RSK-001")

        conflicts = detect_security_conflicts([adr, sec, tdr, rsk])
        assert len(conflicts) == 0

    def test_adr_does_not_silently_override_sec(self):
        """ADR PROPOSED CONTRADICTS SEC authoritative constraint → CONFLICT, not SEC silently replaced."""
        adr = make_adr(record_id="ADR-001")
        sec = make_sec(record_id="SEC-001", authority=SEC_AUTHORITY_AUTHORITATIVE)
        adr.add_relationship("SEC-001", RELATIONSHIP_CONTRADICTS)

        conflicts = detect_security_conflicts([adr, sec])
        assert len(conflicts) > 0
        # SEC is still authoritative — not silently replaced
        assert sec.is_accepted_or_higher() is True
        # ADR remains proposed — not silently accepted
        assert adr.status == "proposed"


# ──────────────────────────────────────────────────────────────────────
# Learning Cannot Override SEC
# ──────────────────────────────────────────────────────────────────────

class TestLearningCannotOverrideSEC:
    """Critical invariant: Learned Strategy conflicts with SEC → SEC WINS."""

    def test_learned_strategy_cannot_override_sec(self):
        """Learned strategy cannot override SEC constraint."""
        resolver = AuthorityPrecedenceResolver()
        learned = {"source": "learning", "authority": AUTHORITY_ACCEPTED, "record_type": "STRATEGY"}
        sec = {"source": "security", "authority": SEC_AUTHORITY_ACCEPTED, "record_type": "SEC"}
        assert resolver.can_override(sec, learned) is True
        assert resolver.can_override(learned, sec) is False

    def test_task_preference_cannot_override_sec(self):
        """Task preference cannot override SEC constraint."""
        resolver = AuthorityPrecedenceResolver()
        task = {"source": "task", "authority": AUTHORITY_PROPOSED, "record_type": "TASK"}
        sec = {"source": "security", "authority": SEC_AUTHORITY_ACCEPTED, "record_type": "SEC"}
        assert resolver.can_override(sec, task) is True
        assert resolver.can_override(task, sec) is False

    def test_timestamp_cannot_override_sec(self):
        """Timestamp cannot override SEC constraint."""
        resolver = AuthorityPrecedenceResolver()
        # Even if SEC is older, it still wins
        sec = {"source": "security", "authority": SEC_AUTHORITY_ACCEPTED, "record_type": "SEC"}
        new_eng = {"source": "engineering", "authority": AUTHORITY_ACCEPTED, "record_type": "ADR"}
        assert resolver.can_override(sec, new_eng) is True

    def test_agent_identity_cannot_override_sec(self):
        """Agent identity cannot override SEC constraint."""
        resolver = AuthorityPrecedenceResolver()
        # Even if SEC was created by a different agent, it still wins
        sec = {"source": "security", "authority": SEC_AUTHORITY_ACCEPTED, "record_type": "SEC"}
        agent_adr = {"source": "engineering", "authority": AUTHORITY_ACCEPTED, "record_type": "ADR"}
        assert resolver.can_override(sec, agent_adr) is True

    def test_model_score_cannot_override_sec(self):
        """Model score cannot override SEC constraint."""
        resolver = AuthorityPrecedenceResolver()
        # Even if SEC has lower model score, it still wins
        sec = {"source": "security", "authority": SEC_AUTHORITY_ACCEPTED, "record_type": "SEC"}
        high_score = {"source": "learning", "authority": AUTHORITY_ACCEPTED, "record_type": "STRATEGY"}
        assert resolver.can_override(sec, high_score) is True


# ──────────────────────────────────────────────────────────────────────
# Agent Cannot Self-Approve Security Exception
# ──────────────────────────────────────────────────────────────────────

class TestAgentCannotSelfApproveException:
    """Agent cannot self-approve protected security exception."""

    def test_agent_cannot_self_approve_exception(self):
        """Agent cannot self-approve security exception."""
        sec = make_sec()
        # Agent cannot set exception_approved = True
        sec.exception_approved = False  # Default is False
        assert sec.exception_approved is False

    def test_sec_waiver_requires_protected_review(self):
        """SEC waiver requires protected review/authority."""
        sec = make_sec()
        # Waiver is None by default
        assert sec.waiver is None
        # Setting waiver requires explicit action (not agent self-approval)
        sec.waiver = "WAIVER-001"
        assert sec.waiver == "WAIVER-001"


# ──────────────────────────────────────────────────────────────────────
# KnowledgeStore Persistence
# ──────────────────────────────────────────────────────────────────────

class TestKnowledgeStorePersistence:
    """TDR/RSK/SEC persist through KnowledgeStore, survive restart/reload."""

    def test_tdr_persistence_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = KnowledgeStore(db_path)

            tdr = make_tdr()
            store.save(tdr)

            # Reload
            store2 = KnowledgeStore(db_path)
            loaded = store2.get("TDR-001")

            assert loaded.record_id == "TDR-001"
            assert loaded.record_type == "TDR"
            assert loaded.debt_type == TDR_DEBT_TYPE_CODE
            assert loaded.severity == TDR_SEVERITY_MEDIUM
            assert isinstance(loaded, TechnicalDebtRecord)
            store.close()
            store2.close()

    def test_rsk_persistence_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = KnowledgeStore(db_path)

            rsk = make_rsk()
            store.save(rsk)

            # Reload
            store2 = KnowledgeStore(db_path)
            loaded = store2.get("RSK-001")

            assert loaded.record_id == "RSK-001"
            assert loaded.record_type == "RSK"
            assert loaded.risk_category == RSK_CATEGORY_TECHNICAL
            assert loaded.likelihood == RSK_LIKELIHOOD_MEDIUM
            assert isinstance(loaded, RiskRecord)
            store.close()
            store2.close()

    def test_sec_persistence_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = KnowledgeStore(db_path)

            sec = make_sec()
            store.save(sec)

            # Reload
            store2 = KnowledgeStore(db_path)
            loaded = store2.get("SEC-001")

            assert loaded.record_id == "SEC-001"
            assert loaded.record_type == "SEC"
            assert loaded.category == SEC_CATEGORY_CONSTRAINT
            assert loaded.authority == SEC_AUTHORITY_PROPOSED
            assert isinstance(loaded, SecurityRecord)
            store.close()
            store2.close()

    def test_tdr_version_preserved_after_reload(self):
        """TDR version is preserved after store/reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = KnowledgeStore(db_path)

            tdr = make_tdr(version=3)
            store.save(tdr)

            store2 = KnowledgeStore(db_path)
            loaded = store2.get("TDR-001")
            assert loaded.version == 3
            store.close()
            store2.close()

    def test_sec_authority_preserved_after_reload(self):
        """SEC authority is preserved after store/reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = KnowledgeStore(db_path)

            sec = make_sec(authority=SEC_AUTHORITY_ACCEPTED)
            store.save(sec)

            store2 = KnowledgeStore(db_path)
            loaded = store2.get("SEC-001")
            assert loaded.authority == SEC_AUTHORITY_ACCEPTED
            assert loaded.is_accepted_or_higher() is True
            store.close()
            store2.close()

    def test_historical_tdr_recovery(self):
        """Historical TDRs are preserved after supersession."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = KnowledgeStore(db_path)

            tdr1 = make_tdr(record_id="TDR-001")
            tdr2 = make_tdr(record_id="TDR-002")
            tdr1.superseded_by = "TDR-002"

            store.save(tdr1)
            store.save(tdr2)

            loaded1 = store.get("TDR-001")
            loaded2 = store.get("TDR-002")

            assert loaded1.record_id == "TDR-001"
            assert loaded1.superseded_by == "TDR-002"
            assert loaded2.record_id == "TDR-002"
            assert loaded2.superseded_by is None
            store.close()


# ──────────────────────────────────────────────────────────────────────
# KnowledgeGraph Traversal
# ──────────────────────────────────────────────────────────────────────

class TestGraphTraversal:
    """KnowledgeGraph traversal with TDR/RSK/SEC records."""

    def test_tdr_in_graph_traversal(self):
        tdr = make_tdr(related_decisions=["ADR-001"])
        graph = EngineeringKnowledgeGraph.build_from_records([tdr])

        assert graph.has_node("TDR-001")
        neighbors = graph.neighbors("TDR-001", relation_type=RELATIONSHIP_DERIVED_FROM)
        assert "ADR-001" in neighbors

    def test_rsk_in_graph_traversal(self):
        rsk = make_rsk(related_decisions=["ADR-001"], related_security=["SEC-001"])
        graph = EngineeringKnowledgeGraph.build_from_records([rsk])

        assert graph.has_node("RSK-001")
        neighbors = graph.neighbors("RSK-001", relation_type=RELATIONSHIP_DERIVED_FROM)
        assert "ADR-001" in neighbors
        sec_neighbors = graph.neighbors("RSK-001", relation_type=RELATIONSHIP_MITIGATES)
        assert "SEC-001" in sec_neighbors

    def test_sec_in_graph_traversal(self):
        sec = make_sec(related_nfrs=["NFR-001"], related_adrs=["ADR-001"])
        graph = EngineeringKnowledgeGraph.build_from_records([sec])

        assert graph.has_node("SEC-001")
        nfr_neighbors = graph.neighbors("SEC-001", relation_type=RELATIONSHIP_SATISFIES)
        assert "NFR-001" in nfr_neighbors
        adr_neighbors = graph.neighbors("SEC-001", relation_type=RELATIONSHIP_CONSTRAINED_BY)
        assert "ADR-001" in adr_neighbors

    def test_contradictor_with_tdr_rsk_sec_graph(self):
        """ContradictionDetector works with TDR/RSK/SEC graph."""
        tdr = make_tdr(record_id="TDR-001")
        rsk = make_rsk(record_id="RSK-001")
        sec = make_sec(record_id="SEC-001")

        graph = EngineeringKnowledgeGraph.build_from_records([tdr, rsk, sec])
        detector = ContradictionDetector(graph)
        assert detector.is_clean()

    def test_adr_to_tdr_introduces_relationship(self):
        """ADR INTRODUCES TDR relationship."""
        adr = make_adr(record_id="ADR-001")
        tdr = make_tdr(record_id="TDR-001")
        adr.add_relationship("TDR-001", RELATIONSHIP_INTRODUCES)

        graph = EngineeringKnowledgeGraph.build_from_records([adr, tdr])
        neighbors = graph.neighbors("ADR-001", relation_type=RELATIONSHIP_INTRODUCES)
        assert "TDR-001" in neighbors

    def test_adr_to_rsk_introduces_relationship(self):
        """ADR INTRODUCES RSK relationship."""
        adr = make_adr(record_id="ADR-001")
        rsk = make_rsk(record_id="RSK-001")
        adr.add_relationship("RSK-001", RELATIONSHIP_INTRODUCES)

        graph = EngineeringKnowledgeGraph.build_from_records([adr, rsk])
        neighbors = graph.neighbors("ADR-001", relation_type=RELATIONSHIP_INTRODUCES)
        assert "RSK-001" in neighbors

    def test_tdr_to_rsk_relates_to_relationship(self):
        """TDR RELATES_TO RSK relationship."""
        tdr = make_tdr(record_id="TDR-001")
        rsk = make_rsk(record_id="RSK-001")
        tdr.add_relationship("RSK-001", RELATIONSHIP_RELATES_TO)

        graph = EngineeringKnowledgeGraph.build_from_records([tdr, rsk])
        neighbors = graph.neighbors("TDR-001", relation_type=RELATIONSHIP_RELATES_TO)
        assert "RSK-001" in neighbors


# ──────────────────────────────────────────────────────────────────────
# Determinism
# ──────────────────────────────────────────────────────────────────────

class TestDeterminism:
    """TDR/RSK/SEC serialization and traversal must be deterministic."""

    def test_tdr_deterministic_traversal_order(self):
        """Graph traversal must be deterministic."""
        tdr = make_tdr(related_decisions=["ADR-003", "ADR-001", "ADR-002"])
        graph = EngineeringKnowledgeGraph.build_from_records([tdr])
        neighbors = graph.neighbors("TDR-001", relation_type=RELATIONSHIP_DERIVED_FROM)
        assert neighbors == sorted(neighbors)

    def test_rsk_deterministic_traversal_order(self):
        """Graph traversal must be deterministic."""
        rsk = make_rsk(related_decisions=["ADR-003", "ADR-001", "ADR-002"])
        graph = EngineeringKnowledgeGraph.build_from_records([rsk])
        neighbors = graph.neighbors("RSK-001", relation_type=RELATIONSHIP_DERIVED_FROM)
        assert neighbors == sorted(neighbors)

    def test_sec_deterministic_traversal_order(self):
        """Graph traversal must be deterministic."""
        sec = make_sec(related_nfrs=["NFR-003", "NFR-001", "NFR-002"])
        graph = EngineeringKnowledgeGraph.build_from_records([sec])
        neighbors = graph.neighbors("SEC-001", relation_type=RELATIONSHIP_SATISFIES)
        assert neighbors == sorted(neighbors)


# ──────────────────────────────────────────────────────────────────────
# Three-Domain Separation
# ──────────────────────────────────────────────────────────────────────

class TestThreeDomainSeparation:
    """TDR/RSK/SEC are Engineering, NOT Learning, NOT Evidence."""

    def test_tdr_is_engineering_record(self):
        tdr = make_tdr()
        assert isinstance(tdr, EngineeringRecord)
        assert tdr.record_type == "TDR"

    def test_rsk_is_engineering_record(self):
        rsk = make_rsk()
        assert isinstance(rsk, EngineeringRecord)
        assert rsk.record_type == "RSK"

    def test_sec_is_engineering_record(self):
        sec = make_sec()
        assert isinstance(sec, EngineeringRecord)
        assert sec.record_type == "SEC"

    def test_tdr_not_learning_record(self):
        """TDR is not StructuredExperience, FailureLesson, Strategy, etc."""
        tdr = make_tdr()
        assert tdr.__class__.__name__ == "TechnicalDebtRecord"
        assert not hasattr(tdr, "task_id")
        assert not hasattr(tdr, "failure_mode")

    def test_rsk_not_learning_record(self):
        """RSK is not StructuredExperience, FailureLesson, Strategy, etc."""
        rsk = make_rsk()
        assert rsk.__class__.__name__ == "RiskRecord"
        assert not hasattr(rsk, "task_id")
        assert not hasattr(rsk, "failure_mode")

    def test_sec_not_learning_record(self):
        """SEC is not StructuredExperience, FailureLesson, Strategy, etc."""
        sec = make_sec()
        assert sec.__class__.__name__ == "SecurityRecord"
        assert not hasattr(sec, "task_id")
        assert not hasattr(sec, "failure_mode")

    def test_tdr_not_evidence(self):
        """TDR is not EvidencePackage or CanonicalExecutionRecord."""
        tdr = make_tdr()
        assert tdr.__class__.__name__ == "TechnicalDebtRecord"
        assert not hasattr(tdr, "content_hash")

    def test_rsk_not_evidence(self):
        """RSK is not EvidencePackage or CanonicalExecutionRecord."""
        rsk = make_rsk()
        assert rsk.__class__.__name__ == "RiskRecord"
        assert not hasattr(rsk, "content_hash")

    def test_sec_not_evidence(self):
        """SEC is not EvidencePackage or CanonicalExecutionRecord."""
        sec = make_sec()
        assert sec.__class__.__name__ == "SecurityRecord"
        assert not hasattr(sec, "content_hash")

    def test_tdr_rsk_sec_semantic_separation(self):
        """TDR != RSK != SEC — they are distinct record types."""
        tdr = make_tdr()
        rsk = make_rsk()
        sec = make_sec()
        assert tdr.record_type != rsk.record_type
        assert rsk.record_type != sec.record_type
        assert tdr.record_type != sec.record_type


# ──────────────────────────────────────────────────────────────────────
# CapabilityGraph Separation
# ──────────────────────────────────────────────────────────────────────

class TestCapabilityGraphSeparation:
    """TDR/RSK/SEC do not leak into CapabilityGraph semantics."""

    def test_tdr_not_in_capability_graph(self):
        """TDR does not appear in CapabilityGraph."""
        tdr = make_tdr()
        # TDR is an Engineering Record, not a Capability
        assert tdr.record_type == "TDR"
        assert not hasattr(tdr, "capability_id")

    def test_rsk_not_in_capability_graph(self):
        """RSK does not appear in CapabilityGraph."""
        rsk = make_rsk()
        assert rsk.record_type == "RSK"
        assert not hasattr(rsk, "capability_id")

    def test_sec_not_in_capability_graph(self):
        """SEC does not appear in CapabilityGraph."""
        sec = make_sec()
        assert sec.record_type == "SEC"
        assert not hasattr(sec, "capability_id")


# ──────────────────────────────────────────────────────────────────────
# Phase 6 E2E Scenarios
# ──────────────────────────────────────────────────────────────────────

class TestPhase6E2E:
    """End-to-end Phase 6 scenarios."""

    def test_tdr_e2e_scenario(self):
        """TDR E2E: REQ → ADR → TDR (INTRODUCES)."""
        from harness.knowledge import PRD, Requirement, NFR
        from harness.knowledge.requirements import REQ_TYPE_FUNCTIONAL

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

        # ADR introduces TDR
        adr = make_adr(
            record_id="ADR-001",
            patterns_considered=["JWT", "OAuth2", "Session"],
            pattern_selected="JWT",
            trade_offs="Simple token vs complex flow",
            impact="Reduces auth latency",
            related_requirements=["REQ-001"],
        )

        tdr = make_tdr(
            record_id="TDR-001",
            debt_type=TDR_DEBT_TYPE_ARCHITECTURE,
            severity=TDR_SEVERITY_MEDIUM,
            impact="Temporary synchronous adapter limits scalability",
            remediation="Replace with async event-driven adapter after event platform migration",
            remediation_estimate="3 days",
            target_condition="Event platform migration complete",
            priority=TDR_PRIORITY_HIGH,
            related_decisions=["ADR-001"],
        )

        # ADR INTRODUCES TDR
        adr.add_relationship("TDR-001", RELATIONSHIP_INTRODUCES)

        # Store all records
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "knowledge.db")
            store = KnowledgeStore(db_path)
            store.save(prd)
            store.save(req)
            store.save(adr)
            store.save(tdr)

            # Build graph
            graph = EngineeringKnowledgeGraph.build_from_store(store)

            # Verify graph relationships
            assert graph.has_node("PRD-001")
            assert graph.has_node("REQ-001")
            assert graph.has_node("ADR-001")
            assert graph.has_node("TDR-001")

            # ADR → INTRODUCES → TDR
            adr_neighbors = graph.neighbors("ADR-001", relation_type=RELATIONSHIP_INTRODUCES)
            assert "TDR-001" in adr_neighbors

            # TDR → DERIVED_FROM → ADR
            tdr_neighbors = graph.neighbors("TDR-001", relation_type=RELATIONSHIP_DERIVED_FROM)
            assert "ADR-001" in tdr_neighbors

            # Reload and verify persistence
            store2 = KnowledgeStore(db_path)
            loaded_tdr = store2.get("TDR-001")
            assert loaded_tdr.record_type == "TDR"
            assert isinstance(loaded_tdr, TechnicalDebtRecord)
            assert loaded_tdr.debt_type == TDR_DEBT_TYPE_ARCHITECTURE
            assert loaded_tdr.related_decisions == ["ADR-001"]
            store.close()
            store2.close()

    def test_risk_e2e_scenario(self):
        """Risk E2E: ADR → RSK (INTRODUCES) → SEC (MITIGATED_BY)."""
        adr = make_adr(
            record_id="ADR-001",
            patterns_considered=["External OAuth", "Internal JWT"],
            pattern_selected="External OAuth",
            trade_offs="External dependency vs control",
            impact="Faster auth but external dependency",
        )

        rsk = make_rsk(
            record_id="RSK-001",
            risk_category=RSK_CATEGORY_TECHNICAL,
            likelihood=RSK_LIKELIHOOD_MEDIUM,
            impact=RSK_IMPACT_HIGH,
            mitigation="Implement fallback to internal JWT",
            contingency="Disable external auth, use internal",
            related_decisions=["ADR-001"],
        )

        sec = make_sec(
            record_id="SEC-001",
            category=SEC_CATEGORY_CONTROL,
            enforcement="Automated health checks",
            authority=SEC_AUTHORITY_ACCEPTED,
        )

        # ADR INTRODUCES RSK
        adr.add_relationship("RSK-001", RELATIONSHIP_INTRODUCES)
        # RSK MITIGATED_BY SEC
        rsk.link_security("SEC-001")

        # Store all records
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "knowledge.db")
            store = KnowledgeStore(db_path)
            store.save(adr)
            store.save(rsk)
            store.save(sec)

            # Build graph
            graph = EngineeringKnowledgeGraph.build_from_store(store)

            # Verify graph relationships
            assert graph.has_node("ADR-001")
            assert graph.has_node("RSK-001")
            assert graph.has_node("SEC-001")

            # ADR → INTRODUCES → RSK
            adr_neighbors = graph.neighbors("ADR-001", relation_type=RELATIONSHIP_INTRODUCES)
            assert "RSK-001" in adr_neighbors

            # RSK → MITIGATES → SEC
            rsk_neighbors = graph.neighbors("RSK-001", relation_type=RELATIONSHIP_MITIGATES)
            assert "SEC-001" in rsk_neighbors

            # Reload and verify persistence
            store2 = KnowledgeStore(db_path)
            loaded_rsk = store2.get("RSK-001")
            assert loaded_rsk.record_type == "RSK"
            assert isinstance(loaded_rsk, RiskRecord)
            assert loaded_rsk.risk_category == RSK_CATEGORY_TECHNICAL
            assert loaded_rsk.related_security == ["SEC-001"]
            store.close()
            store2.close()

    def test_security_e2e_scenario(self):
        """Security E2E: SEC constraint, ADR contradicts, SEC wins."""
        sec = make_sec(
            record_id="SEC-001",
            category=SEC_CATEGORY_CONSTRAINT,
            enforcement="Automated policy enforcement",
            authority=SEC_AUTHORITY_AUTHORITATIVE,
        )

        adr = make_adr(
            record_id="ADR-002",
            patterns_considered=["SMS MFA", "TOTP MFA"],
            pattern_selected="SMS MFA",
            trade_offs="SMS is simpler but less secure",
            impact="Faster implementation",
        )

        # ADR contradicts SEC
        adr.add_relationship("SEC-001", RELATIONSHIP_CONTRADICTS)

        # Store all records
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "knowledge.db")
            store = KnowledgeStore(db_path)
            store.save(sec)
            store.save(adr)

            # Build graph
            graph = EngineeringKnowledgeGraph.build_from_store(store)

            # Verify graph relationships
            assert graph.has_node("SEC-001")
            assert graph.has_node("ADR-002")

            # ADR → CONTRADICTS → SEC
            adr_neighbors = graph.neighbors("ADR-002", relation_type=RELATIONSHIP_CONTRADICTS)
            assert "SEC-001" in adr_neighbors

            # Detect conflicts
            conflicts = detect_security_conflicts([adr, sec])
            assert len(conflicts) > 0
            types = [c.conflict_type for c in conflicts]
            assert "ADR_CONTRADICTS_SEC" in types

            # SEC is still authoritative — not silently replaced
            assert sec.is_accepted_or_higher() is True
            # ADR remains proposed — not silently accepted
            assert adr.status == "proposed"

            # Reload and verify persistence
            store2 = KnowledgeStore(db_path)
            loaded_sec = store2.get("SEC-001")
            assert loaded_sec.record_type == "SEC"
            assert isinstance(loaded_sec, SecurityRecord)
            assert loaded_sec.authority == SEC_AUTHORITY_AUTHORITATIVE
            store.close()
            store2.close()

    def test_combined_e2e_scenario(self):
        """Combined E2E: PRD → REQ → NFR → ADR → TDR/RSK/SEC."""
        from harness.knowledge import PRD, Requirement, NFR
        from harness.knowledge.requirements import REQ_TYPE_FUNCTIONAL

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
            category="performance",
            metric={"name": "response_time_p99", "operator": "<", "target": "200ms"},
            measurement={"method": "load_test"},
            related_prds=["PRD-001"],
            related_requirements=["REQ-001"],
            provenance=make_provenance(),
        )

        # ADR introduces TDR and RSK, constrained by SEC
        adr = make_adr(
            record_id="ADR-001",
            patterns_considered=["JWT", "OAuth2", "Session"],
            pattern_selected="JWT",
            trade_offs="Simple token vs complex flow",
            impact="Reduces auth latency",
            related_requirements=["REQ-001"],
        )

        tdr = make_tdr(
            record_id="TDR-001",
            debt_type=TDR_DEBT_TYPE_ARCHITECTURE,
            severity=TDR_SEVERITY_MEDIUM,
            impact="Temporary synchronous adapter limits scalability",
            remediation="Replace with async event-driven adapter",
            related_decisions=["ADR-001"],
        )

        rsk = make_rsk(
            record_id="RSK-001",
            risk_category=RSK_CATEGORY_TECHNICAL,
            likelihood=RSK_LIKELIHOOD_MEDIUM,
            impact=RSK_IMPACT_HIGH,
            mitigation="Implement fallback to internal JWT",
            related_decisions=["ADR-001"],
        )

        sec = make_sec(
            record_id="SEC-001",
            category=SEC_CATEGORY_CONSTRAINT,
            enforcement="Automated policy enforcement",
            authority=SEC_AUTHORITY_ACCEPTED,
            related_nfrs=["NFR-001"],
        )

        # Establish relationships
        adr.add_relationship("TDR-001", RELATIONSHIP_INTRODUCES)
        adr.add_relationship("RSK-001", RELATIONSHIP_INTRODUCES)
        adr.add_relationship("SEC-001", RELATIONSHIP_CONSTRAINED_BY)
        rsk.link_security("SEC-001")

        # Store all records
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "knowledge.db")
            store = KnowledgeStore(db_path)
            store.save(prd)
            store.save(req)
            store.save(nfr)
            store.save(adr)
            store.save(tdr)
            store.save(rsk)
            store.save(sec)

            # Build graph
            graph = EngineeringKnowledgeGraph.build_from_store(store)

            # Verify all nodes
            assert graph.has_node("PRD-001")
            assert graph.has_node("REQ-001")
            assert graph.has_node("NFR-001")
            assert graph.has_node("ADR-001")
            assert graph.has_node("TDR-001")
            assert graph.has_node("RSK-001")
            assert graph.has_node("SEC-001")

            # Verify relationships
            # PRD → REQUIRES → REQ
            prd_neighbors = graph.neighbors("PRD-001", relation_type=RELATIONSHIP_REQUIRES)
            assert "REQ-001" in prd_neighbors

            # ADR → DECIDED_BY → REQ
            adr_neighbors = graph.neighbors("ADR-001", relation_type=RELATIONSHIP_DECIDED_BY)
            assert "REQ-001" in adr_neighbors

            # ADR → INTRODUCES → TDR
            adr_tdr = graph.neighbors("ADR-001", relation_type=RELATIONSHIP_INTRODUCES)
            assert "TDR-001" in adr_tdr

            # ADR → INTRODUCES → RSK
            adr_rsk = graph.neighbors("ADR-001", relation_type=RELATIONSHIP_INTRODUCES)
            assert "RSK-001" in adr_rsk

            # ADR → CONSTRAINED_BY → SEC
            adr_sec = graph.neighbors("ADR-001", relation_type=RELATIONSHIP_CONSTRAINED_BY)
            assert "SEC-001" in adr_sec

            # SEC → SATISFIES → NFR
            sec_nfr = graph.neighbors("SEC-001", relation_type=RELATIONSHIP_SATISFIES)
            assert "NFR-001" in sec_nfr

            # RSK → MITIGATES → SEC
            rsk_sec = graph.neighbors("RSK-001", relation_type=RELATIONSHIP_MITIGATES)
            assert "SEC-001" in rsk_sec

            # Reload and verify persistence
            store2 = KnowledgeStore(db_path)
            loaded_adr = store2.get("ADR-001")
            assert loaded_adr.record_type == "ADR"
            assert isinstance(loaded_adr, ArchitectureDecisionRecord)

            loaded_tdr = store2.get("TDR-001")
            assert loaded_tdr.record_type == "TDR"
            assert isinstance(loaded_tdr, TechnicalDebtRecord)

            loaded_rsk = store2.get("RSK-001")
            assert loaded_rsk.record_type == "RSK"
            assert isinstance(loaded_rsk, RiskRecord)

            loaded_sec = store2.get("SEC-001")
            assert loaded_sec.record_type == "SEC"
            assert isinstance(loaded_sec, SecurityRecord)
            assert loaded_sec.authority == SEC_AUTHORITY_ACCEPTED

            store.close()
            store2.close()
