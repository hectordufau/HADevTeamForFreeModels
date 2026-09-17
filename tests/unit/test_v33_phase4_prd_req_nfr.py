# tests/unit/test_v33_phase4_prd_req_nfr.py — Phase 4 Acceptance Tests
"""Phase 4: PRD / Requirements / NFR — deterministic behavior verification.

Tests cover:
- PRD creation, validation, completeness, versioning
- Requirement creation, validation, lifecycle, versioning, stable identity
- Acceptance criterion creation, identity, validation
- NFR creation, category, numeric metric, qualitative, validation, versioning
- PRD → REQ relationship, REQ → NFR relationship
- Orphan requirement detection, global NFR behavior
- Requirement completeness, NFR completeness
- Coverage metrics, structured requirement conflict, structured NFR conflict
- KnowledgeStore persistence, restart/reload
- Graph rebuild, graph traversal, inverse traversal
- Authority preservation, provenance preservation
- Historical version recovery, deterministic serialization
- Engineering/Learning separation, Engineering/Evidence separation
- Phase 4 E2E scenario: PRD-001 → REQ-001 → AC-001/AC-002, NFR-001/NFR-002
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.knowledge import (
    PRD,
    Requirement,
    NFR,
    AcceptanceCriterion,
    KnowledgeStore,
    EngineeringKnowledgeGraph,
    Provenance,
    AUTHORITY_PROPOSED,
    AUTHORITY_ACCEPTED,
    AUTHORITY_DEPRECATED,
    AUTHORITY_ARCHIVED,
    VALID_AUTHORITY_LEVELS,
    VALID_REQ_TYPES,
    VALID_NFR_CATEGORIES,
    VALID_NFR_SCOPES,
    NFR_SCOPE_GLOBAL,
    NFR_SCOPE_PRD,
    NFR_SCOPE_REQUIREMENT,
    NFR_CATEGORY_PERFORMANCE,
    NFR_CATEGORY_SECURITY,
    REQ_TYPE_FUNCTIONAL,
    REQ_TYPE_PRODUCT,
    MEASUREMENT_LOAD_TEST,
    compute_prd_coverage,
    find_orphan_requirements,
    find_requirements_for_prd,
    find_nfrs_for_requirement,
    find_requirements_constrained_by_nfr,
    detect_requirement_conflicts,
    detect_nfr_conflicts,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def make_provenance(**kwargs):
    defaults = {"author": "test_agent", "source": "pipeline", "timestamp": "2026-01-01T00:00:00"}
    defaults.update(kwargs)
    return Provenance(**defaults)


def _build_graph(records):
    """Helper: build graph from records using correct API."""
    graph = EngineeringKnowledgeGraph()
    for record in records:
        graph.add_node(record.record_id)
    for record in records:
        for rel in record.get_relationships():
            graph.add_edge(rel.source_id, rel.relation_type, rel.target_id)
    return graph


def make_prd(**kwargs):
    defaults = {
        "record_id": "PRD-001",
        "title": "Identity Platform",
        "description": "Identity and access management platform",
        "objective": "Build a centralized identity platform",
        "scope": "Authentication, authorization, and user management",
        "provenance": make_provenance(),
        "stakeholders": ["product", "engineering", "security"],
    }
    defaults.update(kwargs)
    return PRD(**defaults)


def make_req(**kwargs):
    defaults = {
        "record_id": "REQ-001",
        "title": "Support MFA authentication",
        "description": "The system must support multi-factor authentication",
        "req_type": REQ_TYPE_FUNCTIONAL,
        "priority": "high",
        "provenance": make_provenance(),
        "parent_prd": "PRD-001",
    }
    defaults.update(kwargs)
    return Requirement(**defaults)


def make_nfr(**kwargs):
    defaults = {
        "record_id": "NFR-001",
        "title": "Performance",
        "description": "Response time must be within acceptable bounds",
        "category": NFR_CATEGORY_PERFORMANCE,
        "scope": NFR_SCOPE_GLOBAL,
        "provenance": make_provenance(),
    }
    defaults.update(kwargs)
    return NFR(**defaults)


# ──────────────────────────────────────────────────────────────────────
# 1. PRD Creation, Validation, Completeness
# ──────────────────────────────────────────────────────────────────────

class TestPRDCreation:
    def test_prd_creation(self):
        prd = make_prd()
        assert prd.record_id == "PRD-001"
        assert prd.record_type == "PRD"
        assert prd.objective == "Build a centralized identity platform"
        assert prd.scope == "Authentication, authorization, and user management"
        assert prd.status == "draft"
        assert prd.authority == AUTHORITY_PROPOSED

    def test_prd_requires_objective(self):
        prd = make_prd(objective="")
        # Empty objective is valid at construction level
        # Completeness check flags it
        assert prd.get_completeness()["has_objective"] is False

    def test_prd_requires_scope(self):
        prd = make_prd(scope="")
        # Empty scope is valid at construction level
        # Completeness check flags it
        assert prd.get_completeness()["has_scope"] is False

    def test_prd_requires_valid_req_ids(self):
        with pytest.raises(Exception):
            make_prd(requirements=["INVALID-001"])

    def test_prd_requires_valid_nfr_ids(self):
        with pytest.raises(Exception):
            make_prd(related_nfrs=["INVALID-001"])

    def test_prd_requires_provenance(self):
        with pytest.raises(Exception):
            PRD(
                record_id="PRD-001",
                title="Test",
                description="Test",
                objective="Test",
                scope="Test",
                provenance=None,
            )

    def test_prd_completeness(self):
        prd = make_prd(requirements=["REQ-001"])
        completeness = prd.get_completeness()
        assert completeness["has_objective"] is True
        assert completeness["has_scope"] is True
        assert completeness["has_requirements"] is True
        assert prd.is_complete() is True

    def test_prd_incomplete_when_no_requirements(self):
        prd = make_prd(requirements=[])
        assert prd.is_complete() is False

    def test_prd_incomplete_when_no_objective(self):
        prd = make_prd(objective="")
        assert prd.is_complete() is False


# ──────────────────────────────────────────────────────────────────────
# 2. Requirement Creation, Validation, Lifecycle, Versioning
# ──────────────────────────────────────────────────────────────────────

class TestRequirementCreation:
    def test_req_creation(self):
        req = make_req()
        assert req.record_id == "REQ-001"
        assert req.record_type == "REQ"
        assert req.status == "proposed"
        assert req.authority == AUTHORITY_PROPOSED
        assert req.parent_prd == "PRD-001"

    def test_req_requires_parent_prd(self):
        req = make_req()
        req.parent_prd = None
        assert req.is_complete() is False

    def test_req_requires_provenance(self):
        with pytest.raises(Exception):
            Requirement(
                record_id="REQ-001",
                title="Test",
                description="Test",
                provenance=None,
            )

    def test_req_id_prefix_must_match(self):
        with pytest.raises(Exception):
            Requirement(
                record_id="INVALID-001",
                title="Test",
                description="Test",
                provenance=make_provenance(),
            )

    def test_req_lifecycle_valid_transitions(self):
        req = make_req()
        req.transition_status("accepted")
        assert req.status == "accepted"
        req.transition_status("implemented")
        assert req.status == "implemented"
        req.transition_status("verified")
        assert req.status == "verified"
        req.transition_status("deprecated")
        assert req.status == "deprecated"
        req.transition_status("superseded")
        assert req.status == "superseded"

    def test_req_lifecycle_invalid_transition(self):
        req = make_req()
        with pytest.raises(Exception):
            req.transition_status("verified")  # Can't skip accepted/implemented

    def test_req_lifecycle_terminal_state(self):
        req = make_req()
        req.transition_status("accepted")
        req.transition_status("deprecated")
        req.transition_status("superseded")
        assert req.is_terminal() is True
        with pytest.raises(Exception):
            req.transition_status("accepted")

    def test_req_versioning(self):
        req = make_req()
        assert req.version == 1
        new_req = req.create_new_version("REQ-002")
        assert new_req.version == 2
        assert req.superseded_by == "REQ-002"
        assert new_req.provenance.parent_record == "REQ-001"
        assert new_req.status == "proposed"
        assert new_req.authority == AUTHORITY_PROPOSED

    def test_req_stable_identity_across_versions(self):
        req = make_req()
        original_id = req.record_id
        new_req = req.create_new_version("REQ-002")
        # The logical requirement is still REQ-001 (REQ-002 is the new version)
        assert req.record_id == original_id
        assert new_req.record_id == "REQ-002"
        assert req.superseded_by == "REQ-002"

    def test_req_completeness(self):
        req = make_req()
        req.add_acceptance_criterion("AC-001", "User can log in with MFA")
        assert req.is_complete() is True

    def test_req_incomplete_without_ac(self):
        req = make_req()
        assert req.is_complete() is False

    def test_req_incomplete_without_parent(self):
        req = make_req(parent_prd=None)
        assert req.is_complete() is False


# ──────────────────────────────────────────────────────────────────────
# 3. Acceptance Criterion
# ──────────────────────────────────────────────────────────────────────

class TestAcceptanceCriterion:
    def test_ac_creation(self):
        ac = AcceptanceCriterion(
            id="AC-001",
            description="User can log in with MFA",
            verification_method="integration_test",
        )
        assert ac.id == "AC-001"
        assert ac.status == "pending"
        assert ac.version == 1

    def test_ac_id_validation(self):
        with pytest.raises(Exception):
            AcceptanceCriterion(id="INVALID", description="Test")

    def test_ac_requires_description(self):
        with pytest.raises(Exception):
            AcceptanceCriterion(id="AC-001", description="")

    def test_ac_status_values(self):
        ac = AcceptanceCriterion(id="AC-001", description="Test")
        for status in ["pending", "pass", "fail", "skip"]:
            ac.status = status
            assert ac.status == status

    def test_ac_invalid_status(self):
        ac = AcceptanceCriterion(id="AC-001", description="Test")
        ac.status = "invalid"
        with pytest.raises(Exception):
            ac.validate()

    def test_ac_duplicate_prevention(self):
        req = make_req()
        req.add_acceptance_criterion("AC-001", "Test")
        with pytest.raises(Exception):
            req.add_acceptance_criterion("AC-001", "Duplicate")


# ──────────────────────────────────────────────────────────────────────
# 4. NFR Creation
# ──────────────────────────────────────────────────────────────────────

class TestNFR:
    def test_nfr_creation_qualitative(self):
        nfr = make_nfr()
        assert nfr.record_id == "NFR-001"
        assert nfr.record_type == "NFR"
        assert nfr.category == NFR_CATEGORY_PERFORMANCE
        assert nfr.scope == NFR_SCOPE_GLOBAL
        assert nfr.qualitative is True

    def test_nfr_creation_quantitative(self):
        nfr = make_nfr(
            metric={"name": "p95_latency", "operator": "<=", "target": 200, "unit": "ms"},
            measurement={"method": "load_test"},
            qualitative=False,
        )
        assert nfr.qualitative is False
        assert nfr.metric["name"] == "p95_latency"

    def test_nfr_requires_category(self):
        # NFR category is validated in __post_init__ before super() call
        with pytest.raises(Exception):
            make_nfr(category="")

    def test_nfr_category_valid(self):
        nfr = make_nfr(category="performance")
        assert nfr.category == "performance"

    def test_nfr_requires_scope(self):
        with pytest.raises(Exception):
            make_nfr(scope="")

    def test_nfr_defaults_to_qualitative_when_no_metric(self):
        """When metric is not set, NFR defaults to qualitative."""
        nfr = make_nfr()
        assert nfr.qualitative is True
        assert nfr.metric is None

    def test_nfr_metric_validation(self):
        nfr = make_nfr()
        nfr.add_metric("p95_latency", "<=", 200, "ms")
        assert nfr.metric["operator"] == "<="
        assert nfr.metric["target"] == 200

    def test_nfr_invalid_metric_operator(self):
        nfr = make_nfr()
        with pytest.raises(Exception):
            nfr.add_metric("p95_latency", "invalid", 200)

    def test_nfr_measurement_validation(self):
        nfr = make_nfr()
        nfr.add_measurement("load_test")
        assert nfr.measurement["method"] == "load_test"

    def test_nfr_invalid_measurement(self):
        nfr = make_nfr()
        with pytest.raises(Exception):
            nfr.add_measurement("invalid_method")

    def test_nfr_completeness(self):
        nfr = make_nfr()
        assert nfr.is_complete() is True

    def test_nfr_incomplete_without_category(self):
        # Empty category is rejected at construction time via __post_init__
        with pytest.raises(Exception):
            NFR(
                record_id="NFR-001",
                title="Test",
                description="Test",
                category="",
                provenance=make_provenance(timestamp="2026-01-01T00:00:00"),
            )

    def test_nfr_link_requirement(self):
        nfr = make_nfr()
        nfr.link_requirement("REQ-001")
        assert "REQ-001" in nfr.related_requirements

    def test_nfr_link_invalid_requirement(self):
        nfr = make_nfr()
        with pytest.raises(Exception):
            nfr.link_requirement("INVALID-001")

    def test_nfr_global_scope(self):
        nfr = make_nfr(scope=NFR_SCOPE_GLOBAL)
        assert nfr.scope == NFR_SCOPE_GLOBAL

    def test_nfr_lifecycle(self):
        nfr = make_nfr()
        assert nfr.status == "draft"
        nfr.transition_status("review")
        assert nfr.status == "review"
        nfr.transition_status("accepted")
        assert nfr.status == "accepted"


# ──────────────────────────────────────────────────────────────────────
# 5. PRD → REQ Relationships
# ──────────────────────────────────────────────────────────────────────

class TestPRDRequirementRelationships:
    def test_prd_with_requirements(self):
        prd = make_prd(requirements=["REQ-001", "REQ-002"])
        assert "REQ-001" in prd.requirements
        assert "REQ-002" in prd.requirements

    def test_prd_add_requirement(self):
        prd = make_prd()
        prd.add_requirement("REQ-001")
        assert "REQ-001" in prd.requirements
        rels = prd.get_relationships()
        assert any(r.target_id == "REQ-001" for r in rels)

    def test_prd_link_nfr(self):
        prd = make_prd()
        prd.link_nfr("NFR-001")
        assert "NFR-001" in prd.related_nfrs

    def test_req_parent_prd(self):
        req = make_req(parent_prd="PRD-001")
        assert req.parent_prd == "PRD-001"

    def test_req_invalid_parent_prd(self):
        with pytest.raises(Exception):
            make_req(parent_prd="INVALID-001")


# ──────────────────────────────────────────────────────────────────────
# 6. Requirement → NFR Relationships
# ──────────────────────────────────────────────────────────────────────

class TestRequirementNFRelationships:
    def test_req_link_nfr(self):
        req = make_req()
        req.link_nfr("NFR-001")
        assert "NFR-001" in req.related_nfrs
        rels = req.get_relationships()
        assert any(r.target_id == "NFR-001" for r in rels)

    def test_req_link_multiple_nfrs(self):
        req = make_req()
        req.link_nfr("NFR-001")
        req.link_nfr("NFR-002")
        assert len(req.related_nfrs) == 2

    def test_requirement_constrained_by_nfr(self):
        req = make_req()
        req.constraints = ["NFR-001", "NFR-002"]
        assert "NFR-001" in req.constraints


# ──────────────────────────────────────────────────────────────────────
# 7. Orphan Requirements
# ──────────────────────────────────────────────────────────────────────

class TestOrphanRequirements:
    def test_orphan_detection(self):
        req = make_req(parent_prd=None)
        orphans = find_orphan_requirements({"REQ-001": req}, {})
        assert len(orphans) == 1
        assert orphans[0].record_id == "REQ-001"

    def test_non_orphan_requirement(self):
        prd = make_prd(requirements=["REQ-001"])
        req = make_req(parent_prd="PRD-001")
        orphans = find_orphan_requirements({"REQ-001": req}, {"PRD-001": prd})
        assert len(orphans) == 0

    def test_orphan_with_missing_parent(self):
        req = make_req(parent_prd="PRD-999")
        orphans = find_orphan_requirements({"REQ-001": req}, {})
        assert len(orphans) == 1


# ──────────────────────────────────────────────────────────────────────
# 8. Global NFR Behavior
# ──────────────────────────────────────────────────────────────────────

class TestGlobalNFR:
    def test_global_nfr_not_orphan(self):
        """NFR with no REQ edge is NOT automatically invalid if scope is GLOBAL."""
        nfr = make_nfr(scope=NFR_SCOPE_GLOBAL)
        # An NFR with no linked requirements is valid when scope is GLOBAL
        assert nfr.scope == NFR_SCOPE_GLOBAL
        assert len(nfr.related_requirements) == 0
        # It's still complete (qualitative, has category/scope)
        assert nfr.is_complete() is True

    def test_nfr_scope_validation(self):
        # Scope validation happens at construction time
        with pytest.raises(Exception):
            make_nfr(scope="INVALID_SCOPE")

    def test_nfr_scope_valid(self):
        nfr = make_nfr(scope=NFR_SCOPE_GLOBAL)
        assert nfr.scope == NFR_SCOPE_GLOBAL

    def test_nfr_scope_prd(self):
        nfr = make_nfr(scope=NFR_SCOPE_PRD)
        assert nfr.scope == NFR_SCOPE_PRD

    def test_nfr_scope_requirement(self):
        nfr = make_nfr(scope=NFR_SCOPE_REQUIREMENT)
        assert nfr.scope == NFR_SCOPE_REQUIREMENT


# ──────────────────────────────────────────────────────────────────────
# 9. Coverage Metrics
# ──────────────────────────────────────────────────────────────────────

class TestCoverageMetrics:
    def test_prd_coverage(self):
        prd = make_prd(requirements=["REQ-001", "REQ-002"])
        req1 = make_req(record_id="REQ-001", parent_prd="PRD-001")
        req2 = make_req(record_id="REQ-002", parent_prd="PRD-001")
        req1.add_acceptance_criterion("AC-001", "Test 1")
        req2.add_acceptance_criterion("AC-002", "Test 2")

        coverage = compute_prd_coverage(
            prd,
            {"REQ-001": req1, "REQ-002": req2},
            {},
        )
        assert coverage["requirements_total"] == 2
        assert coverage["requirements_present"] == 2
        assert coverage["requirement_coverage"] == 1.0
        assert coverage["orphan_requirements"] == 0

    def test_coverage_with_missing_req(self):
        prd = make_prd(requirements=["REQ-001", "REQ-999"])
        req1 = make_req(record_id="REQ-001", parent_prd="PRD-001")
        coverage = compute_prd_coverage(prd, {"REQ-001": req1}, {})
        assert coverage["requirements_present"] == 1
        assert coverage["requirements_missing"] == 1
        assert coverage["orphan_requirements"] == 1

    def test_coverage_with_nfrs(self):
        prd = make_prd(requirements=["REQ-001"], related_nfrs=["NFR-001"])
        req1 = make_req(record_id="REQ-001", parent_prd="PRD-001")
        nfr1 = make_nfr(record_id="NFR-001")
        coverage = compute_prd_coverage(prd, {"REQ-001": req1}, {"NFR-001": nfr1})
        assert coverage["nfrs_total"] == 1
        assert coverage["nfrs_present"] == 1
        assert coverage["nfr_link_coverage"] == 1.0

    def test_find_requirements_for_prd(self):
        req1 = make_req(record_id="REQ-001", parent_prd="PRD-001")
        req2 = make_req(record_id="REQ-002", parent_prd="PRD-001")
        req3 = make_req(record_id="REQ-003", parent_prd="PRD-002")
        result = find_requirements_for_prd(
            "PRD-001",
            {"REQ-001": req1, "REQ-002": req2, "REQ-003": req3},
        )
        assert len(result) == 2
        assert all(r.parent_prd == "PRD-001" for r in result)

    def test_find_nfrs_for_requirement(self):
        nfr1 = make_nfr(record_id="NFR-001")
        nfr1.link_requirement("REQ-001")
        nfr2 = make_nfr(record_id="NFR-002")
        nfr2.link_requirement("REQ-001")
        result = find_nfrs_for_requirement("REQ-001", {"NFR-001": nfr1, "NFR-002": nfr2})
        assert len(result) == 2

    def test_find_requirements_constrained_by_nfr(self):
        nfr1 = make_nfr(record_id="NFR-001")
        nfr1.link_requirement("REQ-001")
        req1 = make_req(record_id="REQ-001")
        result = find_requirements_constrained_by_nfr(
            "NFR-001",
            {"NFR-001": nfr1},
            {"REQ-001": req1},
        )
        assert len(result) == 1


# ──────────────────────────────────────────────────────────────────────
# 10. Conflict Detection
# ──────────────────────────────────────────────────────────────────────

class TestConflictDetection:
    def test_requirement_conflict_supersedes_both_active(self):
        req1 = make_req(record_id="REQ-001")
        req1.transition_status("accepted")
        req1.superseded_by = "REQ-002"
        req2 = make_req(record_id="REQ-002")
        req2.transition_status("accepted")
        conflicts = detect_requirement_conflicts({"REQ-001": req1, "REQ-002": req2})
        assert len(conflicts) > 0
        assert any(c["conflict_type"] == "SUPERSESSION_CONFLICT" for c in conflicts)

    def test_no_conflict_normal_supersession(self):
        req1 = make_req(record_id="REQ-001")
        req1.transition_status("deprecated")
        req1.superseded_by = "REQ-002"
        req2 = make_req(record_id="REQ-002")
        req2.transition_status("accepted")
        conflicts = detect_requirement_conflicts({"REQ-001": req1, "REQ-002": req2})
        assert len(conflicts) == 0

    def test_nfr_conflict_metric_contradiction(self):
        nfr1 = make_nfr(
            record_id="NFR-001",
            qualitative=False,
            metric={"name": "p95_latency", "operator": "<=", "target": 200, "unit": "ms"},
        )
        nfr2 = make_nfr(
            record_id="NFR-002",
            qualitative=False,
            metric={"name": "p95_latency", "operator": ">=", "target": 500, "unit": "ms"},
        )
        conflicts = detect_nfr_conflicts({"NFR-001": nfr1, "NFR-002": nfr2})
        assert len(conflicts) > 0
        assert any(c["conflict_type"] == "METRIC_CONTRADICTION" for c in conflicts)

    def test_no_nfr_conflict_different_scope(self):
        nfr1 = make_nfr(
            record_id="NFR-001",
            scope=NFR_SCOPE_PRD,
            qualitative=False,
            metric={"name": "p95_latency", "operator": "<=", "target": 200},
        )
        nfr2 = make_nfr(
            record_id="NFR-002",
            scope="COMPONENT",
            qualitative=False,
            metric={"name": "p95_latency", "operator": ">=", "target": 500},
        )
        conflicts = detect_nfr_conflicts({"NFR-001": nfr1, "NFR-002": nfr2})
        # Different scope NFRs don't conflict
        assert len(conflicts) == 0


# ──────────────────────────────────────────────────────────────────────
# 11. KnowledgeStore Persistence & Reload
# ──────────────────────────────────────────────────────────────────────

class TestPhase4Persistence:
    def test_prd_persistence(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            prd = make_prd(requirements=["REQ-001"])
            store = KnowledgeStore(path)
            store.save(prd)
            retrieved = store.get("PRD-001")
            assert retrieved.record_id == "PRD-001"
            assert retrieved.record_type == "PRD"
            assert isinstance(retrieved, PRD)
            store.close()
        finally:
            os.unlink(path)

    def test_req_persistence(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            req = make_req()
            req.add_acceptance_criterion("AC-001", "Test AC")
            store = KnowledgeStore(path)
            store.save(req)
            retrieved = store.get("REQ-001")
            assert retrieved.record_id == "REQ-001"
            assert isinstance(retrieved, Requirement)
            assert len(retrieved.acceptance_criteria) == 1
            store.close()
        finally:
            os.unlink(path)

    def test_nfr_persistence(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            nfr = make_nfr()
            nfr.add_metric("p95_latency", "<=", 200, "ms")
            store = KnowledgeStore(path)
            store.save(nfr)
            retrieved = store.get("NFR-001")
            assert retrieved.record_id == "NFR-001"
            assert isinstance(retrieved, NFR)
            assert retrieved.metric["name"] == "p95_latency"
            store.close()
        finally:
            os.unlink(path)

    def test_restart_reload(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            # Save
            prd = make_prd(requirements=["REQ-001"])
            req = make_req()
            req.add_acceptance_criterion("AC-001", "Test AC")
            nfr = make_nfr()
            nfr.add_metric("p95_latency", "<=", 200, "ms")

            store = KnowledgeStore(path)
            store.save(prd)
            store.save(req)
            store.save(nfr)

            # Reload
            store.reload()

            # Verify
            prd2 = store.get("PRD-001")
            req2 = store.get("REQ-001")
            nfr2 = store.get("NFR-001")

            assert isinstance(prd2, PRD)
            assert isinstance(req2, Requirement)
            assert isinstance(nfr2, NFR)
            assert prd2.requirements == ["REQ-001"]
            assert req2.parent_prd == "PRD-001"
            assert nfr2.metric["operator"] == "<="

            store.close()
        finally:
            os.unlink(path)

    def test_version_preservation(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            req = make_req()
            store = KnowledgeStore(path)
            store.save(req)
            store.close()

            # Open again, create new version
            store2 = KnowledgeStore(path)
            req_old = store2.get("REQ-001")
            new_req = req_old.create_new_version("REQ-002")
            store2.save(req_old)  # Save old version with superseded_by set
            store2.save(new_req)  # Save new version
            store2.close()

            # Open again, verify both versions
            store3 = KnowledgeStore(path)
            req_v1 = store3.get("REQ-001")
            req_v2 = store3.get("REQ-002")
            assert req_v1.version == 1
            assert req_v2.version == 2
            assert req_v1.superseded_by == "REQ-002"
            store3.close()
        finally:
            os.unlink(path)

    def test_authority_preservation(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            prd = make_prd()
            prd.authority = AUTHORITY_ACCEPTED
            store = KnowledgeStore(path)
            store.save(prd)
            store.close()

            store2 = KnowledgeStore(path)
            prd2 = store2.get("PRD-001")
            assert prd2.authority == AUTHORITY_ACCEPTED
            store2.close()
        finally:
            os.unlink(path)

    def test_provenance_preservation(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            req = make_req()
            store = KnowledgeStore(path)
            store.save(req)
            store.close()

            store2 = KnowledgeStore(path)
            req2 = store2.get("REQ-001")
            assert req2.provenance.author == "test_agent"
            assert req2.provenance.source == "pipeline"
            store2.close()
        finally:
            os.unlink(path)


# ──────────────────────────────────────────────────────────────────────
# 12. Graph Integration
# ──────────────────────────────────────────────────────────────────────

class TestPhase4Graph:
    def _build_graph(self, records):
        """Helper: build graph from records using correct API."""
        from harness.knowledge import TraversalOptions
        graph = EngineeringKnowledgeGraph()
        for record in records:
            graph.add_node(record.record_id)
        for record in records:
            for rel in record.get_relationships():
                graph.add_edge(rel.source_id, rel.relation_type, rel.target_id)
        return graph

    def test_graph_with_prd_req_nfr(self):
        prd = make_prd(requirements=["REQ-001"])
        req = make_req()
        nfr = make_nfr()
        nfr.add_metric("p95_latency", "<=", 200, "ms")
        # Manually link REQ → NFR and REQ → PRD for graph edges
        req.link_nfr("NFR-001")
        req.add_relationship("PRD-001", "REQUIRES")

        graph = self._build_graph([prd, req, nfr])

        # Verify outgoing traversal: PRD → REQ
        neighbors = graph.neighbors("PRD-001")
        assert "REQ-001" in neighbors

        # Inverse traversal: incoming to REQ
        incoming = graph.neighbors("REQ-001", direction="incoming")
        assert "PRD-001" in incoming

    def test_graph_rebuild(self):
        prd = make_prd(requirements=["REQ-001"])
        req = make_req()
        nfr = make_nfr()
        nfr.add_metric("p95_latency", "<=", 200, "ms")
        req.link_nfr("NFR-001")
        req.add_relationship("PRD-001", "REQUIRES")

        g1 = self._build_graph([prd, req, nfr])
        g2 = self._build_graph([prd, req, nfr])

        # Same digest
        assert g1.compute_digest() == g2.compute_digest()

    def test_traverse_prd_to_nfr(self):
        prd = make_prd(requirements=["REQ-001"])
        req = make_req()
        nfr = make_nfr()
        nfr.add_metric("p95_latency", "<=", 200, "ms")
        req.link_nfr("NFR-001")
        req.add_relationship("PRD-001", "REQUIRES")

        graph = self._build_graph([prd, req, nfr])

        # Traverse from PRD
        from harness.knowledge import TraversalOptions
        opts = TraversalOptions(max_depth=3)
        visited = graph.traverse("PRD-001", opts)
        assert "REQ-001" in visited
        assert "NFR-001" in visited

    def test_inverse_traverse_req_to_prd(self):
        prd = make_prd(requirements=["REQ-001"])
        req = make_req()
        req.add_relationship("PRD-001", "REQUIRES")

        graph = self._build_graph([prd, req])

        # Traverse from REQ back to PRD (incoming direction)
        from harness.knowledge import TraversalOptions
        opts = TraversalOptions(max_depth=3, direction="incoming")
        visited = graph.traverse("REQ-001", opts)
        assert "PRD-001" in visited


# ──────────────────────────────────────────────────────────────────────
# 13. Serialization Determinism
# ──────────────────────────────────────────────────────────────────────

class TestPhase4Serialization:
    def test_prd_serialization_deterministic(self):
        # Use provenance with fixed timestamp for determinism
        prd1 = PRD(
            record_id="PRD-001", title="Test", description="Test",
            objective="Test", scope="Test",
            provenance=make_provenance(timestamp="2026-01-01T00:00:00"),
            created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00",
        )
        prd2 = PRD(
            record_id="PRD-001", title="Test", description="Test",
            objective="Test", scope="Test",
            provenance=make_provenance(timestamp="2026-01-01T00:00:00"),
            created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00",
        )
        assert prd1.to_dict() == prd2.to_dict()
        assert prd1.to_json() == prd2.to_json()
        assert prd1.compute_hash() == prd2.compute_hash()

    def test_req_serialization_deterministic(self):
        req1 = Requirement(
            record_id="REQ-001", title="Test", description="Test",
            provenance=make_provenance(timestamp="2026-01-01T00:00:00"),
            created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00",
        )
        req1.add_acceptance_criterion("AC-001", "Test")
        req1.updated_at = "2026-01-01T00:00:00"  # normalize
        req1.acceptance_criteria[0].created_at = "2026-01-01T00:00:00"
        req1.acceptance_criteria[0].updated_at = "2026-01-01T00:00:00"
        req2 = Requirement(
            record_id="REQ-001", title="Test", description="Test",
            provenance=make_provenance(timestamp="2026-01-01T00:00:00"),
            created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00",
        )
        req2.add_acceptance_criterion("AC-001", "Test")
        req2.updated_at = "2026-01-01T00:00:00"  # normalize
        req2.acceptance_criteria[0].created_at = "2026-01-01T00:00:00"
        req2.acceptance_criteria[0].updated_at = "2026-01-01T00:00:00"
        assert req1.to_dict() == req2.to_dict()
        assert req1.compute_hash() == req2.compute_hash()

    def test_nfr_serialization_deterministic(self):
        nfr1 = NFR(
            record_id="NFR-001", title="Test", description="Test",
            category="performance",
            provenance=make_provenance(timestamp="2026-01-01T00:00:00"),
            created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00",
        )
        nfr1.add_metric("p95_latency", "<=", 200, "ms")
        nfr1.updated_at = "2026-01-01T00:00:00"  # normalize
        nfr2 = NFR(
            record_id="NFR-001", title="Test", description="Test",
            category="performance",
            provenance=make_provenance(timestamp="2026-01-01T00:00:00"),
            created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00",
        )
        nfr2.add_metric("p95_latency", "<=", 200, "ms")
        nfr2.updated_at = "2026-01-01T00:00:00"  # normalize
        assert nfr1.to_dict() == nfr2.to_dict()

    def test_round_trip_all_types(self):
        prd = make_prd(requirements=["REQ-001"], created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00")
        req = make_req(created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00")
        req.add_acceptance_criterion("AC-001", "Test")
        nfr = make_nfr(created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00")
        nfr.add_metric("p95_latency", "<=", 200, "ms")

        for record in [prd, req, nfr]:
            data = record.to_dict()
            restored = type(record).from_dict(data)
            assert restored.to_dict() == data
            assert restored.compute_hash() == record.compute_hash()


# ──────────────────────────────────────────────────────────────────────
# 14. Domain Separation
# ──────────────────────────────────────────────────────────────────────

class TestDomainSeparation:
    def test_engineering_learning_separation(self):
        """PRD/REQ/NFR records MUST NOT replace learning records."""
        from harness.learning.experience import StructuredExperience
        from harness.learning.strategy import Strategy

        # Engineering records exist
        prd = make_prd()
        req = make_req()
        nfr = make_nfr()

        # Learning records are separate
        assert prd.__class__.__name__ != "StructuredExperience"
        assert prd.__class__.__name__ != "Strategy"
        assert req.__class__.__name__ != "StructuredExperience"
        assert nfr.__class__.__name__ != "Strategy"

    def test_engineering_evidence_separation(self):
        """NFR(Security) is NOT the same as SEC type — preserve distinction.
        NFR is Engineering Record; SEC is deferred to Phase 6."""
        nfr = make_nfr(category="security")
        # NFR has category but is NOT a SEC record
        assert nfr.record_type == "NFR"
        # SEC record type should be valid (even if not implemented yet)
        from harness.knowledge.lifecycle import VALID_RECORD_TYPES
        assert "SEC" in VALID_RECORD_TYPES


# ──────────────────────────────────────────────────────────────────────
# 15. Phase 4 E2E Scenario
# ──────────────────────────────────────────────────────────────────────

class TestPhase4E2E:
    def test_e2e_prd_req_nfr_scenario(self):
        """PRD-001 Identity Platform → REQ-001 MFA → AC-001/AC-002 + NFR-001/NFR-002.
        Persist, reload, rebuild graph, prove traceability."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            # 1. Create PRD-001 "Identity Platform"
            prd = make_prd(
                requirements=["REQ-001"],
                related_nfrs=["NFR-001", "NFR-002"],
            )

            # 2. Create REQ-001 "Support MFA authentication"
            req = make_req()
            req.add_acceptance_criterion("AC-001", "User can log in with MFA")
            req.add_acceptance_criterion("AC-002", "Recovery codes support")

            # 3. Create NFR-001 (Performance) and NFR-002 (Security)
            nfr1 = make_nfr(record_id="NFR-001")
            nfr1.add_metric("p95_latency", "<=", 200, "ms")

            nfr2 = make_nfr(
                record_id="NFR-002",
                category="security",
                qualitative=True,
            )

            # 4. Link REQ → NFR (establishes CONSTRAINED_BY relationship)
            req.link_nfr("NFR-001")
            req.link_nfr("NFR-002")
            # 5. Link REQ → PRD (inverse of PRD → REQUIRES → REQ)
            req.add_relationship("PRD-001", "REQUIRES")

            # 6. Persist
            store = KnowledgeStore(path)
            store.save(prd)
            store.save(req)
            store.save(nfr1)
            store.save(nfr2)

            # 7. Reload
            store.reload()

            # 8. Verify persistence
            prd_loaded = store.get("PRD-001")
            req_loaded = store.get("REQ-001")
            nfr1_loaded = store.get("NFR-001")
            nfr2_loaded = store.get("NFR-002")

            assert isinstance(prd_loaded, PRD)
            assert isinstance(req_loaded, Requirement)
            assert isinstance(nfr1_loaded, NFR)
            assert isinstance(nfr2_loaded, NFR)

            # 9. Verify PRD → REQ
            assert "REQ-001" in prd_loaded.requirements
            assert req_loaded.parent_prd == "PRD-001"

            # 10. Verify REQ → AC
            assert len(req_loaded.acceptance_criteria) == 2

            # 11. Verify REQ → NFR (CONSTRAINED_BY relationships preserved)
            assert "NFR-001" in req_loaded.related_nfrs
            assert "NFR-002" in req_loaded.related_nfrs

            # 12. Verify NFR metric
            assert nfr1_loaded.metric["name"] == "p95_latency"
            assert nfr1_loaded.metric["operator"] == "<="

            # 13. Verify authority/provenance preservation
            assert req_loaded.provenance.author == "test_agent"
            assert req_loaded.authority == AUTHORITY_PROPOSED

            # 14. Build graph from loaded records
            records = [prd_loaded, req_loaded, nfr1_loaded, nfr2_loaded]
            graph = _build_graph(records)

            # 15. Verify graph traversal: PRD → REQ (outgoing)
            from_prd = graph.neighbors("PRD-001")
            assert "REQ-001" in from_prd

            # 16. Inverse traversal: REQ → PRD (incoming)
            from_req = graph.neighbors("REQ-001", direction="incoming")
            assert "PRD-001" in from_req

            # 17. REQ → NFR (CONSTRAINED_BY edges)
            req_nfrs = graph.neighbors("REQ-001", relation_type="CONSTRAINED_BY")
            assert "NFR-001" in req_nfrs
            assert "NFR-002" in req_nfrs

            store.close()
        finally:
            os.unlink(path)

    def test_deterministic_graph_digest(self):
        """Two graphs from same records produce same digest."""
        prd = make_prd(requirements=["REQ-001"])
        req = make_req()
        nfr = make_nfr()
        nfr.add_metric("p95_latency", "<=", 200, "ms")

        g1 = _build_graph([prd, req, nfr])
        g2 = _build_graph([prd, req, nfr])
        assert g1.compute_digest() == g2.compute_digest()
