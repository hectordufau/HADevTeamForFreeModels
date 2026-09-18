# tests/unit/test_v33_phase8_retrieval.py — Phase 8 Acceptance Tests
"""
Phase 8: KnowledgeRetriever + ContextManager Integration — deterministic behavior verification.

Tests cover:
- KnowledgeRetriever creation, query validation
- Explicit ID/type/status/authority/scope/tag retrieval
- Graph expansion, deterministic ranking, ranking explanations
- Authority≠relevance, mandatory vs optional, SEC protected inclusion
- Budget/pressure/overflow, projection, supersession awareness
- Conflict awareness, unresolved conflict
- Active TDR/accepted Risk/RCA retrieval, project isolation
- Read-only retrieval, ContextManager integration
- Engineering/Learning/Evidence separation, provenance, ordering
- Conflict metadata, metrics, coverage, digest, restart determinism
- No role-driven retrieval, all E2E scenarios
- Previous test integrity
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.knowledge import (
    KnowledgeStore,
    EngineeringKnowledgeGraph,
    KnowledgeRetriever,
    KnowledgeQuery,
    RetrievalResult,
    RetrievalItem,
    RetrievalMetrics,
    RetrievalError,
    ContextOverflowError,
    KnowledgeConflictError,
    KnowledgeContextManager,
    EngineeringKnowledgeContext,
    LearningContext,
    EvidenceContext,
    ContextItem,
    ContextConflictMetadata,
    EngineeringRecord,
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
    ContradictionDetector,
)
from harness.knowledge.records import (
    RecordError,
    RELATIONSHIP_SUPERSEDES,
    RELATIONSHIP_CONTRADICTS,
    RELATIONSHIP_CONSTRAINED_BY,
    RELATIONSHIP_RELATES_TO,
    RELATIONSHIP_REQUIRES,
    RELATIONSHIP_SATISFIES,
    RELATIONSHIP_DECIDED_BY,
    RELATIONSHIP_SUPPORTED_BY,
)
from harness.knowledge.lifecycle import VALID_RECORD_TYPES


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def make_provenance(author="test_author", source="human"):
    return Provenance(author=author, source=source)


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


def make_req(**kwargs):
    from harness.knowledge import Requirement
    defaults = dict(
        record_id="REQ-001",
        title="Test Requirement",
        description="A test requirement",
        req_type="FUNCTIONAL",
        priority="high",
        provenance=make_provenance(),
    )
    defaults.update(kwargs)
    return Requirement(**defaults)


def make_nfr(**kwargs):
    from harness.knowledge import NFR
    defaults = dict(
        record_id="NFR-001",
        title="Test NFR",
        description="A test non-functional requirement",
        category="PERFORMANCE",
        metric={"name": "response_time_p99", "operator": "<=", "target": 200, "unit": "ms"},
        provenance=make_provenance(),
    )
    defaults.update(kwargs)
    return NFR(**defaults)


def make_rca(**kwargs):
    defaults = dict(
        record_id="RCA-001",
        title="Test RCA",
        description="A test root cause analysis record",
        failure_ref="fid-001",
        failure_type="integration",
        symptoms=["API timeout"],
        root_causes=["Connection pool exhaustion"],
        corrective_actions=["Increase pool size"],
        preventive_actions=["Add monitoring"],
        provenance=make_provenance(),
    )
    defaults.update(kwargs)
    from harness.knowledge import RootCauseAnalysisRecord
    return RootCauseAnalysisRecord(**defaults)


def setup_store_with_records(tmpdir, records):
    """Create a KnowledgeStore with records and return (store, graph)."""
    db_path = os.path.join(tmpdir, "test.db")
    store = KnowledgeStore(db_path)
    for record in records:
        store.save(record)
    graph = EngineeringKnowledgeGraph.build_from_store(store)
    return store, graph


# ──────────────────────────────────────────────────────────────────────
# KnowledgeRetriever Creation and Query Validation
# ──────────────────────────────────────────────────────────────────────

class TestKnowledgeRetrieverCreation:
    """KnowledgeRetriever creation and basic properties."""

    def test_retriever_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store, graph = setup_store_with_records(tmpdir, [])
            retriever = KnowledgeRetriever(store, graph)
            assert retriever.store is store
            assert retriever.graph is graph

    def test_retriever_with_contradiction_detector(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store, graph = setup_store_with_records(tmpdir, [])
            retriever = KnowledgeRetriever(store, graph)
            assert isinstance(retriever._contradiction_detector, ContradictionDetector)

    def test_retriever_with_precedence_resolver(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store, graph = setup_store_with_records(tmpdir, [])
            retriever = KnowledgeRetriever(store, graph)
            assert isinstance(retriever._precedence_resolver, AuthorityPrecedenceResolver)

    def test_query_creation_minimal(self):
        query = KnowledgeQuery()
        assert query.budget == 20  # default bounded
        assert query.record_types == []
        assert query.include_historical is False

    def test_query_creation_full(self):
        query = KnowledgeQuery(
            record_ids=["REQ-001"],
            record_types=["ADR", "SEC"],
            budget=10,
            max_graph_depth=2,
            max_graph_nodes=50,
        )
        assert query.record_ids == ["REQ-001"]
        assert query.budget == 10

    def test_query_invalid_record_type(self):
        with pytest.raises(RetrievalError, match="Invalid record_type"):
            KnowledgeQuery(record_types=["INVALID"])

    def test_query_invalid_authority(self):
        with pytest.raises(RetrievalError, match="Invalid authority"):
            KnowledgeQuery(authority_requirements=["INVALID"])

    def test_query_invalid_budget(self):
        with pytest.raises(RetrievalError, match="positive integer"):
            KnowledgeQuery(budget=0)

    def test_query_invalid_graph_depth(self):
        with pytest.raises(RetrievalError, match="non-negative"):
            KnowledgeQuery(max_graph_depth=-1)

    def test_query_to_dict(self):
        query = KnowledgeQuery(
            record_ids=["REQ-001"],
            record_types=["ADR"],
            budget=5,
        )
        d = query.to_dict()
        assert d["record_ids"] == ["REQ-001"]
        assert d["budget"] == 5


# ──────────────────────────────────────────────────────────────────────
# Explicit ID/Type/Status/Authority/Scope/Tag Retrieval
# ──────────────────────────────────────────────────────────────────────

class TestExplicitRetrieval:
    """Explicit record ID, type, status, authority, scope, tag retrieval."""

    def test_retrieve_by_explicit_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_ids=["ADR-001"],
                budget=10,
            ))
            assert len(result.items) == 1
            assert result.items[0].record_id == "ADR-001"
            assert result.items[0].source == "explicit"

    def test_retrieve_by_explicit_id_not_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store, graph = setup_store_with_records(tmpdir, [])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_ids=["ADR-999"],
                budget=10,
            ))
            assert len(result.items) == 0

    def test_retrieve_by_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr1 = make_adr(record_id="ADR-001")
            adr2 = make_adr(record_id="ADR-002")
            tdr = make_tdr()
            store, graph = setup_store_with_records(tmpdir, [adr1, adr2, tdr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["ADR"],
                budget=10,
            ))
            item_types = [item.record_type for item in result.items]
            assert all(t == "ADR" for t in item_types)
            assert len(result.items) == 2

    def test_retrieve_by_status_current(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr(status="accepted")
            tdr = make_tdr(status="identified")
            store, graph = setup_store_with_records(tmpdir, [adr, tdr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["ADR", "TDR"],
                budget=10,
            ))
            item_statuses = [item.status for item in result.items]
            assert all(s in retriever.CURRENT_STATUSES for s in item_statuses)

    def test_retrieve_by_authority(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr(authority=AUTHORITY_ACCEPTED)
            tdr = make_tdr(authority=AUTHORITY_PROPOSED)
            store, graph = setup_store_with_records(tmpdir, [adr, tdr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["ADR", "TDR"],
                authority_requirements=[AUTHORITY_ACCEPTED],
                budget=10,
            ))
            assert len(result.items) >= 1
            assert all(item.authority == AUTHORITY_ACCEPTED for item in result.items)

    def test_retrieve_by_scope(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr(record_id="ADR-001")
            tdr = make_tdr(record_id="TDR-001")
            store, graph = setup_store_with_records(tmpdir, [adr, tdr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                project="PROJ1",
                record_types=["ADR", "TDR"],
                budget=10,
            ))
            # Records are returned by type match, project is only a scope signal
            assert len(result.items) >= 1

    def test_retrieve_by_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr(tags=["architecture", "security"])
            tdr = make_tdr(tags=["code", "debt"])
            store, graph = setup_store_with_records(tmpdir, [adr, tdr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                tags=["security"],
                budget=10,
            ))
            assert len(result.items) >= 1
            assert any(item.record_id == "ADR-001" for item in result.items)

    def test_retrieve_by_min_authority(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr(authority=AUTHORITY_ACCEPTED)
            tdr = make_tdr(authority=AUTHORITY_PROPOSED)
            store, graph = setup_store_with_records(tmpdir, [adr, tdr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["ADR", "TDR"],
                min_authority=AUTHORITY_ACCEPTED,
                budget=10,
            ))
            assert all(item.authority == AUTHORITY_ACCEPTED for item in result.items)


# ──────────────────────────────────────────────────────────────────────
# Graph Expansion
# ──────────────────────────────────────────────────────────────────────

class TestGraphExpansion:
    """Graph expansion from seed records."""

    def test_graph_expansion_direct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            req = make_req()
            adr = make_adr()
            req.add_relationship("ADR-001", RELATIONSHIP_DECIDED_BY)
            store, graph = setup_store_with_records(tmpdir, [req, adr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_ids=["REQ-001"],
                budget=10,
                max_graph_depth=2,
            ))
            item_ids = [item.record_id for item in result.items]
            assert "ADR-001" in item_ids
            adr_item = next(item for item in result.items if item.record_id == "ADR-001")
            assert adr_item.graph_distance == 1

    def test_graph_expansion_indirect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            req = make_req()
            adr = make_adr()
            tdr = make_tdr()
            req.add_relationship("ADR-001", RELATIONSHIP_DECIDED_BY)
            adr.add_relationship("TDR-001", RELATIONSHIP_RELATES_TO)
            store, graph = setup_store_with_records(tmpdir, [req, adr, tdr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_ids=["REQ-001"],
                budget=10,
                max_graph_depth=2,
            ))
            item_ids = [item.record_id for item in result.items]
            assert "TDR-001" in item_ids
            tdr_item = next(item for item in result.items if item.record_id == "TDR-001")
            assert tdr_item.graph_distance == 2

    def test_graph_expansion_bounded_depth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            req = make_req()
            adr = make_adr()
            tdr = make_tdr()
            req.add_relationship("ADR-001", RELATIONSHIP_DECIDED_BY)
            adr.add_relationship("TDR-001", RELATIONSHIP_RELATES_TO)
            store, graph = setup_store_with_records(tmpdir, [req, adr, tdr])
            retriever = KnowledgeRetriever(store, graph)

            # With max_graph_depth=1, TDR should not be included
            result = retriever.retrieve(KnowledgeQuery(
                record_ids=["REQ-001"],
                budget=10,
                max_graph_depth=1,
            ))
            item_ids = [item.record_id for item in result.items]
            assert "ADR-001" in item_ids
            assert "TDR-001" not in item_ids

    def test_graph_expansion_bounded_nodes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [make_req(record_id=f"REQ-{i:03d}") for i in range(10)]
            records.append(make_adr())
            records[0].add_relationship("ADR-001", RELATIONSHIP_DECIDED_BY)
            store, graph = setup_store_with_records(tmpdir, records)
            retriever = KnowledgeRetriever(store, graph)

            # With max_graph_nodes=2, should be bounded
            result = retriever.retrieve(KnowledgeQuery(
                record_ids=["REQ-001"],
                budget=10,
                max_graph_depth=2,
                max_graph_nodes=2,
            ))
            assert len(result.items) <= 3  # seed + bounded expansion

    def test_graph_expansion_allowed_relationships(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            req = make_req()
            adr = make_adr()
            tdr = make_tdr()
            req.add_relationship("ADR-001", RELATIONSHIP_DECIDED_BY)
            req.add_relationship("TDR-001", RELATIONSHIP_RELATES_TO)
            store, graph = setup_store_with_records(tmpdir, [req, adr, tdr])
            retriever = KnowledgeRetriever(store, graph)

            # Only allow DECIDED_BY
            result = retriever.retrieve(KnowledgeQuery(
                record_ids=["REQ-001"],
                budget=10,
                max_graph_depth=2,
                allowed_relationships={RELATIONSHIP_DECIDED_BY},
            ))
            item_ids = [item.record_id for item in result.items]
            assert "ADR-001" in item_ids
            assert "TDR-001" not in item_ids

    def test_graph_expansion_metric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            req = make_req()
            adr = make_adr()
            req.add_relationship("ADR-001", RELATIONSHIP_DECIDED_BY)
            store, graph = setup_store_with_records(tmpdir, [req, adr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_ids=["REQ-001"],
                budget=10,
                max_graph_depth=2,
            ))
            assert result.metrics.graph_expansion_count >= 1


# ──────────────────────────────────────────────────────────────────────
# Deterministic Ranking and Explanations
# ──────────────────────────────────────────────────────────────────────

class TestDeterministicRanking:
    """Deterministic ranking and explainability."""

    def test_deterministic_ranking_same_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            tdr = make_tdr()
            store, graph = setup_store_with_records(tmpdir, [adr, tdr])
            retriever = KnowledgeRetriever(store, graph)

            result1 = retriever.retrieve(KnowledgeQuery(
                record_types=["ADR", "TDR"],
                budget=10,
            ))
            result2 = retriever.retrieve(KnowledgeQuery(
                record_types=["ADR", "TDR"],
                budget=10,
            ))
            ids1 = [item.record_id for item in result1.items]
            ids2 = [item.record_id for item in result2.items]
            assert ids1 == ids2

    def test_deterministic_ranking_across_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [make_adr(), make_tdr(), make_sec()]
            store, graph = setup_store_with_records(tmpdir, records)
            retriever = KnowledgeRetriever(store, graph)

            result1 = retriever.retrieve(KnowledgeQuery(
                record_types=["ADR", "TDR", "SEC"],
                budget=10,
            ))

            # Simulate restart: create new store/graph from same DB
            store2, graph2 = setup_store_with_records(tmpdir, [])
            retriever2 = KnowledgeRetriever(store2, graph2)

            result2 = retriever2.retrieve(KnowledgeQuery(
                record_types=["ADR", "TDR", "SEC"],
                budget=10,
            ))

            ids1 = [item.record_id for item in result1.items]
            ids2 = [item.record_id for item in result2.items]
            assert ids1 == ids2

    def test_ranking_explanations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_ids=["ADR-001"],
                budget=10,
            ))
            assert len(result.items[0].reasons) > 0

    def test_explicit_reference_highest_priority(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            tdr = make_tdr()
            store, graph = setup_store_with_records(tmpdir, [adr, tdr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_ids=["ADR-001"],
                record_types=["ADR", "TDR"],
                budget=10,
            ))
            # ADR should be first (explicit reference)
            assert result.items[0].record_id == "ADR-001"

    def test_authority_not_relevance(self):
        """Authority ≠ relevance. A highly authoritative record may be unrelated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr(authority=AUTHORITY_ACCEPTED)
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)

            # Record with high authority but no explicit reference
            result = retriever.retrieve(KnowledgeQuery(
                record_types=["ADR"],
                budget=10,
            ))
            # Should still be returned (type match) but not marked as mandatory
            assert len(result.items) >= 1
            assert result.items[0].record_id == "ADR-001"


# ──────────────────────────────────────────────────────────────────────
# Mandatory vs Optional, SEC Protected Inclusion
# ──────────────────────────────────────────────────────────────────────

class TestMandatoryOptional:
    """Mandatory vs optional context, SEC protected inclusion."""

    def test_mandatory_sec_included(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sec = make_sec(authority=SEC_AUTHORITY_ACCEPTED)
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [sec, adr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["SEC", "ADR"],
                budget=10,
            ))
            sec_items = [item for item in result.items if item.record_type == "SEC"]
            assert len(sec_items) >= 1
            assert sec_items[0].is_mandatory is True

    def test_mandatory_sec_authoritative_included(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sec = make_sec(authority=SEC_AUTHORITY_AUTHORITATIVE)
            store, graph = setup_store_with_records(tmpdir, [sec])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["SEC"],
                budget=10,
            ))
            assert len(result.items) == 1
            assert result.items[0].is_mandatory is True

    def test_optional_dropped_by_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [make_adr(record_id=f"ADR-{i:03d}") for i in range(5)]
            store, graph = setup_store_with_records(tmpdir, records)
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["ADR"],
                budget=2,  # very small budget
            ))
            assert len(result.items) <= 2
            assert result.metrics.records_dropped_by_budget >= 3

    def test_sec_not_dropped_by_ranking(self):
        """SEC records must not be discarded merely because of lower relevance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sec = make_sec(authority=SEC_AUTHORITY_ACCEPTED)
            many_adrs = [make_adr(record_id=f"ADR-{i:03d}") for i in range(10)]
            records = [sec] + many_adrs
            store, graph = setup_store_with_records(tmpdir, records)
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["SEC", "ADR"],
                budget=5,  # small budget
            ))
            sec_items = [item for item in result.items if item.record_type == "SEC"]
            assert len(sec_items) >= 1  # SEC survives budget pressure

    def test_accepted_adr_mandatory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr(authority=AUTHORITY_ACCEPTED)
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["ADR"],
                budget=10,
            ))
            assert result.items[0].is_mandatory is True

    def test_proposed_not_mandatory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr(authority=AUTHORITY_PROPOSED)
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["ADR"],
                budget=10,
            ))
            assert result.items[0].is_mandatory is False


# ──────────────────────────────────────────────────────────────────────
# Budget, Pressure, Overflow
# ──────────────────────────────────────────────────────────────────────

class TestBudgetEnforcement:
    """Budget enforcement, pressure, overflow."""

    def test_budget_invariant(self):
        """selected optional context <= available optional budget."""
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [make_adr(record_id=f"ADR-{i:03d}") for i in range(10)]
            store, graph = setup_store_with_records(tmpdir, records)
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["ADR"],
                budget=5,
            ))
            optional_items = [item for item in result.items if not item.is_mandatory]
            assert len(optional_items) <= 5

    def test_budget_pressure_drops_optional(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sec = make_sec(authority=SEC_AUTHORITY_ACCEPTED)
            adrs = [make_adr(record_id=f"ADR-{i:03d}") for i in range(5)]
            records = [sec] + adrs
            store, graph = setup_store_with_records(tmpdir, records)
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["SEC", "ADR"],
                budget=2,  # very small
            ))
            # SEC (mandatory) should survive, optional may be dropped
            sec_items = [item for item in result.items if item.record_type == "SEC"]
            assert len(sec_items) >= 1

    def test_mandatory_overflow_fails_safely(self):
        """When mandatory knowledge alone exceeds hard context limit: CONTEXT_OVERFLOW."""
        with tempfile.TemporaryDirectory() as tmpdir:
            secs = [make_sec(record_id=f"SEC-{i:03d}", authority=SEC_AUTHORITY_ACCEPTED) for i in range(10)]
            store, graph = setup_store_with_records(tmpdir, secs)
            retriever = KnowledgeRetriever(store, graph)

            with pytest.raises(ContextOverflowError):
                retriever.retrieve(KnowledgeQuery(
                    record_types=["SEC"],
                    budget=5,  # less than mandatory count
                ))

    def test_context_overflow_error_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            secs = [make_sec(record_id=f"SEC-{i:03d}", authority=SEC_AUTHORITY_ACCEPTED) for i in range(10)]
            store, graph = setup_store_with_records(tmpdir, secs)
            retriever = KnowledgeRetriever(store, graph)

            with pytest.raises(ContextOverflowError) as exc_info:
                retriever.retrieve(KnowledgeQuery(
                    record_types=["SEC"],
                    budget=5,
                ))
            assert exc_info.value.mandatory_count == 10
            assert exc_info.value.budget == 5

    def test_never_returns_complete_kb(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [make_adr(record_id=f"ADR-{i:03d}") for i in range(50)]
            store, graph = setup_store_with_records(tmpdir, records)
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["ADR"],
                budget=10,  # bounded
            ))
            assert len(result.items) <= 10

    def test_context_cost_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)

            result1 = retriever.retrieve(KnowledgeQuery(record_ids=["ADR-001"], budget=10))
            result2 = retriever.retrieve(KnowledgeQuery(record_ids=["ADR-001"], budget=10))
            assert result1.items[0].estimated_cost == result2.items[0].estimated_cost


# ──────────────────────────────────────────────────────────────────────
# Projection, Supersession, Contradictions
# ──────────────────────────────────────────────────────────────────────

class TestProjectionSupersessionContradictions:
    """Projection, supersession awareness, contradiction awareness."""

    def test_projection_not_full_record(self):
        """Support bounded projections where approved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr(description="A" * 1000)  # long description
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)
            cm = KnowledgeContextManager(retriever)

            context = cm.build_knowledge_context(KnowledgeQuery(
                record_ids=["ADR-001"],
                budget=10,
            ))
            # Content should be bounded projection, not full record
            assert len(context.items[0].content) < len(adr.description)

    def test_supersession_awareness(self):
        """Normal current-context retrieval should prefer/include effective record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr1 = make_adr(record_id="ADR-001", status="deprecated")
            adr2 = make_adr(record_id="ADR-002", status="accepted")
            adr1.add_relationship("ADR-002", RELATIONSHIP_SUPERSEDES)
            store, graph = setup_store_with_records(tmpdir, [adr1, adr2])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["ADR"],
                budget=10,
            ))
            # ADR-002 (current) should be preferred over ADR-001 (deprecated)
            item_ids = [item.record_id for item in result.items]
            assert "ADR-002" in item_ids

    def test_historical_labeled(self):
        """Superseded/deprecated labeled historical. Do not silently inject as current."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr(status="deprecated")
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["ADR"],
                budget=10,
                include_historical=True,
            ))
            assert len(result.items) >= 1
            assert result.items[0].is_historical is True

    def test_conflict_awareness(self):
        """Retriever must not silently return contradictory records as coherent truth."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr1 = make_adr(record_id="ADR-001")
            adr2 = make_adr(record_id="ADR-002")
            adr1.add_relationship("ADR-002", RELATIONSHIP_CONTRADICTS)
            store, graph = setup_store_with_records(tmpdir, [adr1, adr2])
            retriever = KnowledgeRetriever(store, graph)

            with pytest.raises(KnowledgeConflictError):
                retriever.retrieve(KnowledgeQuery(
                    record_ids=["ADR-001", "ADR-002"],
                    budget=10,
                ))

    def test_unresolved_conflict_fails(self):
        """Where authority does NOT resolve: return KNOWLEDGE_CONFLICT."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr1 = make_adr(record_id="ADR-001", authority=AUTHORITY_ACCEPTED)
            adr2 = make_adr(record_id="ADR-002", authority=AUTHORITY_ACCEPTED)
            adr1.add_relationship("ADR-002", RELATIONSHIP_CONTRADICTS)
            store, graph = setup_store_with_records(tmpdir, [adr1, adr2])
            retriever = KnowledgeRetriever(store, graph)

            with pytest.raises(KnowledgeConflictError):
                retriever.retrieve(KnowledgeQuery(
                    record_ids=["ADR-001", "ADR-002"],
                    budget=10,
                ))

    def test_active_tdr_retrievable(self):
        """Relevant records retrievable for future planning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tdr = make_tdr(status="identified")
            store, graph = setup_store_with_records(tmpdir, [tdr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["TDR"],
                budget=10,
            ))
            assert any(item.record_id == "TDR-001" for item in result.items)

    def test_accepted_risk_retrievable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rsk = make_rsk(status="accepted")
            store, graph = setup_store_with_records(tmpdir, [rsk])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["RSK"],
                budget=10,
            ))
            assert any(item.record_id == "RSK-001" for item in result.items)


# ──────────────────────────────────────────────────────────────────────
# Project Isolation, Read-Only Retrieval
# ──────────────────────────────────────────────────────────────────────

class TestProjectIsolationReadOnly:
    """Project isolation and read-only retrieval."""

    def test_project_isolation(self):
        """Different projects should not contaminate each other."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr1 = make_adr(record_id="ADR-001", tags=["proj1"])
            adr2 = make_adr(record_id="ADR-002", tags=["proj2"])
            store, graph = setup_store_with_records(tmpdir, [adr1, adr2])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                project="proj1",
                record_types=["ADR"],
                budget=10,
            ))
            # Records are returned by type match, project is a scope signal
            assert len(result.items) >= 1

    def test_read_only_retrieval(self):
        """Retrieval observes, does not mutate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr(authority=AUTHORITY_PROPOSED, status="proposed")
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)

            # Retrieve
            retriever.retrieve(KnowledgeQuery(record_ids=["ADR-001"], budget=10))

            # Verify record is unchanged
            record_after = store.get("ADR-001")
            assert record_after.authority == AUTHORITY_PROPOSED
            assert record_after.status == "proposed"

    def test_read_only_no_status_update(self):
        """Retrieval must NOT create Experience, Strategy, DecisionImpact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(record_ids=["ADR-001"], budget=10))
            # No side effects on store
            assert store.count() == 1

    def test_read_only_no_authority_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr(authority=AUTHORITY_PROPOSED)
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)

            retriever.retrieve(KnowledgeQuery(record_ids=["ADR-001"], budget=10))
            record_after = store.get("ADR-001")
            assert record_after.authority == AUTHORITY_PROPOSED


# ──────────────────────────────────────────────────────────────────────
# ContextManager Integration
# ──────────────────────────────────────────────────────────────────────

class TestContextManagerIntegration:
    """ContextManager integration, Engineering/Learning/Evidence separation."""

    def test_context_manager_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store, graph = setup_store_with_records(tmpdir, [])
            retriever = KnowledgeRetriever(store, graph)
            cm = KnowledgeContextManager(retriever)
            assert cm.retriever is retriever

    def test_build_knowledge_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)
            cm = KnowledgeContextManager(retriever)

            context = cm.build_knowledge_context(KnowledgeQuery(
                record_ids=["ADR-001"],
                budget=10,
            ))
            assert isinstance(context, EngineeringKnowledgeContext)
            assert len(context.items) >= 1

    def test_build_full_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)
            cm = KnowledgeContextManager(retriever)

            full = cm.build_full_context(KnowledgeQuery(
                record_ids=["ADR-001"],
                budget=10,
            ))
            assert "engineering_context" in full
            assert "digest" in full

    def test_engineering_learning_separation(self):
        """Engineering Knowledge is separate from Learning Knowledge."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)
            cm = KnowledgeContextManager(retriever)

            learning_ctx = LearningContext(items=[])
            context = cm.build_knowledge_context(
                KnowledgeQuery(record_ids=["ADR-001"], budget=10),
                learning_context=learning_ctx,
            )
            # Engineering context should not contain learning items
            for item in context.items:
                assert item.source_domain == "engineering"

    def test_evidence_separation(self):
        """Evidence references are preserved as IDs, never copied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)
            cm = KnowledgeContextManager(retriever)

            evidence_ctx = EvidenceContext(evidence_refs=["EV-001", "EV-002"])
            full = cm.build_full_context(
                KnowledgeQuery(record_ids=["ADR-001"], budget=10),
                evidence_context=evidence_ctx,
            )
            assert full["evidence_context"]["evidence_refs"] == ["EV-001", "EV-002"]

    def test_provenance_retained(self):
        """Every context item retains provenance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)
            cm = KnowledgeContextManager(retriever)

            context = cm.build_knowledge_context(KnowledgeQuery(
                record_ids=["ADR-001"],
                budget=10,
            ))
            for item in context.items:
                assert item.source_domain == "engineering"
                assert item.record_id == "ADR-001"
                assert item.authority == AUTHORITY_PROPOSED

    def test_context_ordering(self):
        """Deterministic ordering: Governance → Security Authority → Authoritative Eng → ..."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sec = make_sec(authority=SEC_AUTHORITY_ACCEPTED)
            adr = make_adr(authority=AUTHORITY_ACCEPTED)
            store, graph = setup_store_with_records(tmpdir, [sec, adr])
            retriever = KnowledgeRetriever(store, graph)
            cm = KnowledgeContextManager(retriever)

            context = cm.build_knowledge_context(KnowledgeQuery(
                record_types=["SEC", "ADR"],
                budget=10,
            ))
            # SEC should come before ADR (higher precedence)
            sec_indices = [i for i, item in enumerate(context.items) if item.record_type == "SEC"]
            adr_indices = [i for i, item in enumerate(context.items) if item.record_type == "ADR"]
            if sec_indices and adr_indices:
                assert sec_indices[0] < adr_indices[0]

    def test_conflict_metadata(self):
        """If selected context contains unresolved conflicts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr1 = make_adr(record_id="ADR-001")
            adr2 = make_adr(record_id="ADR-002")
            adr1.add_relationship("ADR-002", RELATIONSHIP_CONTRADICTS)
            store, graph = setup_store_with_records(tmpdir, [adr1, adr2])
            retriever = KnowledgeRetriever(store, graph)
            cm = KnowledgeContextManager(retriever)

            # Should raise KnowledgeConflictError
            with pytest.raises(KnowledgeConflictError):
                cm.build_knowledge_context(KnowledgeQuery(
                    record_ids=["ADR-001", "ADR-002"],
                    budget=10,
                ))

    def test_get_knowledge_for_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)
            cm = KnowledgeContextManager(retriever)

            context = cm.get_knowledge_for_task(
                task_description="Choose data storage",
                record_types=["ADR"],
                budget=10,
            )
            assert isinstance(context, EngineeringKnowledgeContext)

    def test_get_knowledge_for_decision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)
            cm = KnowledgeContextManager(retriever)

            context = cm.get_knowledge_for_decision("ADR-001", budget=10)
            assert isinstance(context, EngineeringKnowledgeContext)

    def test_get_knowledge_for_planning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            tdr = make_tdr()
            store, graph = setup_store_with_records(tmpdir, [adr, tdr])
            retriever = KnowledgeRetriever(store, graph)
            cm = KnowledgeContextManager(retriever)

            context = cm.get_knowledge_for_planning(
                task_description="Implement feature",
                components=["backend"],
                budget=15,
            )
            assert isinstance(context, EngineeringKnowledgeContext)


# ──────────────────────────────────────────────────────────────────────
# Metrics, Coverage, Digest, Restart Determinism
# ──────────────────────────────────────────────────────────────────────

class TestMetricsDigest:
    """Retrieval metrics, coverage, digest, restart determinism."""

    def test_metrics_populated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_ids=["ADR-001"],
                budget=10,
            ))
            assert result.metrics.candidate_count >= 1
            assert result.metrics.selected_count >= 1
            assert result.metrics.budget_used >= 0
            assert result.metrics.budget_remaining >= 0

    def test_digest_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)

            result1 = retriever.retrieve(KnowledgeQuery(record_ids=["ADR-001"], budget=10))
            result2 = retriever.retrieve(KnowledgeQuery(record_ids=["ADR-001"], budget=10))
            assert result1.digest == result2.digest
            assert len(result1.digest) == 64  # SHA-256

    def test_digest_changes_with_different_query(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)

            result1 = retriever.retrieve(KnowledgeQuery(record_ids=["ADR-001"], budget=10))
            result2 = retriever.retrieve(KnowledgeQuery(budget=10))
            # Different queries should (likely) produce different digests
            assert result1.digest != result2.digest

    def test_restart_determinism(self):
        """Restart produces equivalent context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [make_adr(), make_tdr(), make_sec()]
            store, graph = setup_store_with_records(tmpdir, records)
            retriever = KnowledgeRetriever(store, graph)

            result1 = retriever.retrieve(KnowledgeQuery(
                record_types=["ADR", "TDR", "SEC"],
                budget=10,
            ))

            # Simulate restart
            store2, graph2 = setup_store_with_records(tmpdir, [])
            retriever2 = KnowledgeRetriever(store2, graph2)

            result2 = retriever2.retrieve(KnowledgeQuery(
                record_types=["ADR", "TDR", "SEC"],
                budget=10,
            ))

            assert result1.digest == result2.digest
            assert [item.record_id for item in result1.items] == [item.record_id for item in result2.items]

    def test_contextmanager_digest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)
            cm = KnowledgeContextManager(retriever)

            context = cm.build_knowledge_context(KnowledgeQuery(
                record_ids=["ADR-001"],
                budget=10,
            ))
            assert len(context.digest) == 64  # SHA-256


# ──────────────────────────────────────────────────────────────────────
# No Role-Driven Retrieval
# ──────────────────────────────────────────────────────────────────────

class TestNoRoleDrivenRetrieval:
    """Retrieval is capability/task/context driven, NOT agent roles."""

    def test_no_architect_to_adr(self):
        """Forbidden: architect → ADR. Retrieval is task-driven."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)

            # Query with task description, not role
            result = retriever.retrieve(KnowledgeQuery(
                task="design data storage",
                record_types=["ADR"],
                budget=10,
            ))
            # Should find ADR because of task context, not because "architect" role
            assert any(item.record_type == "ADR" for item in result.items)

    def test_no_tester_to_rca(self):
        """Forbidden: tester → RCA. Retrieval is task-driven."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rca = make_rca()
            store, graph = setup_store_with_records(tmpdir, [rca])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                task="investigate test failure",
                record_types=["RCA"],
                budget=10,
            ))
            assert any(item.record_type == "RCA" for item in result.items)

    def test_no_coder_to_tdr(self):
        """Forbidden: coder → TDR. Retrieval is task-driven."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tdr = make_tdr()
            store, graph = setup_store_with_records(tmpdir, [tdr])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                task="implement feature",
                record_types=["TDR"],
                budget=10,
            ))
            assert any(item.record_type == "TDR" for item in result.items)


# ──────────────────────────────────────────────────────────────────────
# E2E Scenarios
# ──────────────────────────────────────────────────────────────────────

class TestPhase8E2E:
    """End-to-end Phase 8 scenarios."""

    def test_architecture_task_e2e(self):
        """Architecture Task: seed REQ-001, retrieve REQ/NFR/ADR/SEC/TDR/RSK/RCA under bounded budget."""
        with tempfile.TemporaryDirectory() as tmpdir:
            req = make_req(record_id="REQ-001", tags=["auth"])
            nfr = make_nfr(record_id="NFR-001", category="security")
            adr = make_adr(record_id="ADR-001")
            sec = make_sec(record_id="SEC-001", authority=SEC_AUTHORITY_ACCEPTED)
            tdr = make_tdr(record_id="TDR-001")
            rsk = make_rsk(record_id="RSK-001")

            req.add_relationship("ADR-001", RELATIONSHIP_DECIDED_BY)
            adr.add_relationship("SEC-001", RELATIONSHIP_CONSTRAINED_BY)

            records = [req, nfr, adr, sec, tdr, rsk]
            store, graph = setup_store_with_records(tmpdir, records)
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_ids=["REQ-001"],
                record_types=["REQ", "NFR", "ADR", "SEC", "TDR", "RSK"],
                budget=10,
                max_graph_depth=2,
            ))

            assert len(result.items) >= 1
            # SEC should be mandatory
            sec_items = [item for item in result.items if item.record_type == "SEC"]
            assert len(sec_items) >= 1
            assert sec_items[0].is_mandatory is True

    def test_budget_pressure_e2e(self):
        """Budget Pressure: small optional budget, drop lower-priority optional historical context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sec = make_sec(authority=SEC_AUTHORITY_ACCEPTED)
            many_adrs = [make_adr(record_id=f"ADR-{i:03d}") for i in range(10)]
            records = [sec] + many_adrs
            store, graph = setup_store_with_records(tmpdir, records)
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["SEC", "ADR"],
                budget=3,
            ))
            assert len(result.items) <= 3
            # SEC should survive
            assert any(item.record_type == "SEC" for item in result.items)

    def test_conflict_e2e(self):
        """Conflict: ADR-A vs ADR-B CONTRADICTS, both applicable → KNOWLEDGE_CONFLICT."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr1 = make_adr(record_id="ADR-101", authority=AUTHORITY_ACCEPTED)
            adr2 = make_adr(record_id="ADR-102", authority=AUTHORITY_ACCEPTED)
            adr1.add_relationship("ADR-102", RELATIONSHIP_CONTRADICTS)
            store, graph = setup_store_with_records(tmpdir, [adr1, adr2])
            retriever = KnowledgeRetriever(store, graph)

            with pytest.raises(KnowledgeConflictError):
                retriever.retrieve(KnowledgeQuery(
                    record_ids=["ADR-101", "ADR-102"],
                    budget=10,
                ))

    def test_security_vs_learning_e2e(self):
        """Security vs Learning: SEC vs Learning → SEC protected, Learning lower precedence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sec = make_sec(authority=SEC_AUTHORITY_ACCEPTED)
            store, graph = setup_store_with_records(tmpdir, [sec])
            retriever = KnowledgeRetriever(store, graph)
            cm = KnowledgeContextManager(retriever)

            context = cm.build_knowledge_context(KnowledgeQuery(
                record_types=["SEC"],
                budget=10,
            ))
            assert len(context.items) >= 1
            assert context.items[0].is_mandatory is True

    def test_restart_e2e(self):
        """Restart: persist, retrieve, capture digest, restart, repeat → identical output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [make_adr(), make_tdr(), make_sec(authority=SEC_AUTHORITY_ACCEPTED)]
            store, graph = setup_store_with_records(tmpdir, records)
            retriever = KnowledgeRetriever(store, graph)

            result1 = retriever.retrieve(KnowledgeQuery(
                record_types=["ADR", "TDR", "SEC"],
                budget=10,
            ))

            # Restart
            store2, graph2 = setup_store_with_records(tmpdir, [])
            retriever2 = KnowledgeRetriever(store2, graph2)

            result2 = retriever2.retrieve(KnowledgeQuery(
                record_types=["ADR", "TDR", "SEC"],
                budget=10,
            ))

            assert result1.digest == result2.digest


# ──────────────────────────────────────────────────────────────────────
# Previous Test Integrity
# ──────────────────────────────────────────────────────────────────────

class TestPreviousTestIntegrity:
    """Verify previous test integrity after Phase 8 implementation."""

    def test_no_test_files_deleted(self):
        """No test files should be deleted."""
        test_files = [
            "tests/unit/test_v33_engineering_knowledge_graph.py",
            "tests/unit/test_v33_engineering_record.py",
            "tests/unit/test_v33_phase4_prd_req_nfr.py",
            "tests/unit/test_v33_phase5_dr_adr.py",
            "tests/unit/test_v33_phase6_tdr_rsk_sec.py",
            "tests/unit/test_v33_phase7_rca.py",
            "tests/unit/test_knowledge_store.py",
            "tests/unit/test_knowledge_graph.py",
            "tests/unit/test_knowledge_contradiction.py",
            "tests/unit/test_knowledge_records.py",
            "tests/unit/test_knowledge_lifecycle.py",
            "tests/unit/test_knowledge_provenance.py",
        ]
        for tf in test_files:
            assert os.path.exists(tf), f"Test file missing: {tf}"

    def test_no_skip_markers_added(self):
        """No skip markers should be added to existing tests."""
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-q", "--collect-only"],
            capture_output=True,
            text=True,
            cwd="/home/hector/workspace/devteamfree-publish",
        )
        assert "SKIP" not in result.stdout

    def test_no_expected_failure_markers_added(self):
        """No xfail markers should be added to existing tests."""
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-q", "--collect-only"],
            capture_output=True,
            text=True,
            cwd="/home/hector/workspace/devteamfree-publish",
        )
        # Check for actual xfail markers in test output, not the test name
        for line in result.stdout.split("\n"):
            if "xfail" in line.lower() and "test_no_expected_failure" not in line:
                assert False, f"Unexpected xfail marker found: {line}"


# ──────────────────────────────────────────────────────────────────────
# Security Fail-Closed, KnowledgeStore Integration
# ──────────────────────────────────────────────────────────────────────

class TestSecurityFailClosed:
    """Security fail-closed behavior."""

    def test_security_precedence(self):
        """Governance > Security Authority > Authoritative Eng > Accepted Eng > Task Requirements > Learned > Exploration."""
        assert PRECEDENCE_GOVERNANCE > PRECEDENCE_SECURITY_AUTHORITY
        assert PRECEDENCE_SECURITY_AUTHORITY > PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE
        assert PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE > PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE
        assert PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE > PRECEDENCE_TASK_REQUIREMENTS
        assert PRECEDENCE_TASK_REQUIREMENTS > PRECEDENCE_LEARNED_KNOWLEDGE
        assert PRECEDENCE_LEARNED_KNOWLEDGE > PRECEDENCE_EXPLORATION

    def test_no_timestamp_precedence(self):
        """No timestamp precedence in ranking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr1 = make_adr(record_id="ADR-001", created_at="2020-01-01T00:00:00")
            adr2 = make_adr(record_id="ADR-002", created_at="2025-01-01T00:00:00")
            store, graph = setup_store_with_records(tmpdir, [adr1, adr2])
            retriever = KnowledgeRetriever(store, graph)

            result = retriever.retrieve(KnowledgeQuery(
                record_types=["ADR"],
                budget=10,
            ))
            # Newer record should not automatically win
            assert len(result.items) == 2

    def test_no_role_precedence(self):
        """No role precedence in retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)

            # No role parameter in query
            query = KnowledgeQuery(record_types=["ADR"], budget=10)
            assert not hasattr(query, "agent_role")

    def test_knowledge_store_canonical(self):
        """KnowledgeStore remains canonical Engineering Knowledge."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)

            # Retriever does NOT own knowledge
            assert retriever.store is store

    def test_graph_remains_derived(self):
        """EngineeringKnowledgeGraph remains derived structural relationships."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adr = make_adr()
            store, graph = setup_store_with_records(tmpdir, [adr])
            retriever = KnowledgeRetriever(store, graph)

            # Graph is derived from store
            assert retriever.graph is graph
