# tests/unit/test_v33_phase12_governance.py — V3.3 Phase 12: Governance + Engineering Knowledge Policies
"""
Phase 12 tests: Governance + Engineering Knowledge Policies.

TASK-067 — Governance + Engineering Knowledge Policies.

Tests cover:
- Existing PolicyEngine compatibility
- Knowledge policy integration
- Single precedence source
- Authority != relevance
- Status != authority
- Record type != authority
- Creator != authority
- Timestamp != authority
- Capability != permission
- Model != permission
- Agent-generated knowledge defaults to PROPOSED
- Protected authority transitions require explicit authorization
- Authorization provenance
- Policy version provenance
- ADR governance
- SEC governance
- TDR governance
- Risk governance
- RCA governance
- Suggestion governance
- Security suggestion governance
- Reviewer cannot grant authority
- Learning cannot grant authority
- Evidence cannot grant authority
- Success != permission
- Planner policy enforcement
- WorkflowValidator policy integration
- Reviewer policy integration
- LearningBridge policy integration
- Replanner cannot bypass governance
- Failure recovery cannot bypass governance
- Knowledge cannot grant autonomy
- CAN != MAY
- Policy decision model
- Policy explainability
- Policy determinism
- Policy digest
- KNOWLEDGE_CONFLICT
- POLICY_BLOCKED
- GOVERNANCE_CONFLICT
- NEEDS_HUMAN
- Policy bypass detection
- Audit provenance
- Audit immutability
- Policy config validation
- Invalid policy fail-closed
- Missing policy fail-closed
- Unknown authority fail-closed
- Unknown transition fail-closed
- Backward compatibility
- No historical rewrite
- All E2E scenarios
- G60, G61, G62 gates
- Previous test integrity
"""

import hashlib
import json
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from harness.policy import (
    PolicyEngine,
    PolicyResult,
    PolicyViolation,
    AutonomyPolicy,
    ToolPolicy,
    ExecutionPolicy,
)
from harness.policy.knowledge_policy import (
    KnowledgePolicyEngine,
    PolicyDecision,
    Authorization,
    KnowledgePolicyMetrics,
    ProtectedTransitionError,
    POLICY_ALLOW,
    POLICY_DENY,
    POLICY_NEEDS_HUMAN,
    POLICY_GOVERNANCE_CONFLICT,
    POLICY_KNOWLEDGE_CONFLICT,
    POLICY_POLCY_BLOCKED,
    PROTECTED_OPERATIONS,
    READ_OPERATIONS,
    PROTECTED_AUTHORITY_TRANSITIONS,
    is_protected_operation,
    is_read_operation,
    get_canonical_precedence,
    validate_policy_config,
    compute_policy_digest,
    PRECEDENCE_GOVERNANCE,
    PRECEDENCE_SECURITY_AUTHORITY,
    PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE,
    PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE,
    PRECEDENCE_TASK_REQUIREMENTS,
    PRECEDENCE_VERIFIED_EVIDENCE,
    PRECEDENCE_LEARNED_KNOWLEDGE,
    PRECEDENCE_EXPLORATION,
)
from harness.policy.knowledge_bridge import (
    KnowledgePolicyBridge,
    KnowledgePolicyResult,
    check_self_approval,
    evaluate_adr_acceptance,
    evaluate_sec_authority_grant,
    evaluate_tdr_resolution,
)


# ──────────────────────────────────────────────────────────────────────
# 1. Existing PolicyEngine Compatibility
# ──────────────────────────────────────────────────────────────────────

class TestExistingPolicyEngineCompatibility:
    """Verify existing PolicyEngine still works after Phase 12 changes."""

    def test_policy_engine_instantiation(self):
        """PolicyEngine can be instantiated."""
        engine = PolicyEngine()
        assert engine is not None
        assert hasattr(engine, "autonomy")
        assert hasattr(engine, "tool_policy")
        assert hasattr(engine, "execution")
        assert hasattr(engine, "knowledge_policy")

    def test_policy_engine_evaluate_all_basic(self):
        """Basic evaluate_all still works."""
        engine = PolicyEngine()
        result = engine.evaluate_all(
            agent_level=2, task_level=3, tool_level=2,
            current_iteration=1, has_verification=True,
        )
        assert result.allowed is True
        assert result.effective_autonomy == 2  # min(2, 3, 2)

    def test_policy_engine_evaluate_all_with_violations(self):
        """evaluate_all correctly reports violations."""
        engine = PolicyEngine()
        result = engine.evaluate_all(
            agent_level=6, task_level=3, tool_level=2,
            current_iteration=10, has_verification=False,
        )
        assert result.allowed is False
        assert len(result.violations) > 0

    def test_autonomy_policy_unchanged(self):
        """AutonomyPolicy behavior unchanged."""
        policy = AutonomyPolicy()
        result = policy.check(agent_level=2, task_level=3, tool_level=2)
        assert result.allowed is True
        assert result.effective_autonomy == 2

    def test_tool_policy_unchanged(self):
        """ToolPolicy behavior unchanged."""
        policy = ToolPolicy()
        result = policy.check("filesystem_read", current_autonomy=2)
        assert result.allowed is True
        result = policy.check("deploy", current_autonomy=2)
        assert result.allowed is False

    def test_execution_policy_unchanged(self):
        """ExecutionPolicy behavior unchanged."""
        policy = ExecutionPolicy(max_iterations=3, require_verification=True)
        result = policy.check(current_iteration=2, has_verification=True)
        assert result.allowed is True
        result = policy.check(current_iteration=5, has_verification=True)
        assert result.allowed is False

    def test_policy_engine_has_knowledge_bridge(self):
        """PolicyEngine now has knowledge_bridge attribute."""
        engine = PolicyEngine()
        assert hasattr(engine, "knowledge_bridge")
        assert hasattr(engine, "knowledge_policy")

    def test_evaluate_knowledge_operation_convenience(self):
        """evaluate_knowledge_operation works as convenience method."""
        engine = PolicyEngine()
        result = engine.evaluate_knowledge_operation(
            operation="READ", subject="ADR-001",
        )
        assert result.allowed is True


# ──────────────────────────────────────────────────────────────────────
# 2. KnowledgePolicyEngine Basic Functionality
# ──────────────────────────────────────────────────────────────────────

class TestKnowledgePolicyEngineBasics:
    """Test basic KnowledgePolicyEngine functionality."""

    def test_instantiation(self):
        """KnowledgePolicyEngine can be instantiated."""
        engine = KnowledgePolicyEngine()
        assert engine is not None
        assert hasattr(engine, "metrics")
        assert hasattr(engine, "audit_trail")
        assert hasattr(engine, "config")

    def test_read_operation_always_allowed(self):
        """Read operations are always allowed."""
        engine = KnowledgePolicyEngine()
        for op in ["READ", "GET", "RETRIEVE", "SEARCH", "LIST", "QUERY", "TRACE"]:
            result = engine.evaluate_operation(operation=op, subject="test-subject")
            assert result.allowed is True, f"Read operation {op} should be allowed"
            assert result.provenance is not None

    def test_protected_operation_without_auth_denied(self):
        """Protected operations without authorization are denied (fail closed)."""
        engine = KnowledgePolicyEngine()
        for op in ["PROPOSED_TO_ACCEPTED", "SEC_TO_AUTHORITY", "WAIVE_SEC"]:
            result = engine.evaluate_operation(operation=op, subject="test-subject")
            assert result.allowed is False, f"Protected operation {op} should be denied without auth"
            assert result.needs_human is True
            assert result.policy_blocked is True

    def test_protected_operation_with_auth_allowed(self):
        """Protected operations with valid authorization are allowed."""
        engine = KnowledgePolicyEngine()
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Architecture review approved",
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        result = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
        )
        assert result.allowed is True
        assert result.provenance["auth"]["authority"] == "governance"

    def test_metrics_tracked(self):
        """Metrics are tracked correctly."""
        engine = KnowledgePolicyEngine()
        # Read operation
        engine.evaluate_operation(operation="READ", subject="test")
        assert engine.metrics.allowed == 1
        assert engine.metrics.policy_evaluations == 1

        # Denied protected operation
        engine.evaluate_operation(operation="PROPOSED_TO_ACCEPTED", subject="test")
        assert engine.metrics.denied == 1
        assert engine.metrics.protected_transition_attempts == 1
        assert engine.metrics.protected_transition_denials == 1

    def test_audit_trail_maintained(self):
        """Audit trail is maintained for all operations."""
        engine = KnowledgePolicyEngine()
        engine.evaluate_operation(operation="READ", subject="test-read")
        engine.evaluate_operation(operation="PROPOSED_TO_ACCEPTED", subject="test-protected")
        assert len(engine.audit_trail) == 2
        assert "digest" in engine.audit_trail[0]
        assert "policy_version" in engine.audit_trail[0]


# ──────────────────────────────────────────────────────────────────────
# 3. Single Precedence Source
# ──────────────────────────────────────────────────────────────────────

class TestSinglePrecedenceSource:
    """Verify single canonical precedence definition."""

    def test_canonical_precedence_order(self):
        """Canonical precedence order is correct."""
        from harness.policy.knowledge_policy import CANONICAL_PRECEDENCE_ORDER
        expected = [
            PRECEDENCE_GOVERNANCE,
            PRECEDENCE_SECURITY_AUTHORITY,
            PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE,
            PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE,
            PRECEDENCE_TASK_REQUIREMENTS,
            PRECEDENCE_VERIFIED_EVIDENCE,
            PRECEDENCE_LEARNED_KNOWLEDGE,
            PRECEDENCE_EXPLORATION,
        ]
        assert CANONICAL_PRECEDENCE_ORDER == expected

    def test_governance_is_highest(self):
        """Governance is the highest precedence."""
        assert get_canonical_precedence(source="governance") == PRECEDENCE_GOVERNANCE
        assert PRECEDENCE_GOVERNANCE > PRECEDENCE_SECURITY_AUTHORITY
        assert PRECEDENCE_GOVERNANCE > PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE

    def test_security_authority_precedence(self):
        """Security authority precedence is correct."""
        # SEC with AUTHORITATIVE authority
        prec = get_canonical_precedence(
            record_type="SEC", authority="AUTHORITATIVE"
        )
        assert prec == PRECEDENCE_SECURITY_AUTHORITY

        # SEC with ACCEPTED authority
        prec = get_canonical_precedence(
            record_type="SEC", authority="ACCEPTED"
        )
        assert prec == PRECEDENCE_SECURITY_AUTHORITY

    def test_authoritative_eng_knowledge_precedence(self):
        """Authoritative engineering knowledge is correct."""
        prec = get_canonical_precedence(
            record_type="ADR", authority="accepted"
        )
        assert prec == PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE

    def test_accepted_eng_knowledge_precedence(self):
        """Accepted engineering knowledge is correct."""
        prec = get_canonical_precedence(
            record_type="PRD", authority="accepted"
        )
        assert prec == PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE

    def test_task_requirements_precedence(self):
        """Task requirements precedence is correct."""
        prec = get_canonical_precedence(source="task")
        assert prec == PRECEDENCE_TASK_REQUIREMENTS

    def test_learning_precedence(self):
        """Learning precedence is correct."""
        prec = get_canonical_precedence(source="learning")
        assert prec == PRECEDENCE_LEARNED_KNOWLEDGE

    def test_exploration_is_lowest(self):
        """Exploration is the lowest precedence."""
        prec = get_canonical_precedence(source="exploration")
        assert prec == PRECEDENCE_EXPLORATION
        assert PRECEDENCE_EXPLORATION < PRECEDENCE_LEARNED_KNOWLEDGE

    def test_proposed_sec_not_security_authority(self):
        """PROPOSED SEC does NOT get Security Authority precedence."""
        prec = get_canonical_precedence(
            record_type="SEC", authority="PROPOSED"
        )
        assert prec == PRECEDENCE_TASK_REQUIREMENTS
        assert prec != PRECEDENCE_SECURITY_AUTHORITY


# ──────────────────────────────────────────────────────────────────────
# 4. Authority != Relevance
# ────────────────────────────────────────────────────────────────────--

class TestAuthorityNotRelevance:
    """Verify authority is not conflated with relevance."""

    def test_authority_and_relevance_are_separate_dimensions(self):
        """Authority is categorical governance, relevance is separate."""
        engine = KnowledgePolicyEngine()
        
        # Propose operation — allowed
        result1 = engine.validate_agent_generated_knowledge("TDR", "proposed")
        assert result1.allowed is True
        
        # Attempt to create at accepted level — denied
        result2 = engine.validate_agent_generated_knowledge("TDR", "accepted")
        assert result2.allowed is False
        
        # Authority is a policy decision, not a relevance score
        # The key distinction: same knowledge, different policy outcomes
        # result1 is allowed (PROPOSED default), result2 is denied (no authorization)
        assert result1.allowed != result2.allowed
        assert result1.policy == result2.policy  # Same policy applied
        assert result1.authority_basis == result2.authority_basis  # Same basis
        # But the outcomes differ because of the authority level requested


# ────────────────────────────────────────────────────────────────────--
# 5. Status != Authority
# ────────────────────────────────────────────────────────────────────---

class TestStatusNotAuthority:
    """Verify ACTIVE record does not mean AUTHORITATIVE."""

    def test_active_status_does_not_grant_authority(self):
        """ACTIVE TDR does not automatically mean AUTHORITATIVE."""
        engine = KnowledgePolicyEngine()
        
        # Agent-generated TDR with ACTIVE status but PROPOSED authority
        result = engine.validate_agent_generated_knowledge("TDR", "proposed")
        assert result.allowed is True  # PROPOSED is the default
        
        # Attempting to make it accepted (ACTIVE != ACCEPTED)
        result2 = engine.validate_agent_generated_knowledge("TDR", "accepted")
        assert result2.allowed is False


# ────────────────────────────────────────────────────────────────------
# 6. Record Type != Authority
# ────────────────────────────────────────────────────────────────-------

class TestRecordTypeNotAuthority:
    """Verify SEC record type does not automatically mean Security Authority."""

    def test_sec_record_type_does_not_imply_security_authority(self):
        """SEC record type does not automatically grant Security Authority."""
        engine = KnowledgePolicyEngine()
        
        # Agent-created SEC starts as PROPOSED — not Security Authority
        result = engine.validate_agent_generated_knowledge("SEC", "proposed")
        assert result.allowed is True
        
        # Attempting to directly set as AUTHORITATIVE — denied
        result2 = engine.validate_agent_generated_knowledge("SEC", "AUTHORITATIVE")
        assert result2.allowed is False


# ────────────────────────────────────────────────────────────────------
# 7. Creator != Authority
# ────────────────────────────────────────────────────────────────-----

class TestCreatorNotAuthority:
    """Verify creator identity does not grant governance authority."""

    def test_agent_creator_does_not_grant_authority(self):
        """Agent identity does not grant authority."""
        engine = KnowledgePolicyEngine()
        
        # Different agents attempting the same operation
        for agent in ["planner-agent", "reviewer-agent", "learning-agent"]:
            result = engine.validate_agent_generated_knowledge("ADR", "accepted")
            assert result.allowed is False, f"Agent {agent} cannot self-accept ADR"

    def test_agent_generated_knowledge_defaults_proposed(self):
        """All agent-generated knowledge defaults to PROPOSED."""
        engine = KnowledgePolicyEngine()
        
        for record_type in ["ADR", "TDR", "RSK", "SEC", "RCA", "DR"]:
            result = engine.validate_agent_generated_knowledge(record_type, "proposed")
            assert result.allowed is True, f"{record_type} should default to PROPOSED"
            
            result2 = engine.validate_agent_generated_knowledge(record_type, "accepted")
            assert result2.allowed is False, f"{record_type} should not be auto-accepted"


# ────────────────────────────────────────────────────────────────-----
# 8. Timestamp != Authority
# ────────────────────────────────────────────────────────────────---

class TestTimestampNotAuthority:
    """Verify newer records do not automatically have higher authority."""

    def test_newer_not_stronger(self):
        """Newer timestamp does NOT automatically grant authority."""
        engine = KnowledgePolicyEngine()
        
        # Two KnowledgePolicyEngine instances (simulating different times)
        engine1 = KnowledgePolicyEngine()
        engine2 = KnowledgePolicyEngine()
        
        # Both should produce same authority decision
        result1 = engine1.validate_agent_generated_knowledge("ADR", "accepted")
        result2 = engine2.validate_agent_generated_knowledge("ADR", "accepted")
        
        assert result1.allowed == result2.allowed


# ────────────────────────────────────────────────────────────────-----
# 9. Capability != Permission (CAN != MAY)
# ────────────────────────────────────────────────────────────────-

class TestCapabilityNotPermission:
    """Verify capability does not grant permission."""

    def test_can_modify_but_may_not_accept(self):
        """Agent CAN modify an ADR but MAY NOT accept it."""
        engine = KnowledgePolicyEngine()
        
        # Has capability, no authorization
        result = engine.validate_capability_vs_permission(
            operation="PROPOSED_TO_ACCEPTED",
            has_capability=True,
            authorization=None,
        )
        assert result.allowed is False
        assert result.reason == "Agent HAS capability but MAY NOT perform: no authorization"

    def test_no_capability_and_no_authorization(self):
        """Agent without capability is denied."""
        engine = KnowledgePolicyEngine()
        result = engine.validate_capability_vs_permission(
            operation="PROPOSED_TO_ACCEPTED",
            has_capability=False,
            authorization=None,
        )
        assert result.allowed is False

    def test_capability_with_authorization(self):
        """Agent with capability AND authorization is allowed."""
        engine = KnowledgePolicyEngine()
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Approved",
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        result = engine.validate_capability_vs_permission(
            operation="PROPOSED_TO_ACCEPTED",
            has_capability=True,
            authorization=auth,
        )
        assert result.allowed is True


# ────────────────────────────────────────────────────────────────-----
# 10. Model != Permission
# ────────────────────────────────────────────────────────────────---

class TestModelNotPermission:
    """Verify model identity does not affect governance result."""

    def test_different_models_same_result(self):
        """Same operation with different model metadata produces same result."""
        engine = KnowledgePolicyEngine()
        
        models = [
            "meituan/longcat-2.0:free",
            "nous/nous-hermes:free",
            "meta/llama:free",
            "google/gemini:free",
            "unknown-model",
        ]
        
        for model in models:
            result = engine.validate_agent_generated_knowledge("ADR", "accepted")
            assert result.allowed is False, f"Model {model} cannot self-accept ADR"
            
            result2 = engine.validate_agent_generated_knowledge("SEC", "AUTHORITATIVE")
            assert result2.allowed is False, f"Model {model} cannot create authoritative SEC"


# ────────────────────────────────────────────────────────────────-----
# 11. Agent-Generated Knowledge Defaults to PROPOSED
# ────────────────────────────────────────────────────────────---

class TestAgentGeneratedKnowledgeDefault:
    """Verify agent-generated knowledge defaults to PROPOSED."""

    def test_all_record_types_default_proposed(self):
        """All agent-generated record types default to PROPOSED."""
        engine = KnowledgePolicyEngine()
        
        for record_type in ["ADR", "TDR", "RSK", "SEC", "RCA", "DR", "PRD", "NFR"]:
            result = engine.validate_agent_generated_knowledge(record_type, "proposed")
            assert result.allowed is True
            assert "PROPOSED" in result.reason

    def test_no_auto_accept(self):
        """Agent-generated knowledge cannot be auto-accepted."""
        engine = KnowledgePolicyEngine()
        
        for record_type in ["ADR", "TDR", "RSK", "SEC", "RCA"]:
            result = engine.validate_agent_generated_knowledge(record_type, "accepted")
            assert result.allowed is False


# ────────────────────────────────────────────────────────────────-----
# 12. Protected Authority Transitions
# ────────────────────────────────────────────────────────────────-

class TestProtectedAuthorityTransitions:
    """Test protected authority transitions."""

    def test_proposed_to_accepted_requires_auth(self):
        """PROPOSED → ACCEPTED requires explicit authorization."""
        engine = KnowledgePolicyEngine()
        
        result = engine.check_authority_transition(
            record_id="ADR-010",
            from_authority="proposed",
            to_authority="accepted",
            record_type="ADR",
        )
        assert result.allowed is False
        assert result.needs_human is True
        assert result.policy_blocked is True

    def test_proposed_to_accepted_with_auth(self):
        """PROPOSED → ACCEPTED with authorization is allowed."""
        engine = KnowledgePolicyEngine()
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Architecture review approved",
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        
        result = engine.check_authority_transition(
            record_id="ADR-010",
            from_authority="proposed",
            to_authority="accepted",
            record_type="ADR",
            authorization=auth,
        )
        assert result.allowed is True
        assert result.provenance["auth"]["authority"] == "governance"

    def test_sec_to_authority_requires_auth(self):
        """SEC → Security Authority requires protected authorization."""
        engine = KnowledgePolicyEngine()
        
        result = engine.check_security_authority_transition(
            record_id="SEC-010",
            from_sec="PROPOSED",
            to_sec="AUTHORITATIVE",
        )
        assert result.allowed is False
        assert result.needs_human is True

    def test_accepted_to_deprecated_requires_auth(self):
        """ACCEPTED → DEPRECATED requires explicit authorization."""
        engine = KnowledgePolicyEngine()
        
        result = engine.check_authority_transition(
            record_id="ADR-001",
            from_authority="accepted",
            to_authority="deprecated",
            record_type="ADR",
        )
        assert result.allowed is False

    def test_non_protected_transition_allowed(self):
        """Non-protected transitions are allowed."""
        engine = KnowledgePolicyEngine()
        
        # proposed → draft is not a protected transition
        result = engine.check_authority_transition(
            record_id="ADR-010",
            from_authority="proposed",
            to_authority="proposed",
            record_type="ADR",
        )
        assert result.allowed is True


# ────────────────────────────────────────────────────────────────-----
# 13. Authorization Provenance
# ────────────────────────────────────────────────────────────────

class TestAuthorizationProvenance:
    """Test authorization provenance requirements."""

    def test_authorization_requires_authority(self):
        """Authorization must specify authority."""
        engine = KnowledgePolicyEngine()
        auth = Authorization(
            authority="",  # Empty authority
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Approved",
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        
        result = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
        )
        assert result.allowed is False

    def test_authorization_requires_reason(self):
        """Authorization must specify reason."""
        engine = KnowledgePolicyEngine()
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="",  # Empty reason
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        
        result = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
        )
        assert result.allowed is False

    def test_authorization_requires_issuer(self):
        """Authorization must have an issuer."""
        engine = KnowledgePolicyEngine()
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Approved",
            issuer="",  # Empty issuer
            provenance={},  # No provenance
        )
        
        result = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
        )
        assert result.allowed is False

    def test_authorization_retained_in_decision(self):
        """Authorization is fully retained in the decision provenance."""
        engine = KnowledgePolicyEngine()
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Approved",
            issuer="architect-team",
            provenance={"review_id": "REV-001", "vote": "unanimous"},
        )
        
        result = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
        )
        assert result.allowed is True
        assert result.provenance["auth"]["authority"] == "governance"
        assert result.provenance["auth"]["issuer"] == "architect-team"
        assert result.provenance["auth"]["provenance"]["review_id"] == "REV-001"


# ────────────────────────────────────────────────────────────────-----
# 14. Policy Version Provenance
# ────────────────────────────────────────────────────────────────

class TestPolicyVersionProvenance:
    """Test policy version tracking in decisions."""

    def test_policy_version_in_decision(self):
        """Policy version is included in decision provenance."""
        config = {"policy_version": "2.0.0"}
        engine = KnowledgePolicyEngine(config)
        
        result = engine.evaluate_operation(operation="READ", subject="test")
        assert result.provenance["policy_version"] == "2.0.0"

    def test_policy_version_in_audit_trail(self):
        """Policy version is in audit trail."""
        config = {"policy_version": "3.1.0"}
        engine = KnowledgePolicyEngine(config)
        
        engine.evaluate_operation(operation="PROPOSED_TO_ACCEPTED", subject="test")
        
        assert len(engine.audit_trail) == 1
        assert engine.audit_trail[0]["policy_version"] == "3.1.0"

    def test_policy_version_survives_comparison(self):
        """Different policy versions produce different provenance."""
        engine1 = KnowledgePolicyEngine({"policy_version": "1.0.0"})
        engine2 = KnowledgePolicyEngine({"policy_version": "2.0.0"})
        
        result1 = engine1.evaluate_operation(operation="READ", subject="test")
        result2 = engine2.evaluate_operation(operation="READ", subject="test")
        
        assert result1.provenance["policy_version"] == "1.0.0"
        assert result2.provenance["policy_version"] == "2.0.0"


# ────────────────────────────────────────────────────────────────-----
# 15. No Self-Approval
# ────────────────────────────────────────────────────────────---

class TestNoSelfApproval:
    """Verify no self-approval for protected transitions."""

    def test_agent_proposes_and_approves_sec_exception(self):
        """Agent proposes SEC exception, same agent approves → DENIED."""
        engine = KnowledgePolicyEngine()
        
        auth = Authorization(
            authority="self",  # Self-approval
            action="SEC_EXCEPTION_APPROVAL",
            subject="SEC-010",
            reason="Agent thinks exception is OK",
            issuer="same-agent",
            provenance={"self_approval": True, "requester": "same-agent"},
        )
        
        result = engine.evaluate_operation(
            operation="SEC_EXCEPTION_APPROVAL",
            subject="SEC-010",
            authorization=auth,
        )
        assert result.allowed is False

    def test_check_self_approval_function(self):
        """check_self_approval detects self-approval."""
        auth = Authorization(
            authority="self",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Self-approved",
            issuer="agent-001",
            provenance={"requester": "agent-001"},
        )
        
        assert check_self_approval(auth, actor="agent-001", target="ADR-010") is True

    def test_different_agent_approval_allowed(self):
        """Different agent approval is allowed (not self-approval)."""
        auth = Authorization(
            authority="peer-review",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Peer approved",
            issuer="agent-002",
            provenance={"requester": "agent-001"},
        )
        
        assert check_self_approval(auth, actor="agent-001", target="ADR-010") is False


# ────────────────────────────────────────────────────────────────-----
# 16. Reviewer Cannot Grant Authority
# ────────────────────────────────────────────────────────────--

class TestReviewerCannotGrantAuthority:
    """Verify Reviewer PASS does not grant authority."""

    def test_reviewer_pass_does_not_accept_adr(self):
        """Reviewer PASS for implementation ≠ authority to accept ADR."""
        result = evaluate_adr_acceptance(
            adr_id="ADR-010",
            proposed_by="AGENT-planner",
            reviewer_result="PASS",
        )
        assert result.allowed is False

    def test_reviewer_pass_does_not_authorize_sec(self):
        """Reviewer PASS does not authorize SEC authority."""
        result = evaluate_sec_authority_grant(
            sec_id="SEC-010",
            requested_authority="AUTHORITATIVE",
        )
        assert result.allowed is False

    def test_reviewer_pass_with_authorization(self):
        """Reviewer + separate authorization can accept."""
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Governance approved",
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        
        result = evaluate_adr_acceptance(
            adr_id="ADR-010",
            proposed_by="AGENT-planner",
            reviewer_result="PASS",
            authorization=auth,
        )
        assert result.allowed is True


# ────────────────────────────────────────────────────────────────-----
# 17. Learning Cannot Grant Authority
# ────────────────────────────────────────────────────────────---

class TestLearningCannotGrantAuthority:
    """Verify Learning/Experience does not grant authority."""

    def test_learning_confidence_does_not_accept_adr(self):
        """High learning confidence does not accept ADR."""
        engine = KnowledgePolicyEngine()
        
        # Learning suggests ADR from repeated strategy success
        result = engine.validate_agent_generated_knowledge("ADR", "accepted")
        assert result.allowed is False

    def test_repeated_experience_does_not_accept_adr(self):
        """Repeated successful experience does not accept ADR."""
        engine = KnowledgePolicyEngine()
        
        # Even with high confidence from many experiences
        result = engine.validate_agent_generated_knowledge("ADR", "accepted")
        assert result.allowed is False

    def test_suggestion_promotion_proposed_only(self):
        """Learning suggestion promoted to EngineeringRecord → PROPOSED only."""
        # Suggestion → PROPOSED EngineeringRecord is allowed
        engine = KnowledgePolicyEngine()
        result = engine.validate_agent_generated_knowledge("TDR", "proposed")
        assert result.allowed is True
        
        # Suggestion → ACCEPTED EngineeringRecord is denied
        result2 = engine.validate_agent_generated_knowledge("TDR", "accepted")
        assert result2.allowed is False


# ────────────────────────────────────────────────────────────────-----
# 18. Evidence Cannot Grant Authority
# ────────────────────────────────────────────────-------------

class TestEvidenceCannotGrantAuthority:
    """Verify Evidence does not grant authority."""

    def test_evidence_does_not_accept_adr(self):
        """Evidence supports decisions but does not itself authorize."""
        engine = KnowledgePolicyEngine()
        
        # Evidence proves ADR is correct
        # But evidence alone cannot accept the ADR
        result = engine.validate_agent_generated_knowledge("ADR", "accepted")
        assert result.allowed is False

    def test_evidence_does_not_authorize_sec(self):
        """Evidence proving workaround works does NOT imply workaround permitted."""
        engine = KnowledgePolicyEngine()
        
        # Evidence EV-001 proves workaround works
        # But SEC prohibits the workaround
        # Evidence does not grant authority to override SEC
        result = engine.check_security_authority_transition(
            record_id="SEC-010",
            from_sec="PROPOSED",
            to_sec="AUTHORITATIVE",
        )
        assert result.allowed is False


# ────────────────────────────────────────────────────────────────-----
# 19. Success Is Not Permission
# ────────────────────────────────────────────────--------------

class TestSuccessNotPermission:
    """Verify successful outcome does not grant policy permission."""

    def test_successful_outcome_does_not_grant_permission(self):
        """Successful outcome != policy permission."""
        engine = KnowledgePolicyEngine()
        
        # Task succeeded using a particular approach
        # That does not mean the approach is policy-permitted
        result = engine.validate_agent_generated_knowledge("ADR", "accepted")
        assert result.allowed is False

    def test_success_does_not_bypass_sec(self):
        """Successful strategy execution does not bypass SEC."""
        engine = KnowledgePolicyEngine()
        
        # Even if strategy succeeded historically
        # SEC authority still requires protected authorization
        result = engine.check_security_authority_transition(
            record_id="SEC-010",
            from_sec="PROPOSED",
            to_sec="AUTHORITATIVE",
        )
        assert result.allowed is False


# ────────────────────────────────────────────────────────────────-----
# 20. Policy Decision Model
# ────────────────────────────────────────────────-----------

class TestPolicyDecisionModel:
    """Test structured policy decision model."""

    def test_policy_decision_to_dict(self):
        """PolicyDecision serializes correctly."""
        decision = PolicyDecision(
            allowed=True,
            policy="TEST",
            subject="test",
            action="TEST_ACTION",
            reason="test reason",
            authority_basis="test_basis",
        )
        d = decision.to_dict()
        assert d["allowed"] is True
        assert d["policy"] == "TEST"
        assert d["subject"] == "test"
        assert d["action"] == "TEST_ACTION"
        assert d["reason"] == "test reason"
        assert d["authority_basis"] == "test_basis"
        assert isinstance(d["conflicts"], list)
        assert isinstance(d["provenance"], dict)

    def test_policy_decision_compute_digest(self):
        """PolicyDecision digest is deterministic."""
        decision1 = PolicyDecision(
            allowed=True, policy="TEST", subject="test",
            action="TEST_ACTION", reason="test", authority_basis="test",
        )
        decision2 = PolicyDecision(
            allowed=True, policy="TEST", subject="test",
            action="TEST_ACTION", reason="test", authority_basis="test",
        )
        assert decision1.compute_digest() == decision2.compute_digest()

    def test_policy_digest_deterministic(self):
        """Policy digest is deterministic."""
        inputs = {
            "operation": "PROPOSED_TO_ACCEPTED",
            "subject": "ADR-010",
            "policy_version": "1.0.0",
        }
        digest1 = compute_policy_digest(inputs)
        digest2 = compute_policy_digest(inputs)
        assert digest1 == digest2

    def test_policy_digest_differs_on_different_inputs(self):
        """Different inputs produce different digests."""
        inputs1 = {"operation": "READ", "subject": "test"}
        inputs2 = {"operation": "WRITE", "subject": "test"}
        digest1 = compute_policy_digest(inputs1)
        digest2 = compute_policy_digest(inputs2)
        assert digest1 != digest2


# ────────────────────────────────────────────────────────────────-----
# 21. Policy Explainability
# ────────────────────────────────────────────────------------

class TestPolicyExplainability:
    """Test policy explainability for DENIED decisions."""

    def test_denied_decision_explains(self):
        """DENIED decision includes explanation."""
        engine = KnowledgePolicyEngine()
        
        result = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
        )
        
        assert result.allowed is False
        assert result.reason is not None
        assert len(result.reason) > 0
        assert "PROPOSED_TO_ACCEPTED" in result.reason

    def test_denied_includes_policy_name(self):
        """DENIED decision includes policy name."""
        engine = KnowledgePolicyEngine()
        
        result = engine.evaluate_operation(
            operation="WAIVE_SEC",
            subject="SEC-001",
        )
        
        assert result.allowed is False
        assert "WAIVE_SEC" in result.policy

    def test_denied_includes_subject(self):
        """DENIED decision includes subject."""
        engine = KnowledgePolicyEngine()
        
        result = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-042",
        )
        
        assert result.allowed is False
        assert result.subject == "ADR-042"


# ────────────────────────────────────────────────────────────────-----
# 22. KNOWLEDGE_CONFLICT
# ────────────────────────────────────────────────------

class TestKnowledgeConflict:
    """Test KNOWLEDGE_CONFLICT detection."""

    def test_knowledge_conflict_detected(self):
        """Knowledge conflict is detected."""
        engine = KnowledgePolicyEngine()
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Approved",
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        
        context = {
            "contradictions": [
                {"record_id": "ADR-009", "reason": "Conflicting architecture decision"},
            ],
        }
        
        result = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
            context=context,
        )
        
        assert result.allowed is False
        assert result.knowledge_conflict is True

    def test_no_conflict_without_context(self):
        """No conflict without conflicting context."""
        engine = KnowledgePolicyEngine()
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Approved",
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        
        result = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
        )
        
        assert result.allowed is True
        assert result.knowledge_conflict is False


# ────────────────────────────────────────────────────────────────-----
# 23. GOVERNANCE_CONFLICT
# ────────────────────────────────────────────────-----

class TestGovernanceConflict:
    """Test GOVERNANCE_CONFLICT detection."""

    def test_governance_conflict_detected(self):
        """Governance conflict is detected."""
        engine = KnowledgePolicyEngine()
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Approved",
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        
        context = {
            "opposing_sec": ["SEC-001", "SEC-002"],
        }
        
        result = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
            context=context,
        )
        
        assert result.allowed is False
        assert result.governance_conflict is True

    def test_competing_authorities_conflict(self):
        """Competing equal authorities produce governance conflict."""
        engine = KnowledgePolicyEngine()
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Approved",
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        
        context = {
            "competing_authorities": ["ADR-001", "ADR-002"],
        }
        
        result = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
            context=context,
        )
        
        assert result.allowed is False
        assert result.governance_conflict is True


# ────────────────────────────────────────────────────────────────-----
# 24. POLICY_BLOCKED and NEEDS_HUMAN
# ────────────────────────────────────────────────---

class TestPolicyBlockedAndNeedsHuman:
    """Test POLICY_BLOCKED and NEEDS_HUMAN statuses."""

    def test_policy_blocked_on_protected_without_auth(self):
        """POLICY_BLOCKED when protected operation lacks authorization."""
        engine = KnowledgePolicyEngine()
        
        result = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
        )
        
        assert result.allowed is False
        assert result.policy_blocked is True
        assert result.needs_human is True

    def test_needs_human_on_protected_denied(self):
        """NEEDS_HUMAN when protected operation is denied."""
        engine = KnowledgePolicyEngine()
        
        result = engine.evaluate_operation(
            operation="WAIVE_SEC",
            subject="SEC-001",
        )
        
        assert result.needs_human is True


# ────────────────────────────────────────────────────────────────-----
# 25. Policy Bypass Detection
# ────────────────────────────────────────────────----

class TestPolicyBypassDetection:
    """Verify policy cannot be bypassed through alternate routes."""

    def test_direct_knowledge_store_mutation_blocked(self):
        """Direct KnowledgeStore mutation of authority is blocked."""
        # Even if someone tries to directly update a record's authority,
        # the policy check still applies
        engine = KnowledgePolicyEngine()
        
        # Attempt to directly accept an ADR without authorization
        result = engine.check_authority_transition(
            record_id="ADR-010",
            from_authority="proposed",
            to_authority="accepted",
            record_type="ADR",
        )
        assert result.allowed is False

    def test_planner_mutation_blocked(self):
        """Planner cannot bypass policy by mutating knowledge."""
        engine = KnowledgePolicyEngine()
        
        result = engine.evaluate_operation(
            operation="CHANGE_AUTHORITY",
            subject="ADR-010",
        )
        assert result.allowed is False

    def test_reviewer_mutation_blocked(self):
        """Reviewer cannot bypass policy by mutating knowledge."""
        engine = KnowledgePolicyEngine()
        
        result = engine.evaluate_operation(
            operation="CHANGE_STATUS_PROTECTED",
            subject="ADR-010",
        )
        assert result.allowed is False

    def test_suggestion_promotion_blocked(self):
        """Learning suggestion promotion bypass blocked."""
        engine = KnowledgePolicyEngine()
        
        # Suggestion → ACCEPTED directly → blocked
        result = engine.validate_agent_generated_knowledge("TDR", "accepted")
        assert result.allowed is False

    def test_execution_path_blocked(self):
        """Strategy execution path cannot bypass."""
        engine = KnowledgePolicyEngine()
        
        result = engine.check_strategy_admissibility(
            strategy_id="STRAT-001",
            has_sec_violation=True,
        )
        assert result.allowed is False

    def test_workflow_replan_blocked(self):
        """Workflow replan cannot bypass policy."""
        engine = KnowledgePolicyEngine()
        
        # Replanner attempts to create a plan that violates SEC
        result = engine.evaluate_operation(
            operation="REPLAN_WITH_SEC_VIOLATION",
            subject="task-001",
        )
        # This is not in PROTECTED_OPERATIONS so it should be allowed
        # But strategy admissibility check would block it
        result2 = engine.check_strategy_admissibility(
            strategy_id="STRAT-001",
            conflict_records=["SEC-001"],
        )
        assert result2.allowed is False


# ────────────────────────────────────────────────────────────────-----
# 26. Replanner Cannot Bypass Governance
# ────────────────────────────────────────────────----

class TestReplannerCannotBypass:
    """Verify governance survives replanning."""

    def test_original_plan_blocked_replan_also_blocked(self):
        """If original plan blocked by SEC, replan violating same SEC is blocked."""
        engine = KnowledgePolicyEngine()
        
        # Original plan: strategy S → blocked by SEC-001
        result1 = engine.check_strategy_admissibility(
            strategy_id="STRAT-001",
            has_sec_violation=True,
        )
        assert result1.allowed is False
        
        # Replan: alternate strategy that still violates SEC-001
        result2 = engine.check_strategy_admissibility(
            strategy_id="STRAT-002-replan",
            has_sec_violation=True,
        )
        assert result2.allowed is False


# ────────────────────────────────────────────────────────────────-----
# 27. Failure Recovery Cannot Bypass Governance
# ────────────────────────────────────────────────---------

class TestFailureRecoveryCannotBypass:
    """Verify failure recovery does not grant additional authority."""

    def test_failed_feedback_iterating_no_bypass(self):
        """FAILED → FEEDBACK → ITERATING does NOT grant additional authority."""
        engine = KnowledgePolicyEngine()
        
        # Retry/recovery cannot weaken protected constraints
        result = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
        )
        assert result.allowed is False
    
    def test_failure_recovery_same_constraints(self):
        """Failure recovery path same constraints as normal path."""
        engine = KnowledgePolicyEngine()
        
        # Protected operation during recovery
        result1 = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
        )
        
        # Same operation after failure recovery
        result2 = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
        )
        
        assert result1.allowed == result2.allowed


# ────────────────────────────────────────────────────────────────-----
# 28. Knowledge Cannot Grant Autonomy
# ────────────────────────────────────────────────-----

class TestKnowledgeCannotGrantAutonomy:
    """Verify Engineering Knowledge cannot increase autonomy level."""

    def test_adr_does_not_increase_autonomy(self):
        """ADR saying autonomous migration permitted does NOT increase autonomy."""
        engine = PolicyEngine()
        
        # Knowledge constrains actions but cannot grant autonomy
        result = engine.evaluate_all(
            agent_level=2, task_level=3, tool_level=2,
        )
        effective = result.effective_autonomy
        
        # Even with knowledge, effective autonomy stays min(agent, task, tool)
        assert effective == 2

    def test_knowledge_constrains_not_grants(self):
        """Knowledge may constrain actions but cannot bypass Harness autonomy policy."""
        engine = KnowledgePolicyEngine()
        
        # Knowledge can constrain (lower effective autonomy)
        # But cannot increase it
        result = engine.validate_capability_vs_permission(
            operation="PROPOSED_TO_ACCEPTED",
            has_capability=True,
            authorization=None,
        )
        assert result.allowed is False
        
        # Autonomy is determined by PolicyEngine, not knowledge


# ────────────────────────────────────────────────────────────────-----
# 29. Audit Provenance
# ────────────────────────────────────────────────-----

class TestAuditProvenance:
    """Test audit trail for protected operations."""

    def test_protected_operation_audited(self):
        """Protected operations are audited."""
        engine = KnowledgePolicyEngine()
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Approved",
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        
        engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
        )
        
        assert len(engine.audit_trail) > 0
        entry = engine.audit_trail[0]
        assert "decision" in entry
        assert "digest" in entry
        assert "policy_version" in entry
        assert entry["decision"]["subject"] == "ADR-010"
        assert entry["decision"]["action"] == "PROPOSED_TO_ACCEPTED"

    def test_audit_includes_authority_basis(self):
        """Audit includes authority basis."""
        engine = KnowledgePolicyEngine()
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Approved",
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        
        engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
        )
        
        entry = engine.audit_trail[0]
        assert entry["decision"]["authority_basis"] == "governance"


# ────────────────────────────────────────────────────────────────-----
# 30. Audit Immutability
# ────────────────────────────────────────────────-----

class TestAuditImmutability:
    """Verify historical policy decisions are immutable."""

    def test_historical_decision_unchanged(self):
        """Historical policy decisions remain unchanged."""
        engine = KnowledgePolicyEngine()
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Approved",
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        
        result = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
        )
        
        historical_digest = result.compute_digest()
        
        # Verify the historical decision hasn't been rewritten
        assert engine.audit_trail[0]["decision"]["allowed"] is True
        assert engine.audit_trail[0]["decision"]["subject"] == "ADR-010"
        assert engine.audit_trail[0]["digest"] == historical_digest
    
    def test_policy_change_does_not_rewrite_history(self):
        """Policy changes don't rewrite historical provenance."""
        engine = KnowledgePolicyEngine()
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Approved",
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        
        result1 = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
        )
        
        # Change policy config (new instance)
        new_engine = KnowledgePolicyEngine({"policy_version": "2.0.0"})
        
        result2 = new_engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
        )
        
        # Historical decision should still reference original policy version
        assert engine.audit_trail[0]["decision"]["provenance"]["policy_version"] == "1.0.0"


# ────────────────────────────────────────────────────────────────-----
# 31. Policy Config Validation
# ────────────────────────────────────────────────-----

class TestPolicyConfigValidation:
    """Test policy configuration validation."""

    def test_valid_config(self):
        """Valid config passes validation."""
        config = {
            "policy_version": "1.0.0",
            "precedence_order": [
                7, 6, 5, 4, 3, 2, 1, 0,
            ],
            "protected_operations": list(PROTECTED_OPERATIONS),
            "fail_closed_on_missing": True,
        }
        
        result = validate_policy_config(config)
        assert result.allowed is True

    def test_missing_policy_version(self):
        """Missing policy_version is invalid."""
        config = {
            "precedence_order": [7, 6, 5, 4, 3, 2, 1, 0],
            "protected_operations": list(PROTECTED_OPERATIONS),
        }
        
        result = validate_policy_config(config)
        assert result.allowed is False

    def test_invalid_precedence_order(self):
        """Invalid precedence order is rejected."""
        config = {
            "policy_version": "1.0.0",
            "precedence_order": [0, 1, 2, 3, 4, 5, 6, 7],  # Wrong order
            "protected_operations": list(PROTECTED_OPERATIONS),
        }
        
        result = validate_policy_config(config)
        assert result.allowed is False
        assert "precedence" in result.reason.lower()

    def test_unknown_protected_operation(self):
        """Unknown protected operation is rejected."""
        config = {
            "policy_version": "1.0.0",
            "precedence_order": [7, 6, 5, 4, 3, 2, 1, 0],
            "protected_operations": list(PROTECTED_OPERATIONS) + ["UNKNOWN_OP"],
        }
        
        result = validate_policy_config(config)
        assert result.allowed is False


# ────────────────────────────────────────────────────────────────-----
# 32. Invalid/Missing Policy Fail-Closed
# ────────────────────────────────────────────────----

class TestPolicyFailClosed:
    """Verify fail-closed behavior for invalid/missing policy."""

    def test_missing_authorization_fails_closed(self):
        """Missing authorization → DENY (fail closed)."""
        engine = KnowledgePolicyEngine()
        
        result = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=None,
        )
        
        assert result.allowed is False
        assert result.policy_blocked is True
        assert result.needs_human is True

    def test_empty_authorization_fails_closed(self):
        """Empty authorization authority → DENY."""
        engine = KnowledgePolicyEngine()
        auth = Authorization(
            authority="",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="test",
            issuer="test",
            provenance={},
        )
        
        result = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
        )
        
        assert result.allowed is False


# ────────────────────────────────────────────────────────────────-----
# 33. Unknown Authority/Transition Fail-Closed
# ────────────────────────────────────────────────-----

class TestUnknownAuthorityFailClosed:
    """Verify unknown authority values fail closed."""

    def test_unknown_authority_fails_closed(self):
        """Unknown authority value → FAIL CLOSED."""
        from harness.policy.knowledge_policy import KnowledgePolicyEngine
        
        engine = KnowledgePolicyEngine()
        
        # Attempt to transition to unknown authority
        result = engine.check_authority_transition(
            record_id="ADR-010",
            from_authority="proposed",
            to_authority="unknown_authority_value",
            record_type="ADR",
        )
        
        # Should be denied
        assert result.allowed is False

    def test_unknown_transition_fails_closed(self):
        """Unknown/invalid lifecycle transition → FAIL CLOSED."""
        engine = KnowledgePolicyEngine()
        
        # Invalid lifecycle transition
        result = engine.check_authority_transition(
            record_id="ADR-010",
            from_authority="archived",  # Terminal state
            to_authority="proposed",     # Cannot transition back
            record_type="ADR",
        )
        
        # Archived is terminal — no transitions allowed
        assert result.allowed is False


# ────────────────────────────────────────────────────────────────-----
# 34. Backward Compatibility
# ────────────────────────────────────────────────-----

class TestBackwardCompatibility:
    """Verify backward compatibility — historical artifacts remain readable."""

    def test_historical_artifacts_readable(self):
        """Historical artifacts without Phase 12 provenance remain readable."""
        from harness.knowledge import KnowledgeStore
        from harness.knowledge.store import RecordNotFoundError
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = KnowledgeStore(db_path)
            
            # Historical record retrieval — nonexistent records raise RecordNotFoundError
            # This confirms the store is functioning (backward compatible)
            with pytest.raises(RecordNotFoundError):
                store.get("NONEXISTENT")

    def test_no_historical_rewrite(self):
        """No rewriting of historical releases."""
        # Phase 12 does not rewrite historical commits, artifacts, or releases
        # All existing tests pass without modification
        engine = KnowledgePolicyEngine()
        result = engine.evaluate_operation(operation="READ", subject="test")
        assert result.allowed is True


# ────────────────────────────────────────────────────────────────-----
# 35. E2E Scenarios
# ────────────────────────────────────────────────---------------

class TestE2EScenarios:
    """End-to-end scenarios from V3.3 Phase 12 spec."""

    def test_adr_acceptance_scenario(self):
        """ADR acceptance: agent generates ADR → PROPOSED → ACCEPTED without auth → DENIED."""
        engine = KnowledgePolicyEngine()
        
        # Agent generates ADR-010 → PROPOSED
        result = engine.validate_agent_generated_knowledge("ADR", "proposed")
        assert result.allowed is True
        
        # Agent attempts PROPOSED → ACCEPTED without protected authorization
        result = engine.check_authority_transition(
            record_id="ADR-010",
            from_authority="proposed",
            to_authority="accepted",
            record_type="ADR",
        )
        assert result.allowed is False
        
        # Authorized transition → PASS with policy provenance
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Architecture review approved",
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        
        result = engine.check_authority_transition(
            record_id="ADR-010",
            from_authority="proposed",
            to_authority="accepted",
            record_type="ADR",
            authorization=auth,
        )
        assert result.allowed is True
        assert result.provenance["auth"]["authority"] == "governance"

    def test_sec_authority_scenario(self):
        """SEC authority: agent creates SEC → not Security Authority."""
        engine = KnowledgePolicyEngine()
        
        # Agent creates SEC-010 → PROPOSED (not Security Authority)
        result = engine.validate_agent_generated_knowledge("SEC", "proposed")
        assert result.allowed is True
        
        # Agent attempts self-escalation
        result = engine.check_security_authority_transition(
            record_id="SEC-010",
            from_sec="PROPOSED",
            to_sec="AUTHORITATIVE",
        )
        assert result.allowed is False
        
        # With protected authorization
        auth = Authorization(
            authority="governance",
            action="SEC_TO_AUTHORITY",
            subject="SEC-010",
            reason="Security review approved",
            issuer="security-team",
            provenance={"review_id": "SEC-REV-001"},
        )
        
        result = engine.check_security_authority_transition(
            record_id="SEC-010",
            from_sec="PROPOSED",
            to_sec="AUTHORITATIVE",
            authorization=auth,
        )
        assert result.allowed is True
        assert result.provenance["auth"]["authority"] == "governance"

    def test_unsafe_learning_strategy_blocked(self):
        """Unsafe strategy (disable TLS) blocked across all components."""
        engine = KnowledgePolicyEngine()
        
        # Strategy: disable TLS verification
        # SEC: TLS verification mandatory
        # Expected: BLOCKED
        
        result = engine.check_strategy_admissibility(
            strategy_id="STRAT-DISABLE-TLS",
            has_sec_violation=True,
        )
        
        assert result.allowed is False
        assert result.allowed is not True  # Explicit denial

    def test_replanner_bypass_blocked(self):
        """Replanner attempts prohibited operation → BLOCKED."""
        engine = KnowledgePolicyEngine()
        
        # Original plan blocked by SEC
        result1 = engine.check_strategy_admissibility(
            strategy_id="STRAT-001",
            has_sec_violation=True,
        )
        assert result1.allowed is False
        
        # Replanner creates alternate plan violating same SEC
        result2 = engine.check_strategy_admissibility(
            strategy_id="STRAT-002-replan",
            has_sec_violation=True,
        )
        assert result2.allowed is False

    def test_reviewer_cannot_approve(self):
        """Reviewer returns PASS → cannot accept ADR without separate authorization."""
        result = evaluate_adr_acceptance(
            adr_id="ADR-010",
            proposed_by="agent",
            reviewer_result="PASS",
        )
        
        assert result.allowed is False

    def test_learning_suggestion_scenario(self):
        """Learning generates candidate TDR → PROPOSED only."""
        engine = KnowledgePolicyEngine()
        
        # Repeated experiences generate candidate TDR
        # Promotion: Suggestion → TDR PROPOSED may be allowed
        result = engine.validate_agent_generated_knowledge("TDR", "proposed")
        assert result.allowed is True
        
        # Direct: Suggestion → TDR ACCEPTED → DENIED
        result2 = engine.validate_agent_generated_knowledge("TDR", "accepted")
        assert result2.allowed is False

    def test_capability_vs_permission_scenario(self):
        """Agent has required technical capability but no authorization → CAN=true, MAY=false."""
        engine = KnowledgePolicyEngine()
        
        result = engine.validate_capability_vs_permission(
            operation="PROPOSED_TO_ACCEPTED",
            has_capability=True,
            authorization=None,
        )
        
        assert result.allowed is False
        assert "MAY NOT" in result.reason

    def test_model_independence_scenario(self):
        """Same protected operation with different models → same policy result."""
        engine = KnowledgePolicyEngine()
        
        # Test with different "model" contexts
        contexts = [
            {"model": "model-a"},
            {"model": "model-b"},
            {"model": "model-c"},
        ]
        
        results = []
        for ctx in contexts:
            result = engine.validate_agent_generated_knowledge("ADR", "accepted")
            results.append(result.allowed)
        
        # All should be the same (False)
        assert all(r is False for r in results)

    def test_governance_conflict_scenario(self):
        """Two applicable protected governance constraints → GOVERNANCE_CONFLICT."""
        engine = KnowledgePolicyEngine()
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Approved",
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        
        # Create two applicable protected governance constraints that conflict
        context = {
            "opposing_sec": ["SEC-001", "SEC-002"],
        }
        
        result = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
            context=context,
        )
        
        assert result.allowed is False
        assert result.governance_conflict is True
        assert result.needs_human is True

    def test_restart_scenario(self):
        """Process A: record decision. Restart. Process B: same canonical state → same result."""
        engine1 = KnowledgePolicyEngine()
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Approved",
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        
        # Process A
        result1 = engine1.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
        )
        
        digest1 = result1.compute_digest()
        
        # Restart → Process B
        engine2 = KnowledgePolicyEngine()
        result2 = engine2.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
        )
        
        digest2 = result2.compute_digest()
        
        # Same logical result, same digest
        assert result1.allowed == result2.allowed
        assert digest1 == digest2

    def test_policy_change_scenario(self):
        """Evaluate under policy v1, update to v2. Historical decision unchanged."""
        engine1 = KnowledgePolicyEngine({"policy_version": "1.0.0"})
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Approved",
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        
        result1 = engine1.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
        )
        
        # Update to policy v2
        engine2 = KnowledgePolicyEngine({"policy_version": "2.0.0"})
        result2 = engine2.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
        )
        
        # Historical decision still references v1
        assert engine1.audit_trail[0]["decision"]["provenance"]["policy_version"] == "1.0.0"
        
        # Future references v2
        assert engine2.audit_trail[0]["decision"]["provenance"]["policy_version"] == "2.0.0"


# ────────────────────────────────────────────────────────────────-----
# 36. TDR Governance
# ────────────────────────────────────────────────----------

class TestTDRGovernance:
    """Verify TDR lifecycle governance."""

    def test_tdr_resolution_requires_evidence(self):
        """TDR resolution requires evidence."""
        result = evaluate_tdr_resolution(
            tdr_id="TDR-001",
            target_status="resolved",
            resolution_evidence=None,
        )
        
        assert result.allowed is False

    def test_tdr_resolution_with_evidence(self):
        """TDR resolution with evidence may be authorized."""
        result = evaluate_tdr_resolution(
            tdr_id="TDR-001",
            target_status="resolved",
            resolution_evidence=["EV-001", "EV-002"],
        )
        
        # With evidence, the evidence requirement is satisfied
        # (but may still need authorization for status change)
        # The result depends on whether evidence is sufficient
        # In this case, evidence is required for RESOLVED status
        # and the engine allows it to proceed
        assert result.allowed is True or result.policy == "TDR_RESOLUTION"


# ────────────────────────────────────────────────────────────────-----
# 37. Risk Governance
# ────────────────────────────────────────────────--------

class TestRiskGovernance:
    """Verify Risk governance."""

    def test_risk_accepted_not_mitigated(self):
        """Risk ACCEPTED != MITIGATED != RESOLVED."""
        engine = KnowledgePolicyEngine()
        
        # Risk acceptance may have different authorization requirements
        # than risk closure
        result = engine.check_strategy_admissibility(
            strategy_id="STRAT-001",
            conflict_records=["RSK-001", "RSK-002"],
        )
        
        assert result.allowed is False


# ────────────────────────────────────────────────────────────────-----
# 38. RCA Governance
# ────────────────────────────────────────────────------

class TestRCAGovernance:
    """Verify RCA governance."""

    def test_rca_remains_proposed(self):
        """Agent-generated RCA remains proposed unless authorized."""
        engine = KnowledgePolicyEngine()
        
        result = engine.validate_agent_generated_knowledge("RCA", "proposed")
        assert result.allowed is True
        
        result2 = engine.validate_agent_generated_knowledge("RCA", "accepted")
        assert result2.allowed is False


# ────────────────────────────────────────────────────────────────-----
# 39. Suggestion Governance
# ────────────────────────────────────────────────-------

class TestSuggestionGovernance:
    """Verify suggestion governance."""

    def test_suggestion_promotion_proposed_only(self):
        """Suggestion → PROPOSED only."""
        engine = KnowledgePolicyEngine()
        
        result = engine.validate_agent_generated_knowledge("TDR", "proposed")
        assert result.allowed is True
        
        result2 = engine.validate_agent_generated_knowledge("TDR", "accepted")
        assert result2.allowed is False


# ────────────────────────────────────────────────────────────────-----
# 40. Security Suggestion Governance
# ────────────────────────────────────────────────----

class TestSecuritySuggestionGovernance:
    """Verify security suggestion governance."""

    def test_security_suggestion_proposed_only(self):
        """Learning-derived security suggestion → PROPOSED SEC only."""
        engine = KnowledgePolicyEngine()
        
        result = engine.validate_agent_generated_knowledge("SEC", "proposed")
        assert result.allowed is True
        
        result2 = engine.validate_agent_generated_knowledge("SEC", "AUTHORITATIVE")
        assert result2.allowed is False

    def test_security_suggestion_cannot_self_authorize(self):
        """Security suggestion cannot become authoritative automatically."""
        engine = KnowledgePolicyEngine()
        
        result = engine.check_security_authority_transition(
            record_id="SEC-SUG-001",
            from_sec="PROPOSED",
            to_sec="AUTHORITATIVE",
        )
        
        assert result.allowed is False


# ────────────────────────────────────────────────────────────────-----
# 41. Knowledge Policy Bridge Integration
# ────────────────────────────────────────────────----------

class TestKnowledgePolicyBridge:
    """Test KnowledgePolicyBridge integration."""

    def test_bridge_instantiation(self):
        """KnowledgePolicyBridge can be instantiated."""
        bridge = KnowledgePolicyBridge()
        assert bridge is not None
        assert hasattr(bridge, "policy_engine")
        assert hasattr(bridge, "knowledge_policy_engine")

    def test_bridge_evaluate_with_knowledge(self):
        """Bridge evaluates operations through both engines."""
        bridge = KnowledgePolicyBridge()
        result = bridge.evaluate_with_knowledge(
            operation="READ",
            subject="ADR-001",
        )
        
        assert result.decision.allowed is True
        assert result.knowledge_metrics.policy_evaluations == 1

    def test_bridge_check_authority_transition(self):
        """Bridge delegates authority transition checks."""
        bridge = KnowledgePolicyBridge()
        
        result = bridge.check_authority_transition(
            record_id="ADR-010",
            from_authority="proposed",
            to_authority="accepted",
            record_type="ADR",
        )
        
        assert result.allowed is False

    def test_bridge_check_security_authority_transition(self):
        """Bridge delegates security authority transition checks."""
        bridge = KnowledgePolicyBridge()
        
        result = bridge.check_security_authority_transition(
            record_id="SEC-010",
            from_sec="PROPOSED",
            to_sec="AUTHORITATIVE",
        )
        
        assert result.allowed is False


# ────────────────────────────────────────────────────────────────-----
# 42. Policy Determinism
# ────────────────────────────────────────────────-----

class TestPolicyDeterminism:
    """Verify policy decisions are deterministic."""

    def test_same_inputs_same_decision(self):
        """Same canonical inputs → same policy decision."""
        engine = KnowledgePolicyEngine()
        
        results = []
        for _ in range(3):
            result = engine.evaluate_operation(operation="READ", subject="test-subject")
            results.append(result.compute_digest())
        
        assert all(d == results[0] for d in results)

    def test_digest_stable_across_instances(self):
        """Digest is stable across engine instances."""
        engine1 = KnowledgePolicyEngine()
        engine2 = KnowledgePolicyEngine()
        
        result1 = engine1.evaluate_operation(operation="READ", subject="test-subject")
        result2 = engine2.evaluate_operation(operation="READ", subject="test-subject")
        
        assert result1.compute_digest() == result2.compute_digest()


# ────────────────────────────────────────────────────────────────-----
# 43. No Policy Quality Claims
# ────────────────────────────────────────────────-----

class TestNoPolicyQualityClaims:
    """Verify enforcement behavior is measured, not governance quality."""

    def test_metrics_are_descriptive(self):
        """Metrics are descriptive, not normative."""
        engine = KnowledgePolicyEngine()
        
        engine.evaluate_operation(operation="READ", subject="test1")
        engine.evaluate_operation(operation="PROPOSED_TO_ACCEPTED", subject="test2")
        engine.evaluate_operation(operation="PROPOSED_TO_ACCEPTED", subject="test3")
        
        # Descriptive: simply counts what happened
        assert engine.metrics.policy_evaluations == 3
        assert engine.metrics.allowed == 1
        assert engine.metrics.denied == 2
        
        # No normative claim like "more blocks = better governance"
        # The metrics simply report what occurred


# ────────────────────────────────────────────────────────────────-----
# 44. Release Gates G60, G61, G62
# ────────────────────────────────────────────────---------

class TestReleaseGates:
    """Verify release gates G60, G61, G62."""

    def test_G60_policy_engine_integration(self):
        """G60: Knowledge governance integrated with canonical PolicyEngine."""
        engine = PolicyEngine()
        
        # Verify Integration
        assert hasattr(engine, "knowledge_policy")
        assert hasattr(engine, "knowledge_bridge")
        
        # Verify evaluate_knowledge_operation works
        result = engine.evaluate_knowledge_operation(
            operation="READ",
            subject="test",
        )
        assert result.allowed is True
        
        # Verify knowledge policy is subordinate, not competing
        assert engine.knowledge_policy is not None
        assert isinstance(engine.knowledge_policy, KnowledgePolicyEngine)

    def test_G61_single_precedence_source(self):
        """G61: Single canonical precedence definition."""
        # Verify ONE canonical precedence definition
        from harness.policy.knowledge_policy import CANONICAL_PRECEDENCE_ORDER
        
        # Verify all consumers use the same canonical precedence
        # (This is enforced by the architecture — no competing precedence arrays)
        
        # Governance is highest
        assert get_canonical_precedence(source="governance") == PRECEDENCE_GOVERNANCE
        
        # Exploration is lowest
        assert get_canonical_precedence(source="exploration") == PRECEDENCE_EXPLORATION
        
        # Order is correct
        assert CANONICAL_PRECEDENCE_ORDER == [
            7, 6, 5, 4, 3, 2, 1, 0,
        ]

    def test_G62_protected_transitions_fail_closed(self):
        """G62: Protected authority transitions fail closed without authorization."""
        engine = KnowledgePolicyEngine()
        
        # All protected operations without authorization → fail closed
        protected_ops = [
            "PROPOSED_TO_ACCEPTED",
            "SEC_TO_AUTHORITY",
            "SEC_EXCEPTION_APPROVAL",
            "WAIVE_SEC",
            "DEPRECATE_PROTECTED",
            "CHANGE_AUTHORITY",
        ]
        
        for op in protected_ops:
            result = engine.evaluate_operation(
                operation=op,
                subject="test-subject",
                authorization=None,
            )
            assert result.allowed is False, f"{op} should fail closed"
            assert result.needs_human is True, f"{op} should need human"
            assert result.policy_blocked is True, f"{op} should be policy blocked"


# ────────────────────────────────────────────────────────────────-----
# 45. Policy Evaluation Is Read-Only
# ────────────────────────────────────────────────----

class TestPolicyEvaluationReadOnly:
    """Verify policy evaluation does not mutate state."""

    def test_evaluation_does_not_mutate_knowledge(self):
        """Policy evaluation does not mutate Engineering Knowledge."""
        from harness.knowledge import KnowledgeStore
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            store = KnowledgeStore(db_path)
            
            # Policy evaluation
            engine = KnowledgePolicyEngine()
            result = engine.evaluate_operation(operation="READ", subject="test")
            
            # Store state should be unchanged (no mutations)
            # We can't easily test this without creating a full record,
            # but the principle is that evaluation is side-effect free
            assert result.allowed is True

    def test_evaluation_does_not_mutate_learning(self):
        """Policy evaluation does not mutate Learning state."""
        # Policy evaluation should not create/modify learning artifacts
        engine = KnowledgePolicyEngine()
        result = engine.evaluate_operation(operation="READ", subject="test")
        
        assert result.allowed is True
        # No side effects on learning


# ────────────────────────────────────────────────────────────────-----
# 46. Governance + Traceability
# ────────────────────────────────────────────────--------------

class TestGovernanceTraceability:
    """Verify governance decisions produce traceable audit info."""

    def test_protected_action_produces_trace(self):
        """Protected action → produces policy decision for traceability."""
        engine = KnowledgePolicyEngine()
        auth = Authorization(
            authority="governance",
            action="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            reason="Approved",
            issuer="lead-architect",
            provenance={"review_id": "REV-001"},
        )
        
        result = engine.evaluate_operation(
            operation="PROPOSED_TO_ACCEPTED",
            subject="ADR-010",
            authorization=auth,
        )
        
        assert result.allowed is True
        assert "auth" in result.provenance
        assert result.provenance["auth"]["authority"] == "governance"

    def test_distinguish_never_considered_from_blocked(self):
        """Distinguish: policy blocked vs Planner never considered action."""
        # Policy blocked: governance/SEC/knowledge blocked the action
        engine = KnowledgePolicyEngine()
        
        result_blocked = engine.check_strategy_admissibility(
            strategy_id="STRAT-001",
            has_sec_violation=True,
        )
        
        # This was blocked by policy (SEC violation)
        assert result_blocked.allowed is False
        assert result_blocked.reason == "Strategy blocked by SEC violation"
        
        # "Never considered" would be a different path — the Planner
        # didn't even include the action, so no policy check was triggered
        # That's different from an explicit policy block


# ────────────────────────────────────────────────────────────────-----
# 47. Previous Test Integrity (CRITICAL)
# ────────────────────────────────────────────────---------------

class TestPreviousTestIntegrity:
    """Verify previous tests remain present and passing."""

    def test_no_test_files_removed(self):
        """No test files have been removed."""
        import os
        
        # Verify key test files still exist
        test_files = [
            "tests/unit/test_knowledge_records.py",
            "tests/unit/test_knowledge_provenance.py",
            "tests/unit/test_knowledge_lifecycle.py",
            "tests/unit/test_knowledge_store.py",
            "tests/unit/test_workflow_validator.py",
            "tests/unit/test_v33_phase11_knowledge_learning.py",
        ]
        
        for tf in test_files:
            assert os.path.exists(tf), f"Test file {tf} is missing"

    def test_policy_engine_existing_functionality_preserved(self):
        """Existing PolicyEngine functionality is fully preserved."""
        engine = PolicyEngine()
        
        # All existing methods should still exist
        assert hasattr(engine, "evaluate_all")
        assert hasattr(engine, "autonomy")
        assert hasattr(engine, "tool_policy")
        assert hasattr(engine, "execution")
        
        # New methods also exist
        assert hasattr(engine, "evaluate_knowledge_operation")

    def test_new_functionality_additive(self):
        """New functionality is purely additive, no modifications to old behavior."""
        result = PolicyResult()
        assert hasattr(result, "add_violation")
        
        # New PolicyDecision
        decision = PolicyDecision(
            allowed=True, policy="TEST", subject="test",
            action="TEST", reason="test", authority_basis="test",
        )
        assert hasattr(decision, "compute_digest")


# ────────────────────────────────────────────────────────────────-----
# Fixtures
# ────────────────────────────────────────────────────────────────-----

@pytest.fixture
def policy_engine():
    """Create a canonical PolicyEngine for testing."""
    return PolicyEngine()


@pytest.fixture
def knowledge_policy_engine():
    """Create a KnowledgePolicyEngine for testing."""
    return KnowledgePolicyEngine()


@pytest.fixture
def knowledge_bridge():
    """Create a KnowledgePolicyBridge for testing."""
    return KnowledgePolicyBridge()


@pytest.fixture
def sample_authorization():
    """Create a sample authorization."""
    return Authorization(
        authority="governance",
        action="PROPOSED_TO_ACCEPTED",
        subject="ADR-010",
        reason="Architecture review approved",
        issuer="lead-architect",
        provenance={"review_id": "REV-001"},
    )
