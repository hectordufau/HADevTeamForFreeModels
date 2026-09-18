# harness/policy/knowledge_bridge.py — V3.3 Phase 12: Knowledge Policy Bridge
"""
Bridge between canonical PolicyEngine and KnowledgePolicyEngine.

Phase 12 Implementation — TASK-067 (Governance + Engineering Knowledge Policies)

Architecture:
- KnowledgePolicyEngine is a SUBORDINATE domain policy component
- It does NOT create an independent authority hierarchy
- PolicyEngine remains canonical and authoritative
- KnowledgePolicyEngine handles domain-specific knowledge policy decisions

Canonical hierarchy:
    PolicyEngine (canonical)
    ├── Execution Policy
    ├── Tool Policy
    ├── Autonomy Policy
    └── Knowledge Policy (subordinate/domain component)
         └── KnowledgePolicyEngine
              ├── Authority Transition Policy
              ├── Protected Operation Policy
              ├── Security Authority Policy
              ├── Strategy Admissibility Policy
              └── Agent-Generated Knowledge Policy

This bridge wires the KnowledgePolicyEngine into the PolicyEngine so that:
- PolicyEngine.evaluate_all() includes knowledge policy checks
- KnowledgePolicyEngine decisions are consistent with PolicyEngine
- Single canonical precedence definition is enforced
- Audit trail is unified
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from . import PolicyEngine, PolicyResult, PolicyViolation
from .knowledge_policy import (
    KnowledgePolicyEngine,
    PolicyDecision,
    Authorization,
    KnowledgePolicyMetrics,
    PROTECTED_OPERATIONS,
    READ_OPERATIONS,
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


class KnowledgePolicyBridgeError(Exception):
    """Raised on knowledge policy bridge errors."""


@dataclass
class KnowledgePolicyResult:
    """Result of knowledge policy evaluation within the canonical PolicyEngine."""
    decision: PolicyDecision
    knowledge_metrics: KnowledgePolicyMetrics
    effective_autonomy: int
    policy_violations: List[Dict[str, str]] = field(default_factory=list)
    knowledge_specific: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "decision": self.decision.to_dict(),
            "knowledge_metrics": self.knowledge_metrics.to_dict(),
            "effective_autonomy": self.effective_autonomy,
            "policy_violations": list(self.policy_violations),
            "knowledge_specific": dict(self.knowledge_specific),
        }


class KnowledgePolicyBridge:
    """
    Bridge between canonical PolicyEngine and KnowledgePolicyEngine.
    
    This is the integration point that ensures:
    - KnowledgePolicyEngine operates as a subordinate domain component
    - PolicyEngine remains canonical
    - Single source of truth for knowledge policy decisions
    - Unified audit trail
    """
    
    def __init__(self, policy_engine: Optional[PolicyEngine] = None,
                 knowledge_policy_engine: Optional[KnowledgePolicyEngine] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.policy_engine = policy_engine or PolicyEngine()
        self.knowledge_policy_engine = knowledge_policy_engine or KnowledgePolicyEngine(config)
        self.config = config or {}
    
    def evaluate_with_knowledge(
        self,
        operation: str,
        subject: str,
        record_type: str = "",
        authority: str = "",
        status: str = "",
        authorization: Optional[Authorization] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> KnowledgePolicyResult:
        """
        Evaluate an operation through both PolicyEngine and KnowledgePolicyEngine.
        
        Returns combined result with knowledge-specific decision.
        
        Args:
            operation: The operation being attempted
            subject: The subject of the operation
            record_type: Engineering record type
            authority: Current authority level
            status: Current lifecycle status
            authorization: Explicit authorization (for protected operations)
            context: Additional context
            
        Returns:
            KnowledgePolicyResult combining both engines' decisions
        """
        # First, evaluate through canonical PolicyEngine
        policy_result = self.policy_engine.evaluate_all(
            **{k: v for k, v in kwargs.items() if k in [
                "agent_level", "task_level", "tool_level",
                "tool_name", "current_autonomy",
                "current_iteration", "has_verification",
            ]}
        )
        
        # Then, evaluate through KnowledgePolicyEngine
        knowledge_decision = self.knowledge_policy_engine.evaluate_operation(
            operation=operation,
            subject=subject,
            record_type=record_type,
            authority=authority,
            status=status,
            authorization=authorization,
            context=context,
        )
        
        # Combine violations
        all_violations = list(policy_result.violations)
        if not knowledge_decision.allowed:
            all_violations.append({
                "policy": f"knowledge:{knowledge_decision.policy}",
                "reason": knowledge_decision.reason,
            })
        
        # Combined result
        effective = policy_result.effective_autonomy
        
        return KnowledgePolicyResult(
            decision=knowledge_decision,
            knowledge_metrics=self.knowledge_policy_engine.metrics,
            effective_autonomy=effective,
            policy_violations=all_violations,
            knowledge_specific={
                "knowledge_allowed": knowledge_decision.allowed,
                "knowledge_conflict": knowledge_decision.knowledge_conflict,
                "governance_conflict": knowledge_decision.governance_conflict,
                "needs_human": knowledge_decision.needs_human,
                "policy_blocked": knowledge_decision.policy_blocked,
            },
        )
    
    def check_authority_transition(
        self,
        record_id: str,
        from_authority: str,
        to_authority: str,
        record_type: str = "",
        authorization: Optional[Authorization] = None,
    ) -> PolicyDecision:
        """
        Check an authority transition through KnowledgePolicyEngine.
        
        This is the single entry point for authority transition checks.
        
        Args:
            record_id: The record being transitioned
            from_authority: Current authority level
            to_authority: Target authority level
            record_type: Engineering record type
            authorization: Explicit authorization
            
        Returns:
            PolicyDecision
        """
        return self.knowledge_policy_engine.check_authority_transition(
            record_id=record_id,
            from_authority=from_authority,
            to_authority=to_authority,
            record_type=record_type,
            authorization=authorization,
        )
    
    def check_security_authority_transition(
        self,
        record_id: str,
        from_sec: str,
        to_sec: str,
        authorization: Optional[Authorization] = None,
    ) -> PolicyDecision:
        """
        Check a security authority transition.
        
        SEC Authority transitions MUST require protected authorization.
        
        Args:
            record_id: The SEC record
            from_sec: Current security authority
            to_sec: Target security authority
            authorization: Explicit authorization
            
        Returns:
            PolicyDecision
        """
        return self.knowledge_policy_engine.check_security_authority_transition(
            record_id=record_id,
            from_sec=from_sec,
            to_sec=to_sec,
            authorization=authorization,
        )


# ──────────────────────────────────────────────────────────────────────
# Convenience functions for common knowledge policy operations
# ──────────────────────────────────────────────────────────────────────

def create_authorization(
    authority: str,
    action: str,
    subject: str,
    reason: str,
    issuer: str = "",
    provenance: Optional[Dict[str, Any]] = None,
    timestamp: str = "",
    auth_id: str = "",
) -> Authorization:
    """
    Create an Authorization for a protected operation.
    
    Args:
        authority: What authority authorizes this
        action: What action is being authorized
        subject: What is being acted upon
        reason: Why this authorization was granted
        issuer: Who issued the authorization
        provenance: Additional provenance
        timestamp: When issued
        auth_id: Unique authorization identifier
        
    Returns:
        Authorization instance
    """
    return Authorization(
        authority=authority,
        action=action,
        subject=subject,
        reason=reason,
        provenance=provenance or {},
        issuer=issuer,
        timestamp=timestamp,
        auth_id=auth_id,
    )


def check_self_approval(
    authorization: Authorization,
    actor: str,
    target: str,
) -> bool:
    """
    Check if an authorization is a self-approval (which is forbidden).
    
    Self-approval: agent proposes a protected transition and the same agent
    approves it. This is forbidden.
    
    Args:
        authorization: The authorization to check
        actor: The actor attempting the operation
        target: The target of the operation
        
    Returns:
        True if this is a self-approval (forbidden)
    """
    # Self-approval: the issuer (who grants authorization) is the same as
    # the actor (who attempts the operation)
    if not authorization.issuer:
        return False  # Empty issuer — can't determine self-approval
    
    return authorization.issuer == actor


def evaluate_adr_acceptance(
    adr_id: str,
    proposed_by: str,
    reviewer_result: str = "",
    authorization: Optional[Authorization] = None,
) -> PolicyDecision:
    """
    Evaluate whether an ADR can be accepted.

    Rules:
    - Agent-generated ADR remains PROPOSED until authorized
    - Reviewer PASS ≠ authority to accept ADR
    - Planner cannot accept ADR merely because it wants to use it
    - Learning cannot accept ADR because strategy performed well

    Args:
        adr_id: ADR being considered for acceptance
        proposed_by: Who proposed the ADR
        reviewer_result: Review result (PASS, FAIL, etc.) — irrelevant for acceptance
        authorization: Explicit authorization to accept

    Returns:
        PolicyDecision
    """
    engine = KnowledgePolicyEngine()

    # If no authorization and agent-generated, deny (remains PROPOSED)
    if authorization is None and proposed_by.startswith("AGENT"):
        decision = engine.validate_agent_generated_knowledge(
            record_type="ADR",
            authority="accepted",  # Attempting to set as accepted
        )
        if not decision.allowed:
            return decision

    # Check authorization for acceptance
    return engine.evaluate_operation(
        operation="PROPOSED_TO_ACCEPTED",
        subject=adr_id,
        record_type="ADR",
        authority="proposed",
        authorization=authorization,
    )


def evaluate_sec_authority_grant(
    sec_id: str,
    requested_authority: str,
    current_authority: str = "PROPOSED",
    authorization: Optional[Authorization] = None,
) -> PolicyDecision:
    """
    Evaluate whether a SEC record can be granted security authority.
    
    Rules:
    - Agent creates SEC → NOT Security Authority (stays PROPOSED)
    - Attempt self-escalation → DENIED
    - Protected authorization → allowed according to policy
    
    Args:
        sec_id: The SEC record
        requested_authority: Target security authority
        current_authority: Current security authority
        authorization: Explicit authorization
        
    Returns:
        PolicyDecision
    """
    engine = KnowledgePolicyEngine()
    return engine.check_security_authority_transition(
        record_id=sec_id,
        from_sec=current_authority,
        to_sec=requested_authority,
        authorization=authorization,
    )


def evaluate_tdr_resolution(
    tdr_id: str,
    target_status: str,
    resolution_evidence: Optional[List[str]] = None,
    authorization: Optional[Authorization] = None,
) -> PolicyDecision:
    """
    Evaluate whether a TDR can be transitioned to a new status.
    
    Rules:
    - TDR lifecycle rules remain
    - Resolution requires evidence where defined
    - Reviewer evidence may support resolution, but evidence exists != automatic transition
    
    Args:
        tdr_id: The TDR record
        target_status: Target lifecycle status
        resolution_evidence: Evidence records supporting the transition
        authorization: Explicit authorization
        
    Returns:
        PolicyDecision
    """
    engine = KnowledgePolicyEngine()
    
    # RESOLVED status requires evidence
    if target_status in ("resolved", "closed"):
        if not resolution_evidence:
            return PolicyDecision(
                allowed=False,
                policy="TDR_RESOLUTION",
                subject=tdr_id,
                action=f"RESOLVE_{target_status}",
                reason=f"TDR '{tdr_id}' cannot be resolved without resolution evidence",
                authority_basis="evidence_requirement",
                provenance={"policy_version": engine._policy_version},
            )
    
    # Authority transition check
    op = f"STATUS_CHANGE:{target_status}"
    return engine.evaluate_operation(
        operation=op,
        subject=tdr_id,
        record_type="TDR",
        authorization=authorization,
    )
