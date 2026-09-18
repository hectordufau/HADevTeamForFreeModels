# harness/policy/knowledge_policy.py — V3.3 Phase 12: Knowledge Policy Engine
"""
Subordinate domain policy component for Engineering Knowledge governance.

Phase 12 Implementation — TASK-067 (Governance + Engineering Knowledge Policies)

Architecture:
- KnowledgePolicyEngine is a SUBORDINATE domain policy component of the canonical
  PolicyEngine hierarchy. It does NOT create a competing authority structure.
- Single canonical precedence definition reused by ALL consumers:
  Governance > Security Authority > Authoritative Engineering Knowledge > 
  Accepted Engineering Knowledge > Task Requirements > Verified Evidence > 
  Learned Knowledge > Exploration

Canonical PolicyEngine hierarchy:
    PolicyEngine
    ├── Execution Policy
    ├── Tool Policy
    ├── Autonomy Policy
    └── Knowledge Policy (subordinate/domain component)

Critical invariants:
- PolicyEngine != KnowledgePolicyEngine competing independently
- KnowledgePolicy is a domain component, not an independent authority
- Protected authority transitions require explicit authorization
- No self-approval, no capability-based authority
- Agent-generated knowledge defaults to PROPOSED
- Cannot infer authority from: agent identity, role, model, model size, model confidence
- Cannot infer authority from: timestamp ordering, record_type alone, creator alone
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


class KnowledgePolicyError(Exception):
    """Raised on knowledge policy errors."""


class ProtectedTransitionError(KnowledgePolicyError):
    """Raised when a protected transition is attempted without authorization."""

    def __init__(self, record_id: str, from_state: str, to_state: str, reason: str):
        self.record_id = record_id
        self.from_state = from_state
        self.to_state = to_state
        self.reason = reason
        super().__init__(
            f"PROTECTED_TRANSITION_DENIED: Cannot transition '{record_id}' "
            f"from '{from_state}' to '{to_state}': {reason}"
        )


class AuthorizationRequiredError(KnowledgePolicyError):
    """Raised when explicit authorization is required but missing."""


# ──────────────────────────────────────────────────────────────────────
# Single Canonical Precedence Definition
# ──────────────────────────────────────────────────────────────────────

# This is THE canonical precedence source. All consumers MUST use this.
# Governance > Security Authority > Authoritative Engineering Knowledge > 
# Accepted Engineering Knowledge > Task Requirements > Verified Evidence > 
# Learned Knowledge > Exploration

PRECEDENCE_GOVERNANCE = 7
PRECEDENCE_SECURITY_AUTHORITY = 6
PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE = 5
PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE = 4
PRECEDENCE_TASK_REQUIREMENTS = 3
PRECEDENCE_VERIFIED_EVIDENCE = 2
PRECEDENCE_LEARNED_KNOWLEDGE = 1
PRECEDENCE_EXPLORATION = 0

# Ordered list of precedence levels (highest to lowest)
CANONICAL_PRECEDENCE_ORDER = [
    PRECEDENCE_GOVERNANCE,
    PRECEDENCE_SECURITY_AUTHORITY,
    PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE,
    PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE,
    PRECEDENCE_TASK_REQUIREMENTS,
    PRECEDENCE_VERIFIED_EVIDENCE,
    PRECEDENCE_LEARNED_KNOWLEDGE,
    PRECEDENCE_EXPLORATION,
]

PRECEDENCE_NAMES = {
    PRECEDENCE_GOVERNANCE: "Governance",
    PRECEDENCE_SECURITY_AUTHORITY: "Security Authority",
    PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE: "Authoritative Engineering Knowledge",
    PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE: "Accepted Engineering Knowledge",
    PRECEDENCE_TASK_REQUIREMENTS: "Task Requirements",
    PRECEDENCE_VERIFIED_EVIDENCE: "Verified Evidence",
    PRECEDENCE_LEARNED_KNOWLEDGE: "Learned Knowledge",
    PRECEDENCE_EXPLORATION: "Exploration",
}


def get_canonical_precedence(source: str = "", record_type: str = "", 
                              authority: str = "", **kwargs) -> int:
    """
    Get canonical precedence level — single source of truth.
    
    Args:
        source: "governance" | "security" | "engineering" | "task" | "learning" | "exploration"
        record_type: Engineering record type (PRD, NFR, DR, ADR, TDR, RSK, SEC, RCA, REQ)
        authority: Authority level (proposed, accepted, AUTHORITATIVE, etc.)
    
    Returns:
        Precedence integer (higher = more authoritative)
    
    This function is the ONLY way to determine precedence. No module should
    define its own precedence constants or ordering.
    """
    # Governance is absolute
    if source == "governance":
        return PRECEDENCE_GOVERNANCE
    
    # Security Authority (SEC records with protected security authority)
    if record_type == "SEC":
        # Map security authority levels to precedence
        sec_auth = authority.upper()
        if sec_auth == "AUTHORITATIVE":
            return PRECEDENCE_SECURITY_AUTHORITY
        elif sec_auth == "ACCEPTED":
            return PRECEDENCE_SECURITY_AUTHORITY
        elif sec_auth == "PROPOSED":
            return PRECEDENCE_TASK_REQUIREMENTS  # Proposed SEC is task-level
        elif sec_auth == "INFERRED":
            return PRECEDENCE_TASK_REQUIREMENTS
        elif sec_auth == "HISTORICAL":
            return PRECEDENCE_EXPLORATION
        elif sec_auth == "DEPRECATED":
            return PRECEDENCE_EXPLORATION
    
    # Source-based precedence (before record type checks)
    if source == "task":
        return PRECEDENCE_TASK_REQUIREMENTS
    if source == "learning":
        return PRECEDENCE_LEARNED_KNOWLEDGE
    if source == "exploration":
        return PRECEDENCE_EXPLORATION
    
    # Authoritative Engineering Knowledge (accepted ADR/DR)
    if record_type in ("ADR", "DR") and authority == "accepted":
        return PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE
    
    # Accepted Engineering Knowledge (accepted PRD, REQ, NFR, TDR, RSK, RCA)
    if authority == "accepted":
        return PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE
    
    # Verified Evidence (evidence-backed knowledge)
    if source == "evidence" or kwargs.get("has_evidence", False):
        return PRECEDENCE_VERIFIED_EVIDENCE
    
    # Task requirements (proposed engineering knowledge falls here)
    return PRECEDENCE_EXPLORATION


# ──────────────────────────────────────────────────────────────────────
# Policy Decision Model
# ──────────────────────────────────────────────────────────────────────

POLICY_ALLOW = "ALLOW"
POLICY_DENY = "DENY"
POLICY_NEEDS_HUMAN = "NEEDS_HUMAN"
POLICY_GOVERNANCE_CONFLICT = "GOVERNANCE_CONFLICT"
POLICY_KNOWLEDGE_CONFLICT = "KNOWLEDGE_CONFLICT"
POLICY_POLCY_BLOCKED = "POLICY_BLOCKED"


@dataclass
class PolicyDecision:
    """
    Structured policy decision — no bare booleans for release-critical decisions.
    
    Every protected operation returns a PolicyDecision with full provenance.
    """
    allowed: bool
    policy: str  # which policy was applied
    subject: str  # what the decision is about (record_id, operation, etc.)
    action: str  # what action was requested
    reason: str  # why this decision was made
    authority_basis: str  # what authority basis was used
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    needs_human: bool = False
    governance_conflict: bool = False
    knowledge_conflict: bool = False
    policy_blocked: bool = False
    
    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "policy": self.policy,
            "subject": self.subject,
            "action": self.action,
            "reason": self.reason,
            "authority_basis": self.authority_basis,
            "conflicts": list(self.conflicts),
            "provenance": dict(self.provenance),
            "needs_human": self.needs_human,
            "governance_conflict": self.governance_conflict,
            "knowledge_conflict": self.knowledge_conflict,
            "policy_blocked": self.policy_blocked,
        }
    
    def compute_digest(self) -> str:
        """Deterministic digest for reproducibility."""
        decision_dict = self.to_dict()
        digest_input = json.dumps(decision_dict, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# Authorization Provenance
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Authorization:
    """
    Explicit authorization for a protected transition.
    
    Protected transitions MUST carry this. Authorization cannot be inferred
    from: agent role, model, task success, review PASS.
    """
    authority: str  # what authority authorizes this (e.g., "governance", "human", "policy:P-v3")
    action: str  # what action is being authorized
    subject: str  # what is being acted upon
    reason: str  # why this authorization was granted
    provenance: Dict[str, Any] = field(default_factory=dict)  # who/what granted it
    issuer: str = ""  # who issued the authorization
    timestamp: str = ""  # when issued
    auth_id: str = ""  # unique authorization identifier
    
    def to_dict(self) -> dict:
        return {
            "authority": self.authority,
            "action": self.action,
            "subject": self.subject,
            "reason": self.reason,
            "provenance": dict(self.provenance),
            "issuer": self.issuer,
            "timestamp": self.timestamp,
            "auth_id": self.auth_id,
        }


# ──────────────────────────────────────────────────────────────────────
# Protected Operations
# ──────────────────────────────────────────────────────────────────────

# Operations that require explicit authorization
PROTECTED_OPERATIONS = {
    "PROPOSED_TO_ACCEPTED",
    "SEC_TO_AUTHORITY",
    "SEC_EXCEPTION_APPROVAL",
    "RISK_ACCEPTANCE_PROTECTED",
    "DEPRECATE_PROTECTED",
    "SUPERSEDE_PROTECTED",
    "CHANGE_AUTHORITY",
    "CHANGE_STATUS_PROTECTED",
    "WAIVE_SEC",
    "OVERRIDE_AUDIT",
}

# Operations that are always allowed for reading
READ_OPERATIONS = {
    "READ",
    "GET",
    "RETRIEVE",
    "SEARCH",
    "LIST",
    "QUERY",
    "TRACE",
    "CHECK_POLICY",  # Evaluating policy is always allowed
}

# Authority transitions that are protected
PROTECTED_AUTHORITY_TRANSITIONS = {
    # Standard authority transitions
    ("proposed", "accepted"),
    ("accepted", "deprecated"),
    ("deprecated", "archived"),
    # Security authority transitions
    ("PROPOSED", "ACCEPTED"),
    ("ACCEPTED", "AUTHORITATIVE"),
    ("INFERRED", "PROPOSED"),
    ("HISTORICAL", "DEPRECATED"),
}


def is_protected_operation(operation: str) -> bool:
    """Check if an operation requires explicit authorization."""
    return operation.upper() in PROTECTED_OPERATIONS


def is_read_operation(operation: str) -> bool:
    """Check if an operation is a read operation."""
    return operation.upper() in READ_OPERATIONS


# ──────────────────────────────────────────────────────────────────────
# Knowledge Policy Metrics
# ──────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgePolicyMetrics:
    """Deterministic descriptive metrics for knowledge policy enforcement."""
    policy_evaluations: int = 0
    allowed: int = 0
    denied: int = 0
    needs_human: int = 0
    knowledge_conflicts: int = 0
    governance_conflicts: int = 0
    protected_transition_attempts: int = 0
    protected_transition_denials: int = 0
    strategy_blocks: int = 0
    security_blocks: int = 0
    
    def to_dict(self) -> dict:
        return {
            "policy_evaluations": self.policy_evaluations,
            "allowed": self.allowed,
            "denied": self.denied,
            "needs_human": self.needs_human,
            "knowledge_conflicts": self.knowledge_conflicts,
            "governance_conflicts": self.governance_conflicts,
            "protected_transition_attempts": self.protected_transition_attempts,
            "protected_transition_denials": self.protected_transition_denials,
            "strategy_blocks": self.strategy_blocks,
            "security_blocks": self.security_blocks,
        }


# ──────────────────────────────────────────────────────────────────────
# KnowledgePolicyEngine — Subordinate Domain Policy Component
# ──────────────────────────────────────────────────────────────────────

class KnowledgePolicyEngine:
    """
    Subordinate domain policy component for Engineering Knowledge governance.
    
    This is NOT an independent PolicyEngine. It operates as a domain-specific
    policy component within the canonical PolicyEngine hierarchy.
    
    Responsibilities:
    - Evaluate knowledge authority transitions
    - Enforce protected operation authorization
    - Provide structured PolicyDecision (not bare booleans)
    - Maintain audit trail for protected operations
    - Detect governance conflicts and knowledge conflicts
    - Enforce: agent-generated knowledge defaults to PROPOSED
    - Enforce: no self-approval, no capability-based authority
    
    Architecture:
    - Wraps/extends the canonical PolicyEngine
    - Single canonical precedence source (get_canonical_precedence)
    - Fail closed: missing authorization → DENY
    - Deterministic: same inputs → same decision → same digest
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 authorization_callback: Optional[Any] = None):
        """
        Initialize the KnowledgePolicyEngine.
        
        Args:
            config: Knowledge policy configuration (optional)
            authorization_callback: Callback to validate authorization
        """
        self.config = config or {}
        self._auth_callback = authorization_callback
        self._metrics = KnowledgePolicyMetrics()
        self._audit_trail: List[Dict[str, Any]] = []
        self._policy_version = self.config.get("policy_version", "1.0.0")
        self._fail_closed_on_missing = True  # Default: DENY when auth context missing
    
    @property
    def metrics(self) -> KnowledgePolicyMetrics:
        """Get current metrics."""
        return self._metrics
    
    @property
    def audit_trail(self) -> List[Dict[str, Any]]:
        """Get audit trail (copy for immutability)."""
        return list(self._audit_trail)
    
    def evaluate_operation(
        self,
        operation: str,
        subject: str,
        record_type: str = "",
        authority: str = "",
        status: str = "",
        authorization: Optional[Authorization] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> PolicyDecision:
        """
        Evaluate a knowledge operation against policy.
        
        Args:
            operation: The operation being attempted
            subject: The subject of the operation (record_id, etc.)
            record_type: Engineering record type
            authority: Current authority level
            status: Current lifecycle status
            authorization: Explicit authorization (for protected operations)
            context: Additional context
            
        Returns:
            PolicyDecision with full provenance
        """
        self._metrics.policy_evaluations += 1
        
        # Read operations are always allowed
        if is_read_operation(operation):
            self._metrics.allowed += 1
            decision = PolicyDecision(
                allowed=True,
                policy="READ_OPERATION",
                subject=subject,
                action=operation,
                reason="Read operations are always allowed",
                authority_basis="read_access",
                provenance={"policy_version": self._policy_version},
            )
            self._audit(decision)
            return decision
        
        # Protected operations require explicit authorization
        if is_protected_operation(operation):
            return self._evaluate_protected_operation(
                operation=operation,
                subject=subject,
                record_type=record_type,
                authority=authority,
                status=status,
                authorization=authorization,
                context=context,
                **kwargs,
            )
        
        # Non-protected write operations → ALLOW
        self._metrics.allowed += 1
        return PolicyDecision(
            allowed=True,
            policy="WRITE_OPERATION",
            subject=subject,
            action=operation,
            reason="Non-protected write operation",
            authority_basis="write_access",
            provenance={"policy_version": self._policy_version},
        )
    
    def _evaluate_protected_operation(
        self,
        operation: str,
        subject: str,
        record_type: str,
        authority: str,
        status: str,
        authorization: Optional[Authorization],
        context: Optional[Dict[str, Any]],
        **kwargs,
    ) -> PolicyDecision:
        """Evaluate a protected operation."""
        self._metrics.protected_transition_attempts += 1
        
        # Check for explicit authorization
        if authorization is None or not authorization.authority:
            # FAIL CLOSED — no authorization → DENY
            self._metrics.protected_transition_denials += 1
            self._metrics.denied += 1
            
            decision = PolicyDecision(
                allowed=False,
                policy=f"PROTECTED:{operation}",
                subject=subject,
                action=operation,
                reason=f"Protected operation '{operation}' requires explicit authorization. No authorization provided.",
                authority_basis="none",
                conflicts=[],
                provenance={"policy_version": self._policy_version, "request_id": subject},
                needs_human=True,
                policy_blocked=True,
            )
            
            self._audit(decision)
            return decision
        
        # Authorization provided — validate it
        if not self._validate_authorization(authorization, operation, subject):
            self._metrics.protected_transition_denials += 1
            self._metrics.denied += 1
            
            decision = PolicyDecision(
                allowed=False,
                policy=f"PROTECTED:{operation}",
                subject=subject,
                action=operation,
                reason=f"Authorization validation failed for '{operation}' on '{subject}'.",
                authority_basis=authorization.authority,
                conflicts=[],
                provenance={"policy_version": self._policy_version, "auth": authorization.to_dict()},
                needs_human=True,
                policy_blocked=True,
            )
            
            self._audit(decision)
            return decision
        
        # Check for conflicts
        conflicts = self._detect_conflicts(operation, subject, record_type, authority, context)
        if conflicts:
            knowledge_conflicts = [c for c in conflicts if c.get("type") == "KNOWLEDGE_CONFLICT"]
            governance_conflicts = [c for c in conflicts if c.get("type") == "GOVERNANCE_CONFLICT"]
            
            if governance_conflicts:
                self._metrics.governance_conflicts += 1
                self._metrics.needs_human += 1
                
                decision = PolicyDecision(
                    allowed=False,
                    policy=f"PROTECTED:{operation}",
                    subject=subject,
                    action=operation,
                    reason="Governance conflict detected. Cannot be automatically resolved.",
                    authority_basis=authorization.authority,
                    conflicts=conflicts,
                    provenance={"policy_version": self._policy_version, "auth": authorization.to_dict()},
                    needs_human=True,
                    governance_conflict=True,
                )
                self._audit(decision)
                return decision
            
            if knowledge_conflicts:
                self._metrics.knowledge_conflicts += 1
                
                decision = PolicyDecision(
                    allowed=False,
                    policy=f"PROTECTED:{operation}",
                    subject=subject,
                    action=operation,
                    reason="Knowledge conflict detected. Resolution required.",
                    authority_basis=authorization.authority,
                    conflicts=conflicts,
                    provenance={"policy_version": self._policy_version, "auth": authorization.to_dict()},
                    needs_human=True,
                    knowledge_conflict=True,
                )
                self._audit(decision)
                return decision
        
        # All checks passed → ALLOW
        self._metrics.allowed += 1
        
        decision = PolicyDecision(
            allowed=True,
            policy=f"PROTECTED:{operation}",
            subject=subject,
            action=operation,
            reason=f"Protected operation '{operation}' authorized by {authorization.authority}.",
            authority_basis=authorization.authority,
            conflicts=[],
            provenance={
                "policy_version": self._policy_version,
                "auth": authorization.to_dict(),
                "authorization_timestamp": authorization.timestamp,
            },
        )
        
        self._audit(decision)
        return decision
    
    def _validate_authorization(self, auth: Authorization, operation: str, subject: str) -> bool:
        """Validate authorization for a protected operation."""
        # Check authorization has required fields
        if not auth.authority:
            return False
        if not auth.reason:
            return False
        
        # Check authorization has provenance
        if not auth.provenance and not auth.issuer:
            return False
        
        # Check authorization is not self-approval
        # (agent proposes and same agent approves)
        if auth.provenance.get("self_approval", False):
            return False
        
        return True
    
    def _detect_conflicts(
        self,
        operation: str,
        subject: str,
        record_type: str,
        authority: str,
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Detect knowledge and governance conflicts."""
        conflicts = []
        
        if not context:
            return conflicts
        
        # Check for opposing SEC constraints
        opposing_sec = context.get("opposing_sec", [])
        if opposing_sec:
            for sec in opposing_sec:
                conflicts.append({
                    "type": "GOVERNANCE_CONFLICT",
                    "record_id": sec,
                    "reason": f"Opposing SEC constraint: {sec}",
                })
        
        # Check for unresolved knowledge contradictions
        contradictions = context.get("contradictions", [])
        if contradictions:
            for c in contradictions:
                conflicts.append({
                    "type": "KNOWLEDGE_CONFLICT",
                    "record_id": c.get("record_id", ""),
                    "reason": c.get("reason", "Unresolved contradiction"),
                })
        
        # Check for authority conflicts (two equal authorities)
        competing_authorities = context.get("competing_authorities", [])
        if len(competing_authorities) > 1:
            conflicts.append({
                "type": "GOVERNANCE_CONFLICT",
                "record_id": subject,
                "reason": f"Multiple competing authorities: {competing_authorities}",
            })
        
        return conflicts
    
    def _audit(self, decision: PolicyDecision) -> None:
        """Add decision to audit trail."""
        self._audit_trail.append({
            "decision": decision.to_dict(),
            "digest": decision.compute_digest(),
            "policy_version": self._policy_version,
        })
    
    def check_authority_transition(
        self,
        record_id: str,
        from_authority: str,
        to_authority: str,
        record_type: str = "",
        authorization: Optional[Authorization] = None,
    ) -> PolicyDecision:
        """
        Check if an authority transition is allowed.
        
        Protected transitions require explicit authorization:
        - PROPOSED → ACCEPTED
        - SEC → Security Authority (any SEC → AUTHORITATIVE)
        - Security exception/override
        - Risk acceptance where protected
        - Knowledge deprecation
        - Supersession of protected records
        
        Args:
            record_id: The record being transitioned
            from_authority: Current authority level
            to_authority: Target authority level
            record_type: Engineering record type
            authorization: Explicit authorization
            
        Returns:
            PolicyDecision
        """
        transition_key = (from_authority, to_authority)
        is_protected = transition_key in PROTECTED_AUTHORITY_TRANSITIONS
        
        # Fail closed: unknown target authority values are denied
        from harness.knowledge.provenance import VALID_AUTHORITY_LEVELS
        from harness.knowledge.risk_security import VALID_SEC_AUTHORITIES
        known_authorities = set(VALID_AUTHORITY_LEVELS) | set(VALID_SEC_AUTHORITIES)
        if to_authority not in known_authorities:
            self._metrics.denied += 1
            return PolicyDecision(
                allowed=False,
                policy="AUTHORITY_TRANSITION",
                subject=record_id,
                action=f"AUTHORITY_TRANSITION:{from_authority}_TO_{to_authority}",
                reason=f"Unknown target authority '{to_authority}' — fail closed",
                authority_basis="unknown_authority",
                provenance={"policy_version": self._policy_version},
            )
        
        # Fail closed: transitions from terminal states are forbidden
        terminal_authorities = {"archived", "deprecated"}
        if from_authority in terminal_authorities:
            self._metrics.denied += 1
            return PolicyDecision(
                allowed=False,
                policy="AUTHORITY_TRANSITION",
                subject=record_id,
                action=f"AUTHORITY_TRANSITION:{from_authority}_TO_{to_authority}",
                reason=f"Cannot transition from terminal state '{from_authority}'",
                authority_basis="terminal_state",
                provenance={"policy_version": self._policy_version},
            )
        
        if is_protected:
            # Map specific transitions to recognized protected operation names
            if from_authority == "proposed" and to_authority == "accepted":
                protected_op = "PROPOSED_TO_ACCEPTED"
            elif from_authority == "accepted" and to_authority == "deprecated":
                protected_op = "DEPRECATE_PROTECTED"
            elif from_authority == "deprecated" and to_authority == "archived":
                protected_op = "DEPRECATE_PROTECTED"
            else:
                protected_op = "CHANGE_AUTHORITY"
            
            return self._evaluate_protected_operation(
                operation=protected_op,
                subject=record_id,
                record_type=record_type,
                authority=from_authority,
                status="",
                authorization=authorization,
                context=None,
            )
        
        # Non-protected transition → ALLOW
        self._metrics.allowed += 1
        return PolicyDecision(
            allowed=True,
            policy="AUTHORITY_TRANSITION",
            subject=record_id,
            action=f"AUTHORITY_TRANSITION:{from_authority}_TO_{to_authority}",
            reason=f"Non-protected transition from '{from_authority}' to '{to_authority}'",
            authority_basis="standard_transition",
            provenance={"policy_version": self._policy_version},
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
        Forbidden: agent creates SEC → automatically Security Authority,
        or Reviewer PASS → Security Authority.
        
        Args:
            record_id: The SEC record
            from_sec: Current security authority (PROPOSED, ACCEPTED, AUTHORITATIVE, etc.)
            to_sec: Target security authority
            authorization: Explicit authorization
            
        Returns:
            PolicyDecision
        """
        # Security authority transitions are ALWAYS protected
        return self.evaluate_operation(
            operation="SEC_TO_AUTHORITY",
            subject=record_id,
            record_type="SEC",
            authority=from_sec,
            authorization=authorization,
        )
    
    def check_strategy_admissibility(
        self,
        strategy_id: str,
        conflict_records: Optional[List[str]] = None,
        has_sec_violation: bool = False,
    ) -> PolicyDecision:
        """
        Check if a strategy is admissible given knowledge constraints.
        
        Strategy blocked by SEC violation or accepted ADR conflict → BLOCKED.
        
        Args:
            strategy_id: The strategy being checked
            conflict_records: Records the strategy conflicts with
            has_sec_violation: Whether strategy violates an SEC constraint
            
        Returns:
            PolicyDecision
        """
        if has_sec_violation:
            self._metrics.security_blocks += 1
            self._metrics.strategy_blocks += 1
            self._metrics.denied += 1
            
            return PolicyDecision(
                allowed=False,
                policy="STRATEGY_ADMISSIBILITY",
                subject=strategy_id,
                action="STRATEGY_EXECUTION",
                reason="Strategy blocked by SEC violation",
                authority_basis="Security Authority",
                conflicts=[{"type": "SEC_VIOLATION", "record_id": r} 
                           for r in (conflict_records or [])],
                provenance={"policy_version": self._policy_version},
            )
        
        if conflict_records:
            self._metrics.strategy_blocks += 1
            self._metrics.denied += 1
            
            return PolicyDecision(
                allowed=False,
                policy="STRATEGY_ADMISSIBILITY",
                subject=strategy_id,
                action="STRATEGY_EXECUTION",
                reason="Strategy conflicts with accepted knowledge",
                authority_basis="Accepted Engineering Knowledge",
                conflicts=[{"type": "KNOWLEDGE_CONFLICT", "record_id": r} 
                           for r in conflict_records],
                provenance={"policy_version": self._policy_version},
                knowledge_conflict=True,
            )
        
        # No conflicts → ADMISSIBLE
        self._metrics.allowed += 1
        return PolicyDecision(
            allowed=True,
            policy="STRATEGY_ADMISSIBILITY",
            subject=strategy_id,
            action="STRATEGY_EXECUTION",
            reason="Strategy is admissible",
            authority_basis="none",
            provenance={"policy_version": self._policy_version},
        )
    
    def validate_agent_generated_knowledge(
        self,
        record_type: str,
        authority: str = "proposed",
    ) -> PolicyDecision:
        """
        Validate agent-generated knowledge authority.
        
        Agent-generated Engineering Knowledge defaults to PROPOSED
        unless explicit policy authorizes another state.
        
        Args:
            record_type: Type of knowledge record
            authority: Attempted authority level
            
        Returns:
            PolicyDecision
        """
        if authority != "proposed":
            # Trying to create at non-PROPOSED level → DENY
            return PolicyDecision(
                allowed=False,
                policy="AGENT_GENERATED_KNOWLEDGE",
                subject=f"AGENT-{record_type}",
                action=f"CREATE_{record_type}",
                reason=f"Agent-generated {record_type} must default to PROPOSED. "
                       f"Attempted authority '{authority}' is not allowed.",
                authority_basis="agent_default",
                provenance={"policy_version": self._policy_version},
            )
        
        # PROPOSED is the default and allowed
        return PolicyDecision(
            allowed=True,
            policy="AGENT_GENERATED_KNOWLEDGE",
            subject=f"AGENT-{record_type}",
            action=f"CREATE_{record_type}",
            reason=f"Agent-generated {record_type} defaults to PROPOSED",
            authority_basis="agent_default",
            provenance={"policy_version": self._policy_version},
        )
    
    def validate_capability_vs_permission(
        self,
        operation: str,
        has_capability: bool,
        authorization: Optional[Authorization] = None,
    ) -> PolicyDecision:
        """
        Enforce: CAN != MAY — capability does not grant permission.
        
        An agent MAY have the technical capability to perform an operation
        (e.g., modify an ADR) but NOT have policy permission (e.g., accept an ADR).
        
        Args:
            operation: The operation being attempted
            has_capability: Whether agent has technical capability
            authorization: Explicit authorization
            
        Returns PolicyDecision
        """
        if not has_capability:
            return PolicyDecision(
                allowed=False,
                policy="CAPABILITY_THEN_PERMISSION",
                subject=operation,
                action=operation,
                reason="Agent lacks capability to perform this operation",
                authority_basis="none",
                provenance={"policy_version": self._policy_version},
            )
        
        # Has capability but needs authorization for protected operations
        if is_protected_operation(operation) and authorization is None:
            return PolicyDecision(
                allowed=False,
                policy="CAPABILITY_THEN_PERMISSION",
                subject=operation,
                action=operation,
                reason="Agent HAS capability but MAY NOT perform: no authorization",
                authority_basis="capability_without_permission",
                provenance={"policy_version": self._policy_version},
            )
        
        if is_protected_operation(operation) and authorization is not None:
            return PolicyDecision(
                allowed=True,
                policy="CAPABILITY_THEN_PERMISSION",
                subject=operation,
                action=operation,
                reason=f"Agent has capability and authorization from {authorization.authority}",
                authority_basis=authorization.authority,
                provenance={"policy_version": self._policy_version, "auth": authorization.to_dict()},
            )
        
        return PolicyDecision(
            allowed=True,
            policy="CAPABILITY_THEN_PERMISSION",
            subject=operation,
            action=operation,
            reason="Agent has capability and permission",
            authority_basis="standard",
            provenance={"policy_version": self._policy_version},
        )


# ──────────────────────────────────────────────────────────────────────
# Policy Configuration Validation
# ──────────────────────────────────────────────────────────────────────

def validate_policy_config(config: Dict[str, Any]) -> PolicyDecision:
    """
    Validate knowledge policy configuration.
    
    Checks:
    - Schema validity
    - Canonical precedence present
    - Protected operations defined
    - Digest configuration
    - Fail closed on invalid config
    
    Args:
        config: Policy configuration dictionary
        
    Returns:
        PolicyDecision (allowed=True if config is valid)
    """
    required_keys = ["policy_version", "precedence_order", "protected_operations"]
    
    for key in required_keys:
        if key not in config:
            return PolicyDecision(
                allowed=False,
                policy="CONFIG_VALIDATION",
                subject="policy_config",
                action="VALIDATE_CONFIG",
                reason=f"Missing required configuration key: '{key}'",
                authority_basis="none",
                provenance={"policy_version": "unknown"},
            )
    
    # Validate canonical precedence
    precedence = config.get("precedence_order", [])
    if precedence != CANONICAL_PRECEDENCE_ORDER:
        return PolicyDecision(
            allowed=False,
            policy="CONFIG_VALIDATION",
            subject="policy_config",
            action="VALIDATE_CONFIG",
            reason="Canonical precedence order mismatch",
            authority_basis="none",
            provenance={"policy_version": config.get("policy_version", "unknown")},
        )
    
    # Validate protected operations
    protected = set(config.get("protected_operations", []))
    if not protected.issubset(PROTECTED_OPERATIONS):
        return PolicyDecision(
            allowed=False,
            policy="CONFIG_VALIDATION",
            subject="policy_config",
            action="VALIDATE_CONFIG",
            reason=f"Unknown protected operations: {protected - PROTECTED_OPERATIONS}",
            authority_basis="none",
            provenance={"policy_version": config.get("policy_version", "unknown")},
        )
    
    # Compute config digest
    config_digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    
    return PolicyDecision(
        allowed=True,
        policy="CONFIG_VALIDATION",
        subject="policy_config",
        action="VALIDATE_CONFIG",
        reason="Policy configuration is valid",
        authority_basis="config_valid",
        provenance={
            "policy_version": config.get("policy_version", "unknown"),
            "config_digest": config_digest,
        },
    )


def compute_policy_digest(decision_inputs: Dict[str, Any]) -> str:
    """
    Compute deterministic policy decision digest.
    
    Uses canonical inputs: decision + policy config + authorization context + action.
    No Python hash(), no set order, no RNG, no timestamp.
    
    Args:
        decision_inputs: Canonical inputs to the policy decision
        
    Returns:
        SHA-256 hex digest
    """
    canonical = json.dumps(decision_inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
