# harness/knowledge/learning_bridge.py — V3.3 Phase 11: KnowledgeLearningBridge
"""
Bidirectional integration between Engineering Knowledge and Learning.

Phase 11 Implementation — TASK-065 (Knowledge <-> Learning Bidirectional Flow)

Architecture:
- KnowledgeLearningBridge is an INTEGRATION layer, NOT a canonical store
- Engineering Knowledge (KnowledgeStore) remains canonical for engineering records
- Learning (ExperienceStore, StrategyStore) remains canonical for learning artifacts
- Bridge uses stable references (record_ids), never duplicates canonical payloads
- Directional: Knowledge -> Learning (constraints), Learning -> Knowledge (signals)

Three-domain separation preserved:
- Engineering Knowledge: PRD, NFR, DR, ADR, TDR, RSK, SEC, RCA
- Learning Knowledge: StructuredExperience, FailureLesson, Strategy, DecisionImpact
- Evidence: EvidencePackage, hash chains

Critical invariants:
- Retrieval does NOT create Learning
- Influence alone does NOT create Learning
- Learning is grounded in observed/evaluated outcomes
- Engineering Knowledge constrains admissible strategies
- Learning optimizes only inside admissible space
- Learning confidence != Engineering authority
- Repeated experience != Engineering authority
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .records import EngineeringRecord
from .provenance import AUTHORITY_ACCEPTED, AUTHORITY_PROPOSED
from .risk_security import (
    PRECEDENCE_GOVERNANCE,
    PRECEDENCE_SECURITY_AUTHORITY,
    PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE,
    PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE,
    PRECEDENCE_TASK_REQUIREMENTS,
    PRECEDENCE_LEARNED_KNOWLEDGE,
    PRECEDENCE_EXPLORATION,
    VALID_SEC_AUTHORITIES,
    SEC_AUTHORITY_RANK,
)


class BridgeError(Exception):
    """Raised on bridge operation errors."""


class AuthorityConflictError(BridgeError):
    """Raised when learning conflicts with protected authority."""

    def __init__(self, strategy_id: str, record_id: str, reason: str):
        self.strategy_id = strategy_id
        self.record_id = record_id
        self.reason = reason
        super().__init__(
            f"AUTHORITY_CONFLICT: Strategy '{strategy_id}' blocked by "
            f"Engineering Record '{record_id}': {reason}"
        )


# ──────────────────────────────────────────────────────────────────────
# KnowledgeInfluence — provenance of knowledge on a decision
# ──────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeInfluence:
    """
    Provenance of an Engineering Record's influence on a decision.

    Captures: which record influenced, how, and with what authority.
    Does NOT claim causality — only that the record was present in context
    and contributed to the decision.
    """
    record_id: str
    record_type: str
    record_version: int = 1
    authority: str = ""
    decision: str = ""  # what decision was influenced
    effect: str = ""  # what effect the knowledge had
    reason: str = ""  # why this record was relevant
    precedence: int = 0

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "record_type": self.record_type,
            "record_version": self.record_version,
            "authority": self.authority,
            "decision": self.decision,
            "effect": self.effect,
            "reason": self.reason,
            "precedence": self.precedence,
        }


# ──────────────────────────────────────────────────────────────────────
# StrategyAdmissibility — result of strategy validation against knowledge
# ──────────────────────────────────────────────────────────────────────

ADMISSIBLE = "ADMISSIBLE"
CONSTRAINED = "CONSTRAINED"
BLOCKED = "BLOCKED"


@dataclass
class StrategyAdmissibility:
    """
    Result of validating a learned Strategy against Engineering Knowledge.

    ADMISSIBLE: strategy is compatible with all applicable knowledge
    CONSTRAINED: strategy is admissible but with constraints (e.g., TDR awareness)
    BLOCKED: strategy conflicts with protected knowledge (SEC, accepted ADR)
    """
    strategy_id: str
    status: str  # ADMISSIBLE | CONSTRAINED | BLOCKED
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "status": self.status,
            "conflicts": list(self.conflicts),
            "constraints": list(self.constraints),
            "provenance": dict(self.provenance),
        }


# ──────────────────────────────────────────────────────────────────────
# LearningProvenance — provenance for learning artifacts
# ──────────────────────────────────────────────────────────────────────

@dataclass
class LearningProvenance:
    """
    Provenance for a learning artifact (Experience, Strategy).

    Retains stable references to: task, plan, execution, knowledge context,
    knowledge influence IDs, review result, evidence refs.
    Does NOT copy complete EngineeringRecords.
    """
    task_id: str = ""
    plan_id: str = ""
    execution_id: str = ""
    knowledge_context_digest: str = ""
    knowledge_influence_ids: List[str] = field(default_factory=list)
    review_ref: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    decision_id: str = ""
    traceability_digest: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "plan_id": self.plan_id,
            "execution_id": self.execution_id,
            "knowledge_context_digest": self.knowledge_context_digest,
            "knowledge_influence_ids": list(self.knowledge_influence_ids),
            "review_ref": self.review_ref,
            "evidence_refs": list(self.evidence_refs),
            "decision_id": self.decision_id,
            "traceability_digest": self.traceability_digest,
        }


# ──────────────────────────────────────────────────────────────────────
# EngineeringKnowledgeSuggestion — learning-derived suggestion for Eng Knowledge
# ──────────────────────────────────────────────────────────────────────

SUGGESTION_STATUS_PROPOSED = "PROPOSED"
SUGGESTION_STATUS_CONFLICTING = "CONFLICTING"
SUGGESTION_STATUS_BLOCKED = "BLOCKED"
SUGGESTION_STATUS_NEEDS_REVIEW = "NEEDS_REVIEW"


@dataclass
class EngineeringKnowledgeSuggestion:
    """
    A suggestion for new Engineering Knowledge derived from Learning.

    Suggestions are ALWAYS proposals — never authoritative.
    They require protected review before becoming Engineering Records.
    Security-related suggestions can NEVER self-authorize.
    """
    suggestion_id: str
    suggestion_type: str  # "TDR", "RSK", "ADR", "SEC", "NFR", etc.
    title: str
    description: str
    source_experience_ids: List[str] = field(default_factory=list)
    source_failure_ids: List[str] = field(default_factory=list)
    source_strategy_ids: List[str] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0
    status: str = SUGGESTION_STATUS_PROPOSED
    conflict_with: Optional[str] = None  # record_id of conflicting knowledge
    authority_basis: str = ""  # why this suggestion is not authoritative

    def to_dict(self) -> dict:
        return {
            "suggestion_id": self.suggestion_id,
            "suggestion_type": self.suggestion_type,
            "title": self.title,
            "description": self.description,
            "source_experience_ids": list(self.source_experience_ids),
            "source_failure_ids": list(self.source_failure_ids),
            "source_strategy_ids": list(self.source_strategy_ids),
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "conflict_with": self.conflict_with,
            "authority_basis": self.authority_basis,
        }


# ──────────────────────────────────────────────────────────────────────
# KnowledgeLearningBridge — integration layer (NOT a canonical store)
# ──────────────────────────────────────────────────────────────────────

class KnowledgeLearningBridge:
    """
    Bidirectional integration between Engineering Knowledge and Learning.

    This is an INTEGRATION layer — NOT a canonical store.
    Engineering Knowledge remains canonical in KnowledgeStore.
    Learning remains canonical in ExperienceStore/StrategyStore.

    Responsibilities:
    - Engineering Knowledge -> Learning: constrain admissible strategies
    - Learning -> Engineering Knowledge: signals, suggestions, candidate concerns
    - Cross-domain lineage via stable references
    - Authority enforcement (Learning cannot override Engineering Knowledge)

    Directional integration:
    - Knowledge -> Learning: constrains admissible strategies, provides context
    - Learning -> Knowledge: signals, suggestions, candidate concerns
    - These are NOT symmetric authority operations
    """

    def __init__(self, knowledge_store: Any = None):
        """
        Initialize the bridge.

        Args:
            knowledge_store: KnowledgeStore instance (canonical Engineering Knowledge)
        """
        self.knowledge_store = knowledge_store
        self._suggestions: Dict[str, EngineeringKnowledgeSuggestion] = {}
        self._admissibility_cache: Dict[str, StrategyAdmissibility] = {}

    # ──────────────────────────────────────────────────────────────────
    # Knowledge -> Learning direction
    # ──────────────────────────────────────────────────────────────────

    def constrain_strategy(
        self,
        strategy: Any,
        knowledge_context: Any = None,
    ) -> StrategyAdmissibility:
        """
        Validate a learned Strategy against Engineering Knowledge.

        Before a learned Strategy influences planning, it must remain subject to:
        - Governance
        - Security Authority
        - Engineering Knowledge constraints
        - Task requirements

        Returns StrategyAdmissibility with status:
        - ADMISSIBLE: no conflicts with protected knowledge
        - CONSTRAINED: admissible but with constraints (e.g., TDR awareness)
        - BLOCKED: conflicts with protected knowledge (SEC, accepted ADR)

        Critical: Learning confidence != Engineering authority.
        Even confidence = 1.0 cannot turn a proposal into ACCEPTED/AUTHORITATIVE.
        """
        strategy_id = getattr(strategy, 'strategy_id', str(id(strategy)))
        conflicts: List[Dict[str, Any]] = []
        constraints: List[Dict[str, Any]] = []

        if knowledge_context is None:
            # No knowledge context — strategy is admissible by default
            return StrategyAdmissibility(
                strategy_id=strategy_id,
                status=ADMISSIBLE,
                provenance={"reason": "no knowledge context"},
            )

        # Check for SEC conflicts (highest priority)
        sec_conflicts = self._check_sec_conflicts(strategy, knowledge_context)
        if sec_conflicts:
            conflicts.extend(sec_conflicts)

        # Check for ADR conflicts
        adr_conflicts = self._check_adr_conflicts(strategy, knowledge_context)
        if adr_conflicts:
            conflicts.extend(adr_conflicts)

        # Check for TDR constraints (not blocking, but constraining)
        tdr_constraints = self._check_tdr_constraints(strategy, knowledge_context)
        if tdr_constraints:
            constraints.extend(tdr_constraints)

        # Check for RSK awareness (not blocking, but constraining)
        rsk_constraints = self._check_rsk_constraints(strategy, knowledge_context)
        if rsk_constraints:
            constraints.extend(rsk_constraints)

        # Determine status
        if conflicts:
            status = BLOCKED
        elif constraints:
            status = CONSTRAINED
        else:
            status = ADMISSIBLE

        result = StrategyAdmissibility(
            strategy_id=strategy_id,
            status=status,
            conflicts=conflicts,
            constraints=constraints,
            provenance={
                "checked_against": "engineering_knowledge",
                "knowledge_context_digest": getattr(knowledge_context, 'digest', ''),
            },
        )

        # Cache for idempotency
        self._admissibility_cache[strategy_id] = result
        return result

    def _check_sec_conflicts(
        self,
        strategy: Any,
        knowledge_context: Any,
    ) -> List[Dict[str, Any]]:
        """Check strategy against SEC constraints."""
        conflicts = []
        sec_items = self._get_items_by_type(knowledge_context, "SEC")

        for sec in sec_items:
            sec_id = sec.record_id
            sec_content = sec.content.lower() if sec.content else ""

            # Check if strategy violates SEC
            strategy_dict = strategy.to_dict() if hasattr(strategy, 'to_dict') else vars(strategy)
            strategy_str = json.dumps(strategy_dict).lower()
            capabilities = [c.lower() for c in strategy_dict.get('capabilities', [])]

            # TLS validation check — strategy disables TLS
            if "tls" in sec_content:
                # Check for tls_disabling, disable_tls, disable-tls, etc.
                tls_disable_patterns = ["tls_disabling", "disable_tls", "disable-tls", "disable tls"]
                has_tls_disable = any(p in strategy_str for p in tls_disable_patterns)
                # Also check capabilities list
                has_tls_disable_cap = any("tls" in c and "disabl" in c for c in capabilities)
                if has_tls_disable or has_tls_disable_cap:
                    conflicts.append({
                        "type": "SEC_VIOLATION",
                        "record_id": sec_id,
                        "reason": f"Strategy conflicts with SEC '{sec_id}': TLS validation mandatory",
                        "authority_basis": "SEC is protected authority",
                    })

            # Authentication check
            if "authentication" in sec_content and "skip" in strategy_str:
                conflicts.append({
                    "type": "SEC_VIOLATION",
                    "record_id": sec_id,
                    "reason": f"Strategy conflicts with SEC '{sec_id}': authentication required",
                    "authority_basis": "SEC is protected authority",
                })

        return conflicts

    def _check_adr_conflicts(
        self,
        strategy: Any,
        knowledge_context: Any,
    ) -> List[Dict[str, Any]]:
        """Check strategy against accepted ADR constraints."""
        conflicts = []
        adr_items = self._get_items_by_type(knowledge_context, "ADR")

        for adr in adr_items:
            if adr.authority != AUTHORITY_ACCEPTED:
                continue  # Only accepted ADRs constrain

            adr_id = adr.record_id
            adr_content = adr.content.lower() if adr.content else ""
            strategy_str = json.dumps(strategy.to_dict() if hasattr(strategy, 'to_dict') else vars(strategy)).lower()

            # Event-driven vs synchronous check
            if "event-driven" in adr_content and "synchronous" in strategy_str:
                conflicts.append({
                    "type": "ADR_CONFLICT",
                    "record_id": adr_id,
                    "reason": f"Strategy conflicts with accepted ADR '{adr_id}': event-driven integration required",
                    "authority_basis": "Accepted ADR is authoritative",
                })

        return conflicts

    def _check_tdr_constraints(
        self,
        strategy: Any,
        knowledge_context: Any,
    ) -> List[Dict[str, Any]]:
        """Check strategy against TDR constraints (constraining, not blocking)."""
        constraints = []
        tdr_items = self._get_items_by_type(knowledge_context, "TDR")

        for tdr in tdr_items:
            if tdr.status not in ("identified", "acknowledged", "remediation_planned"):
                continue  # Only active TDRs constrain

            constraints.append({
                "type": "TDR_AWARENESS",
                "record_id": tdr.record_id,
                "reason": f"Strategy should be aware of active TDR '{tdr.record_id}'",
                "severity": "medium",
            })

        return constraints

    def _check_rsk_constraints(
        self,
        strategy: Any,
        knowledge_context: Any,
    ) -> List[Dict[str, Any]]:
        """Check strategy against RSK awareness (constraining, not blocking)."""
        constraints = []
        rsk_items = self._get_items_by_type(knowledge_context, "RSK")

        for rsk in rsk_items:
            if rsk.status not in ("identified", "assessed", "mitigated", "accepted"):
                continue  # Only active RSKs constrain

            constraints.append({
                "type": "RISK_AWARENESS",
                "record_id": rsk.record_id,
                "reason": f"Strategy should retain mitigation for accepted risk '{rsk.record_id}'",
                "severity": "low",
            })

        return constraints

    def _get_items_by_type(self, knowledge_context: Any, record_type: str) -> List[Any]:
        """Get context items by record type."""
        if knowledge_context is None:
            return []
        items = getattr(knowledge_context, 'items', [])
        return [item for item in items if item.record_type == record_type]

    # ──────────────────────────────────────────────────────────────────
    # Learning -> Knowledge direction
    # ──────────────────────────────────────────────────────────────────

    def create_suggestion(
        self,
        suggestion_type: str,
        title: str,
        description: str,
        source_experience_ids: List[str],
        source_failure_ids: Optional[List[str]] = None,
        source_strategy_ids: Optional[List[str]] = None,
        reason: str = "",
        confidence: float = 0.0,
    ) -> EngineeringKnowledgeSuggestion:
        """
        Create a learning-derived suggestion for Engineering Knowledge.

        Suggestions are ALWAYS proposals — never authoritative.
        They require protected review before becoming Engineering Records.

        Security-related suggestions can NEVER self-authorize.
        """
        suggestion_id = f"SUG-{suggestion_type}-{hashlib.sha256(title.encode()).hexdigest()[:8]}"

        suggestion = EngineeringKnowledgeSuggestion(
            suggestion_id=suggestion_id,
            suggestion_type=suggestion_type,
            title=title,
            description=description,
            source_experience_ids=list(source_experience_ids),
            source_failure_ids=list(source_failure_ids or []),
            source_strategy_ids=list(source_strategy_ids or []),
            reason=reason,
            confidence=round(confidence, 4),
            status=SUGGESTION_STATUS_PROPOSED,
            authority_basis="Learning-derived suggestion requires protected review",
        )

        self._suggestions[suggestion_id] = suggestion
        return suggestion

    def check_suggestion_conflicts(
        self,
        suggestion: EngineeringKnowledgeSuggestion,
    ) -> EngineeringKnowledgeSuggestion:
        """
        Check a suggestion against existing Engineering Knowledge.

        If suggestion conflicts with existing accepted knowledge:
        - Existing knowledge wins
        - Suggestion status -> CONFLICTING / BLOCKED / NEEDS_REVIEW
        """
        if self.knowledge_store is None:
            return suggestion

        # Check for conflicts with existing records
        try:
            existing = self.knowledge_store.find_by_type(suggestion.suggestion_type)
            for record in existing:
                if record.authority == AUTHORITY_ACCEPTED:
                    # Check for content overlap (simple heuristic)
                    if self._content_overlap(suggestion.description, record.description):
                        suggestion.status = SUGGESTION_STATUS_CONFLICTING
                        suggestion.conflict_with = record.record_id
                        suggestion.authority_basis = (
                            f"Conflicts with accepted {suggestion.suggestion_type} '{record.record_id}'"
                        )
                        return suggestion
        except Exception:
            pass

        return suggestion

    def _content_overlap(self, text1: str, text2: str) -> bool:
        """Simple content overlap check."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap > 0.5

    # ──────────────────────────────────────────────────────────────────
    # Outcome-grounded learning
    # ──────────────────────────────────────────────────────────────────

    def build_learning_provenance(
        self,
        task_id: str,
        plan_id: str = "",
        execution_id: str = "",
        knowledge_context: Any = None,
        knowledge_influences: Optional[List[KnowledgeInfluence]] = None,
        review_ref: str = "",
        evidence_refs: Optional[List[str]] = None,
        decision_id: str = "",
        traceability_digest: str = "",
        knowledge_context_digest: str = "",
    ) -> LearningProvenance:
        """
        Build provenance for a learning artifact.

        Retains stable references to: task, plan, execution, knowledge context,
        knowledge influence IDs, review result, evidence refs.
        Does NOT copy complete EngineeringRecords.

        knowledge_context_digest can be passed directly (for restart scenarios
        where the full context is not available) or extracted from knowledge_context.
        """
        ctx_digest = knowledge_context_digest
        if not ctx_digest and knowledge_context is not None:
            ctx_digest = getattr(knowledge_context, 'digest', '')
        return LearningProvenance(
            task_id=task_id,
            plan_id=plan_id,
            execution_id=execution_id,
            knowledge_context_digest=ctx_digest,
            knowledge_influence_ids=[inf.record_id for inf in (knowledge_influences or [])],
            review_ref=review_ref,
            evidence_refs=list(evidence_refs or []),
            decision_id=decision_id,
            traceability_digest=traceability_digest,
        )

    # ──────────────────────────────────────────────────────────────────
    # Coverage metrics
    # ──────────────────────────────────────────────────────────────────

    def compute_coverage_metrics(
        self,
        decisions: List[Dict[str, Any]],
        experiences: List[Dict[str, Any]],
        strategies: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute learning coverage metrics.

        Metrics:
        - knowledge_influenced_decisions
        - knowledge_influenced_decisions_with_outcome
        - knowledge_influenced_decisions_with_experience
        - experiences_with_knowledge_provenance
        - strategies_with_knowledge_provenance
        - blocked_strategies_due_to_engineering_constraints
        """
        influenced = [d for d in decisions if d.get("knowledge_influenced", False)]
        with_outcome = [d for d in influenced if d.get("outcome") in ("IMPROVED", "UNCHANGED", "WORSENED")]
        with_experience = [d for d in influenced if d.get("experience_id")]

        exp_with_provenance = [e for e in experiences if e.get("knowledge_context_digest")]
        strat_with_provenance = [s for s in strategies if s.get("knowledge_provenance")]

        blocked = [s for s in strategies if s.get("admissibility_status") == BLOCKED]

        return {
            "knowledge_influenced_decisions": len(influenced),
            "knowledge_influenced_decisions_with_outcome": len(with_outcome),
            "knowledge_influenced_decisions_with_experience": len(with_experience),
            "experiences_with_knowledge_provenance": len(exp_with_provenance),
            "strategies_with_knowledge_provenance": len(strat_with_provenance),
            "blocked_strategies_due_to_engineering_constraints": len(blocked),
        }

    # ──────────────────────────────────────────────────────────────────
    # Constraint effect metrics
    # ──────────────────────────────────────────────────────────────────

    def compute_constraint_metrics(
        self,
        admissibility_results: List[StrategyAdmissibility],
    ) -> Dict[str, Any]:
        """
        Compute constraint effect metrics.

        Reports: strategies_evaluated, strategies_admissible, strategies_constrained,
        strategies_blocked by authority source.
        """
        total = len(admissibility_results)
        admissible = [r for r in admissibility_results if r.status == ADMISSIBLE]
        constrained = [r for r in admissibility_results if r.status == CONSTRAINED]
        blocked = [r for r in admissibility_results if r.status == BLOCKED]

        # Count blocked by authority source
        blocked_by_sec = 0
        blocked_by_adr = 0
        for r in blocked:
            for conflict in r.conflicts:
                if conflict.get("type") == "SEC_VIOLATION":
                    blocked_by_sec += 1
                elif conflict.get("type") == "ADR_CONFLICT":
                    blocked_by_adr += 1

        return {
            "strategies_evaluated": total,
            "strategies_admissible": len(admissible),
            "strategies_constrained": len(constrained),
            "strategies_blocked": len(blocked),
            "blocked_by_sec": blocked_by_sec,
            "blocked_by_adr": blocked_by_adr,
        }

    # ──────────────────────────────────────────────────────────────────
    # Idempotency
    # ──────────────────────────────────────────────────────────────────

    def get_cached_admissibility(self, strategy_id: str) -> Optional[StrategyAdmissibility]:
        """Get cached admissibility result for idempotency."""
        return self._admissibility_cache.get(strategy_id)

    def clear_cache(self) -> None:
        """Clear admissibility cache."""
        self._admissibility_cache.clear()
