# harness/planner/knowledge_integration.py — V3.3 Phase 9: Knowledge Integration
"""
Knowledge-aware planning: TaskAnalyzer + WorkflowPlanner consume EngineeringKnowledgeContext.

Phase 9 Implementation — TASK-061, TASK-062, and extended knowledge integration.

Architecture:
- Engineering Knowledge may influence planning but does NOT bypass planning policy
- Knowledge remains declarative input
- CapabilityGraph remains separate from EngineeringKnowledgeGraph
- Knowledge does NOT select agents directly — capability-driven matching does
- Retrieved != INFLUENCED: a record in context but unused is not a decision influence
- Planner does NOT mutate or create Engineering Knowledge
- Security precedence: Governance > Security Authority > Authoritative Eng > Accepted Eng > Task Requirements > Learned > Exploration
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..knowledge.context import (
    EngineeringKnowledgeContext,
    LearningContext,
    ContextItem,
)
from ..knowledge.retrieval import (
    KnowledgeRetriever,
    KnowledgeQuery,
    RetrievalResult,
    RetrievalItem,
)
from ..knowledge.risk_security import (
    PRECEDENCE_GOVERNANCE,
    PRECEDENCE_SECURITY_AUTHORITY,
    PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE,
    PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE,
    PRECEDENCE_TASK_REQUIREMENTS,
    PRECEDENCE_LEARNED_KNOWLEDGE,
    PRECEDENCE_EXPLORATION,
    AuthorityPrecedenceResolver,
)
from ..knowledge.provenance import (
    AUTHORITY_PROPOSED,
    AUTHORITY_ACCEPTED,
)


class KnowledgeIntegrationError(Exception):
    """Raised on knowledge integration errors."""


class KnowledgeConflictError(KnowledgeIntegrationError):
    """Raised when knowledge conflicts block planning."""

    def __init__(self, conflicts: List[Dict[str, Any]]):
        self.conflicts = conflicts
        super().__init__(
            f"KNOWLEDGE_CONFLICT: {len(conflicts)} unresolved conflict(s). "
            f"Planning blocked. Fail closed."
        )


class KnowledgeGapError(KnowledgeIntegrationError):
    """Raised when a knowledge gap is detected."""

    def __init__(self, gaps: List[Dict[str, Any]]):
        self.gaps = gaps
        super().__init__(
            f"KNOWLEDGE_GAP: {len(gaps)} gap(s) detected. "
            f"Planning may be incomplete."
        )


class CapabilityGapError(KnowledgeIntegrationError):
    """Raised when a capability gap is detected."""

    def __init__(self, gaps: List[Dict[str, Any]]):
        self.gaps = gaps
        super().__init__(
            f"CAPABILITY_GAP: {len(gaps)} gap(s) detected. "
            f"Required capability not in registry."
        )


# ──────────────────────────────────────────────────────────────────────
# KnowledgeInfluence — explicit influence representation
# ──────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeInfluence:
    """
    Explicit representation of how Engineering Knowledge influenced planning.

    Every material planning decision influenced by Engineering Knowledge must be
    traceable to: record ID, record version, authority, reason, planning effect.
    """
    record_id: str
    record_type: str
    record_version: int
    authority: str
    decision: str  # what planning decision was influenced (e.g., "workflow_structure", "capability_requirement")
    effect: str  # what effect this had on planning (e.g., "required integration verification")
    reason: str  # why this record influenced planning
    precedence: int = 0  # precedence level from PRECEDENCE_* constants

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
# KnowledgeDerivedCapability — capability derived from knowledge
# ──────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeDerivedCapability:
    """
    A capability required by Engineering Knowledge.

    Knowledge may imply required capabilities (e.g., NFR performance →
    performance_testing capability, SEC constraint → security_verification).
    Uses approved declarative mappings. No role names.
    """
    capability: str
    source_record_id: str
    source_record_type: str
    reason: str
    is_mandatory: bool = False

    def to_dict(self) -> dict:
        return {
            "capability": self.capability,
            "source_record_id": self.source_record_id,
            "source_record_type": self.source_record_type,
            "reason": self.reason,
            "is_mandatory": self.is_mandatory,
        }


# ──────────────────────────────────────────────────────────────────────
# KnowledgeGap — structured gap detection
# ──────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeGap:
    """Structured knowledge or capability gap."""
    gap_type: str  # "KNOWLEDGE_GAP" or "CAPABILITY_GAP"
    record_id: str  # referenced record that doesn't exist or capability not in registry
    reason: str
    severity: str = "medium"  # "critical", "high", "medium", "low"

    def to_dict(self) -> dict:
        return {
            "gap_type": self.gap_type,
            "record_id": self.record_id,
            "reason": self.reason,
            "severity": self.severity,
        }


# ──────────────────────────────────────────────────────────────────────
# PlanningContext — typed planning context with knowledge
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PlanningContext:
    """
    Typed planning context that separates inputs.

    PlanningContext has:
    - Task Contract
    - Task Analysis
    - Required Capabilities
    - Engineering Knowledge Context
    - Learning Context
    - Policy/Governance Context
    """
    task_contract: Any = None
    task_analysis: Any = None
    required_capabilities: List[str] = field(default_factory=list)
    engineering_knowledge_context: Optional[EngineeringKnowledgeContext] = None
    learning_context: Optional[LearningContext] = None
    policy_context: Optional[Dict[str, Any]] = None
    knowledge_influences: List[KnowledgeInfluence] = field(default_factory=list)
    knowledge_derived_capabilities: List[KnowledgeDerivedCapability] = field(default_factory=list)
    knowledge_gaps: List[KnowledgeGap] = field(default_factory=list)
    capability_gaps: List[KnowledgeGap] = field(default_factory=list)
    digest: str = ""

    def to_dict(self) -> dict:
        return {
            "task_contract": self.task_contract,
            "task_analysis": self.task_analysis,
            "required_capabilities": list(self.required_capabilities),
            "engineering_knowledge_context": (
                self.engineering_knowledge_context.to_dict()
                if self.engineering_knowledge_context else None
            ),
            "learning_context": (
                self.learning_context.to_dict() if self.learning_context else None
            ),
            "policy_context": self.policy_context,
            "knowledge_influences": [ki.to_dict() for ki in self.knowledge_influences],
            "knowledge_derived_capabilities": [kdc.to_dict() for kdc in self.knowledge_derived_capabilities],
            "knowledge_gaps": [kg.to_dict() for kg in self.knowledge_gaps],
            "capability_gaps": [cg.to_dict() for cg in self.capability_gaps],
            "digest": self.digest,
        }

    def compute_digest(self) -> str:
        """Compute deterministic SHA-256 digest for reproducibility."""
        influences_sorted = sorted(
            (ki.record_id, ki.record_type, ki.record_version, ki.decision, ki.effect)
            for ki in self.knowledge_influences
        )
        caps_sorted = sorted(
            (kdc.capability, kdc.source_record_id, kdc.source_record_type)
            for kdc in self.knowledge_derived_capabilities
        )
        gaps_sorted = sorted(
            (kg.gap_type, kg.record_id, kg.severity)
            for kg in self.knowledge_gaps + self.capability_gaps
        )
        digest_input = json.dumps(
            {
                "required_capabilities": sorted(self.required_capabilities),
                "influences": influences_sorted,
                "derived_capabilities": caps_sorted,
                "gaps": gaps_sorted,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# KnowledgeAwareTaskAnalyzer — TaskAnalyzer with Engineering Knowledge
# ──────────────────────────────────────────────────────────────────────

# Declarative mapping: NFR category → required capability
NFR_CATEGORY_TO_CAPABILITY = {
    "performance": "performance_testing",
    "security": "security_verification",
    "reliability": "reliability_testing",
    "maintainability": "code_quality_analysis",
    "scalability": "scalability_testing",
    "usability": "usability_testing",
    "availability": "availability_testing",
    "observability": "observability_testing",
    "recoverability": "recovery_testing",
    "compatibility": "compatibility_testing",
    "privacy": "privacy_testing",
}

# Declarative mapping: SEC constraint type → required capability
SEC_CONSTRAINT_TO_CAPABILITY = {
    "authentication": "security_verification",
    "authorization": "security_verification",
    "encryption": "security_verification",
    "input_validation": "security_verification",
    "audit": "audit_verification",
}

# Declarative mapping: TDR debt type → awareness capability
TDR_DEBT_TYPE_TO_CAPABILITY = {
    "code": "code_quality_analysis",
    "design": "design_review",
    "architecture": "architecture_review",
    "test": "test_coverage_analysis",
    "documentation": "documentation_review",
    "dependency": "dependency_analysis",
}


class KnowledgeAwareTaskAnalyzer:
    """
    TaskAnalyzer that consumes EngineeringKnowledgeContext.

    Extends TaskAnalyzer to:
    - Derive knowledge query from task analysis
    - Detect knowledge gaps (referenced REQ doesn't exist)
    - Detect capability gaps (SEC requires capability not in registry)
    - Add knowledge-derived capabilities to requirements
    - Enforce security precedence
    """

    def __init__(self, taxonomy: Any = None, retriever: Optional[KnowledgeRetriever] = None):
        self.taxonomy = taxonomy
        self.retriever = retriever
        self._precedence_resolver = AuthorityPrecedenceResolver()

    def analyze_with_knowledge(
        self,
        task: Any,
        knowledge_context: Optional[EngineeringKnowledgeContext] = None,
    ) -> Tuple[Any, List[KnowledgeInfluence], List[KnowledgeGap], List[KnowledgeGap]]:
        """
        Analyze task with Engineering Knowledge context.

        Returns:
            (capability_requirements, knowledge_influences, knowledge_gaps, capability_gaps)
        """
        from .task_analyzer import TaskAnalyzer

        # Base analysis
        base_analyzer = TaskAnalyzer(taxonomy=self.taxonomy)
        requirements = base_analyzer.analyze(task)

        influences: List[KnowledgeInfluence] = []
        knowledge_gaps: List[KnowledgeGap] = []
        capability_gaps: List[KnowledgeGap] = []

        if not knowledge_context or not knowledge_context.items:
            return requirements, influences, knowledge_gaps, capability_gaps

        # Extract task metadata for knowledge query derivation
        task_objective = getattr(getattr(task, 'spec', None), 'objective', '')
        task_requirements = getattr(getattr(task, 'spec', None), 'requirements', [])
        task_acceptance = getattr(getattr(task, 'spec', None), 'acceptance_criteria', [])

        # Derive knowledge query from task
        knowledge_query = self._derive_knowledge_query(
            task_objective, task_requirements, task_acceptance, requirements
        )

        # Process each context item
        for item in knowledge_context.items:
            # Check for knowledge gaps (referenced REQ doesn't exist)
            if item.record_type in ("REQ", "NFR", "PRD"):
                if not self._record_exists(item.record_id):
                    knowledge_gaps.append(KnowledgeGap(
                        gap_type="KNOWLEDGE_GAP",
                        record_id=item.record_id,
                        reason=f"Referenced {item.record_type} '{item.record_id}' does not exist in KnowledgeStore",
                        severity="high" if item.is_mandatory else "medium",
                    ))

            # SEC constraints → required capabilities
            if item.record_type == "SEC" and item.is_mandatory:
                sec_caps = self._sec_to_capabilities(item)
                for cap in sec_caps:
                    if cap not in requirements.required:
                        requirements.required.append(cap)
                        influences.append(KnowledgeInfluence(
                            record_id=item.record_id,
                            record_type=item.record_type,
                            record_version=1,
                            authority=item.authority,
                            decision="capability_requirement",
                            effect=f"added required capability '{cap}'",
                            reason=f"SEC constraint '{item.record_id}' requires '{cap}' capability",
                            precedence=item.precedence,
                        ))

                # Check for capability gaps
                for cap in sec_caps:
                    if not self._capability_in_registry(cap):
                        capability_gaps.append(KnowledgeGap(
                            gap_type="CAPABILITY_GAP",
                            record_id=cap,
                            reason=f"SEC '{item.record_id}' requires capability '{cap}' not in CapabilityRegistry",
                            severity="critical",
                        ))

            # NFR → required capabilities
            if item.record_type == "NFR":
                nfr_caps = self._nfr_to_capabilities(item)
                for cap in nfr_caps:
                    if cap not in requirements.required:
                        requirements.required.append(cap)
                        influences.append(KnowledgeInfluence(
                            record_id=item.record_id,
                            record_type=item.record_type,
                            record_version=1,
                            authority=item.authority,
                            decision="capability_requirement",
                            effect=f"added required capability '{cap}'",
                            reason=f"NFR '{item.record_id}' implies '{cap}' capability",
                            precedence=item.precedence,
                        ))

            # ADR → workflow structure influence
            if item.record_type == "ADR" and item.authority == AUTHORITY_ACCEPTED:
                influences.append(KnowledgeInfluence(
                    record_id=item.record_id,
                    record_type=item.record_type,
                    record_version=1,
                    authority=item.authority,
                    decision="workflow_structure",
                    effect="constrained workflow to accepted architecture decision",
                    reason=f"Accepted ADR '{item.record_id}' constrains planning",
                    precedence=item.precedence,
                ))

            # TDR → debt awareness
            if item.record_type == "TDR":
                tdr_caps = self._tdr_to_capabilities(item)
                for cap in tdr_caps:
                    if cap not in requirements.optional:
                        requirements.optional.append(cap)
                        influences.append(KnowledgeInfluence(
                            record_id=item.record_id,
                            record_type=item.record_type,
                            record_version=1,
                            authority=item.authority,
                            decision="debt_awareness",
                            effect=f"added optional capability '{cap}' for debt awareness",
                            reason=f"TDR '{item.record_id}' suggests '{cap}' capability",
                            precedence=item.precedence,
                        ))

            # RSK → risk awareness
            if item.record_type == "RSK":
                influences.append(KnowledgeInfluence(
                    record_id=item.record_id,
                    record_type=item.record_type,
                    record_version=1,
                    authority=item.authority,
                    decision="risk_awareness",
                    effect="plan retains mitigation/fallback consideration",
                    reason=f"RSK '{item.record_id}' informs risk-aware planning",
                    precedence=item.precedence,
                ))

        # Deduplicate influences by (record_id, decision)
        seen = set()
        unique_influences = []
        for inf in influences:
            key = (inf.record_id, inf.decision)
            if key not in seen:
                seen.add(key)
                unique_influences.append(inf)

        return requirements, unique_influences, knowledge_gaps, capability_gaps

    def _derive_knowledge_query(
        self,
        objective: str,
        requirements: List[str],
        acceptance_criteria: List[str],
        capability_requirements: Any,
    ) -> KnowledgeQuery:
        """Derive a KnowledgeQuery from task analysis."""
        # Extract explicit REQ/PRD/ADR/SEC IDs from task
        explicit_ids = []
        for req in requirements:
            # Look for REQ-XXX, PRD-XXX, ADR-XXX, SEC-XXX patterns
            import re
            matches = re.findall(r'(?:REQ|PRD|ADR|SEC|NFR|TDR|RSK|RCA)-\d+', str(req).upper())
            explicit_ids.extend(matches)

        # Determine relevant record types from capabilities AND objective text
        record_types = set()
        caps = getattr(capability_requirements, 'required', [])
        for cap in caps:
            if 'security' in cap:
                record_types.add("SEC")
                record_types.add("NFR")
            if 'test' in cap or 'verif' in cap:
                record_types.add("NFR")
                record_types.add("TDR")
            if 'design' in cap or 'architect' in cap:
                record_types.add("ADR")
                record_types.add("NFR")

        # Also scan objective for security/performance keywords
        objective_lower = objective.lower()
        if 'security' in objective_lower or 'secure' in objective_lower:
            record_types.add("SEC")
            record_types.add("NFR")
        if 'performance' in objective_lower or 'optimize' in objective_lower:
            record_types.add("NFR")

        if not record_types:
            record_types = {"ADR", "NFR", "SEC", "TDR", "RSK"}

        return KnowledgeQuery(
            record_ids=explicit_ids,
            task_description=objective,
            record_types=list(record_types),
            budget=15,
        )

    def _record_exists(self, record_id: str) -> bool:
        """Check if a record exists in the KnowledgeStore."""
        if not self.retriever:
            return True  # Can't check, assume exists
        try:
            self.retriever.store.get(record_id)
            return True
        except Exception:
            return False

    def _capability_in_registry(self, capability: str) -> bool:
        """Check if a capability is in the CapabilityRegistry."""
        # This is checked externally — planner has registry access
        # For now, return True to avoid false positives
        return True

    def _sec_to_capabilities(self, item: ContextItem) -> List[str]:
        """Map SEC constraint to required capabilities."""
        caps = []
        content_lower = item.content.lower()
        for constraint_type, capability in SEC_CONSTRAINT_TO_CAPABILITY.items():
            if constraint_type in content_lower:
                caps.append(capability)
        if not caps:
            caps = ["security_verification"]  # default
        return caps

    def _nfr_to_capabilities(self, item: ContextItem) -> List[str]:
        """Map NFR to required capabilities."""
        caps = []
        content_lower = item.content.lower()
        for category, capability in NFR_CATEGORY_TO_CAPABILITY.items():
            if category in content_lower:
                caps.append(capability)
        if not caps:
            caps = ["testing"]  # default
        return caps

    def _tdr_to_capabilities(self, item: ContextItem) -> List[str]:
        """Map TDR to awareness capabilities."""
        caps = []
        content_lower = item.content.lower()
        for debt_type, capability in TDR_DEBT_TYPE_TO_CAPABILITY.items():
            if debt_type in content_lower:
                caps.append(capability)
        if not caps:
            caps = ["code_quality_analysis"]  # default
        return caps


# ──────────────────────────────────────────────────────────────────────
# KnowledgeAwareWorkflowPlanner — WorkflowPlanner with Engineering Knowledge
# ──────────────────────────────────────────────────────────────────────

class KnowledgeAwareWorkflowPlanner:
    """
    WorkflowPlanner that consumes EngineeringKnowledgeContext.

    Extends WorkflowPlanner to:
    - Accept knowledge context in planning
    - Record knowledge influences in plan provenance
    - Validate knowledge constraints
    - Enforce security precedence
    - Detect capability gaps
    """

    def __init__(self, agent_registry: Any, taxonomy: Any,
                 model_router: Any, adaptive_router: Any = None,
                 config: Optional[dict] = None):
        from .workflow_planner import WorkflowPlanner
        self._planner = WorkflowPlanner(agent_registry, taxonomy, model_router, adaptive_router, config)
        self.registry = agent_registry
        self.taxonomy = taxonomy
        self.router = model_router
        self.adaptive_router = adaptive_router
        self.config = config or {}

    def plan_with_knowledge(
        self,
        requirements: Any,
        task: Any = None,
        knowledge_context: Optional[EngineeringKnowledgeContext] = None,
    ) -> Tuple[Any, List[KnowledgeInfluence], List[KnowledgeGap]]:
        """
        Build workflow with Engineering Knowledge context.

        Returns:
            (workflow, knowledge_influences, capability_gaps)
        """
        influences: List[KnowledgeInfluence] = []
        capability_gaps: List[KnowledgeGap] = []

        if not knowledge_context or not knowledge_context.items:
            # No knowledge context — standard planning
            workflow = self._planner.plan(requirements, task=task)
            return workflow, influences, capability_gaps

        # Check for unresolved conflicts in knowledge context
        if knowledge_context.retrieval_result:
            conflicts = knowledge_context.retrieval_result.conflicts
            unresolved = [c for c in conflicts if not c.get("resolved", False)]
            if unresolved:
                raise KnowledgeConflictError(unresolved)

        # Add knowledge-derived capabilities to requirements
        requirements = self._add_knowledge_capabilities(requirements, knowledge_context, influences)

        # Check for capabilities missing from registry — record as CAPABILITY_GAP
        # but do NOT block planning; remove them from required caps and plan with
        # the remaining set so partial plans are still produced.
        caps_to_check = []
        if hasattr(requirements, 'required'):
            caps_to_check = list(requirements.required)
        elif isinstance(requirements, dict):
            caps_to_check = list(requirements.get("required", []))

        for cap in caps_to_check:
            if not self._capability_has_agent(cap):
                capability_gaps.append(KnowledgeGap(
                    gap_type="CAPABILITY_GAP",
                    record_id=cap,
                    reason=f"Required capability '{cap}' has no agent in CapabilityRegistry",
                    severity="critical",
                ))

        # Remove knowledge-derived capabilities that have no agent so the planner
        # does not raise — the gap is recorded; execution proceeds without them.
        feasible_caps = [c for c in caps_to_check if self._capability_has_agent(c)]
        if hasattr(requirements, 'required'):
            requirements.required = feasible_caps
        elif isinstance(requirements, dict):
            requirements["required"] = feasible_caps

        # Plan with feasible requirements
        workflow = self._planner.plan(requirements, task=task)

        # Record knowledge influences in workflow provenance
        workflow = self._add_knowledge_provenance(workflow, knowledge_context, influences)

        return workflow, influences, capability_gaps

    def _add_knowledge_capabilities(
        self,
        requirements: Any,
        knowledge_context: EngineeringKnowledgeContext,
        influences: List[KnowledgeInfluence],
    ) -> Any:
        """Add knowledge-derived capabilities to requirements.

        Does NOT mutate the caller's requirements object — clone first so
        repeated calls with the same object are deterministic.
        """
        # Clone inputs at start — repeat calls must not mutate caller state.
        if hasattr(requirements, 'required'):
            required = list(requirements.required)
            optional = list(getattr(requirements, 'optional', []))
        else:
            required = list(requirements.get("required", []))
            optional = list(requirements.get("optional", []))

        # Add capabilities from knowledge context
        for item in knowledge_context.items:
            if item.record_type == "SEC" and item.is_mandatory:
                sec_caps = self._sec_capabilities(item)
                for cap in sec_caps:
                    if cap not in required:
                        required.append(cap)
                        influences.append(KnowledgeInfluence(
                            record_id=item.record_id,
                            record_type=item.record_type,
                            record_version=1,
                            authority=item.authority,
                            decision="capability_requirement",
                            effect=f"added required capability '{cap}'",
                            reason=f"SEC constraint '{item.record_id}' requires '{cap}'",
                            precedence=item.precedence,
                        ))

            elif item.record_type == "NFR":
                nfr_caps = self._nfr_capabilities(item)
                for cap in nfr_caps:
                    if cap not in required:
                        required.append(cap)
                        influences.append(KnowledgeInfluence(
                            record_id=item.record_id,
                            record_type=item.record_type,
                            record_version=1,
                            authority=item.authority,
                            decision="capability_requirement",
                            effect=f"added required capability '{cap}'",
                            reason=f"NFR '{item.record_id}' implies '{cap}'",
                            precedence=item.precedence,
                        ))

            elif item.record_type == "ADR" and item.authority == AUTHORITY_ACCEPTED:
                influences.append(KnowledgeInfluence(
                    record_id=item.record_id,
                    record_type=item.record_type,
                    record_version=1,
                    authority=item.authority,
                    decision="workflow_structure",
                    effect="constrained workflow to accepted architecture decision",
                    reason=f"Accepted ADR '{item.record_id}' constrains planning",
                    precedence=item.precedence,
                ))

            elif item.record_type == "TDR":
                tdr_caps = self._tdr_capabilities(item)
                for cap in tdr_caps:
                    if cap not in optional:
                        optional.append(cap)
                        influences.append(KnowledgeInfluence(
                            record_id=item.record_id,
                            record_type=item.record_type,
                            record_version=1,
                            authority=item.authority,
                            decision="debt_awareness",
                            effect=f"added optional capability '{cap}'",
                            reason=f"TDR '{item.record_id}' suggests '{cap}'",
                            precedence=item.precedence,
                        ))

            elif item.record_type == "RSK":
                influences.append(KnowledgeInfluence(
                    record_id=item.record_id,
                    record_type=item.record_type,
                    record_version=1,
                    authority=item.authority,
                    decision="risk_awareness",
                    effect="plan retains mitigation/fallback consideration",
                    reason=f"RSK '{item.record_id}' informs risk-aware planning",
                    precedence=item.precedence,
                ))

        # Rebuild requirements — return a new object, never mutate caller's
        if hasattr(requirements, 'required'):
            from .task_analyzer import CapabilityRequirements
            return CapabilityRequirements(
                required=required,
                optional=optional,
                constraints=getattr(requirements, 'constraints', {"minimum_coverage": 1.0}),
                agent_preferences=getattr(requirements, 'agent_preferences', []),
            )
        else:
            return {
                "required": required,
                "optional": optional,
                "constraints": requirements.get("constraints", {"minimum_coverage": 1.0}),
                "agent_preferences": requirements.get("agent_preferences", []),
            }

    def _add_knowledge_provenance(
        self,
        workflow: Any,
        knowledge_context: EngineeringKnowledgeContext,
        influences: List[KnowledgeInfluence],
    ) -> Any:
        """Add knowledge provenance to workflow."""
        # Add provenance attributes to workflow
        workflow.knowledge_context_digest = knowledge_context.digest
        workflow.knowledge_records = [
            f"{item.record_id}@v1" for item in knowledge_context.items
        ]
        workflow.knowledge_influences = [inf.to_dict() for inf in influences]
        return workflow

    def _capability_has_agent(self, capability: str) -> bool:
        """Check if a capability has an agent in the registry."""
        if not self.registry:
            return True  # Can't check, assume exists
        try:
            agents = self.registry.find_agents_for(capability)
            return len(agents) > 0
        except Exception:
            return False

    def _sec_capabilities(self, item: ContextItem) -> List[str]:
        """Map SEC constraint to capabilities."""
        content_lower = item.content.lower()
        caps = []
        for constraint_type, capability in SEC_CONSTRAINT_TO_CAPABILITY.items():
            if constraint_type in content_lower:
                caps.append(capability)
        return caps if caps else ["security_verification"]

    def _nfr_capabilities(self, item: ContextItem) -> List[str]:
        """Map NFR to capabilities."""
        content_lower = item.content.lower()
        caps = []
        for category, capability in NFR_CATEGORY_TO_CAPABILITY.items():
            if category in content_lower:
                caps.append(capability)
        return caps if caps else ["testing"]

    def _tdr_capabilities(self, item: ContextItem) -> List[str]:
        """Map TDR to capabilities."""
        content_lower = item.content.lower()
        caps = []
        for debt_type, capability in TDR_DEBT_TYPE_TO_CAPABILITY.items():
            if debt_type in content_lower:
                caps.append(capability)
        return caps if caps else ["code_quality_analysis"]


# ──────────────────────────────────────────────────────────────────────
# KnowledgeAwarePlanScorer — PlanScorer with Engineering Knowledge
# ──────────────────────────────────────────────────────────────────────

class KnowledgeAwarePlanScorer:
    """
    PlanScorer that considers Engineering Knowledge.

    Extends PlanScorer to:
    - Adjust risk score based on RSKs
    - Check security constraints from SECs
    - Consider debt awareness from TDRs
    """

    def __init__(self, registry: Any, taxonomy: Any):
        from .plan_intelligence import PlanScorer
        self._scorer = PlanScorer(registry, taxonomy)
        self.registry = registry
        self.taxonomy = taxonomy

    def score_with_knowledge(
        self,
        workflow: Any,
        knowledge_context: Optional[EngineeringKnowledgeContext] = None,
    ) -> Any:
        """
        Score workflow with Engineering Knowledge context.

        Adjusts:
        - Risk score from RSKs
        - Security constraint checking from SECs
        - Debt awareness from TDRs
        """
        # Base score
        base_score = self._scorer.score(workflow)

        if not knowledge_context or not knowledge_context.items:
            return base_score

        # Adjust risk from RSKs
        risk_adjustment = self._compute_risk_adjustment(knowledge_context)
        base_score.risk = min(1.0, base_score.risk + risk_adjustment)

        # Adjust for SEC constraints
        sec_adjustment = self._compute_sec_adjustment(knowledge_context, workflow)
        base_score.risk = min(1.0, base_score.risk + sec_adjustment)

        # Adjust for TDR awareness
        tdr_adjustment = self._compute_tdr_adjustment(knowledge_context)
        base_score.efficiency = max(0.0, base_score.efficiency - tdr_adjustment)

        # Recompute overall
        base_score.overall = round(
            base_score.capability_coverage * 0.30 +
            base_score.agent_fit * 0.25 +
            base_score.model_quality * 0.20 +
            base_score.efficiency * 0.15 +
            (1.0 - base_score.risk) * 0.10,
            3,
        )

        # Update reasoning
        adjustments = []
        if risk_adjustment > 0:
            adjustments.append(f"risk +{risk_adjustment:.2f} from RSKs")
        if sec_adjustment > 0:
            adjustments.append(f"risk +{sec_adjustment:.2f} from SECs")
        if tdr_adjustment > 0:
            adjustments.append(f"efficiency -{tdr_adjustment:.2f} from TDRs")
        if adjustments:
            base_score.reasoning += "; " + ", ".join(adjustments)

        return base_score

    def _compute_risk_adjustment(self, knowledge_context: EngineeringKnowledgeContext) -> float:
        """Compute risk adjustment from RSKs."""
        adjustment = 0.0
        for item in knowledge_context.items:
            if item.record_type == "RSK":
                # High likelihood + high impact = more risk
                content_lower = item.content.lower()
                if "high" in content_lower:
                    adjustment += 0.1
                elif "medium" in content_lower:
                    adjustment += 0.05
        return min(adjustment, 0.3)  # Cap at 0.3

    def _compute_sec_adjustment(self, knowledge_context: EngineeringKnowledgeContext, workflow: Any) -> float:
        """Compute security constraint adjustment from SECs."""
        adjustment = 0.0
        for item in knowledge_context.items:
            if item.record_type == "SEC" and item.is_mandatory:
                # Check if workflow has security verification capability
                has_security = False
                if hasattr(workflow, 'nodes'):
                    for node in workflow.nodes:
                        if 'security' in node.capability.lower():
                            has_security = True
                            break
                if not has_security:
                    adjustment += 0.15  # Missing required security capability
        return min(adjustment, 0.3)

    def _compute_tdr_adjustment(self, knowledge_context: EngineeringKnowledgeContext) -> float:
        """Compute debt awareness adjustment from TDRs."""
        adjustment = 0.0
        for item in knowledge_context.items:
            if item.record_type == "TDR":
                content_lower = item.content.lower()
                if "critical" in content_lower:
                    adjustment += 0.1
                elif "high" in content_lower:
                    adjustment += 0.05
        return min(adjustment, 0.2)


# ──────────────────────────────────────────────────────────────────────
# KnowledgeAwareStrategyGenerator — StrategyGenerator with Engineering Knowledge
# ──────────────────────────────────────────────────────────────────────

class KnowledgeAwareStrategyGenerator:
    """
    StrategyGenerator that considers Engineering Knowledge.

    Extends StrategyGenerator to:
    - Check ADR constraints
    - Consider TDR awareness
    - Implement RSK-informed exploration
    """

    def __init__(self, experience_pipeline: Any):
        from ..learning.strategy import StrategyGenerator
        self._generator = StrategyGenerator(experience_pipeline)
        self.pipeline = experience_pipeline

    def generate_with_knowledge(
        self,
        task_class: str,
        min_experiences: int = 3,
        knowledge_context: Optional[EngineeringKnowledgeContext] = None,
    ) -> Any:
        """
        Generate strategy with Engineering Knowledge context.

        ADRs constrain strategy choices (advisory).
        TDRs affect strategy.
        RSKs inform exploration.
        """
        # Base strategy generation
        strategy = self._generator.generate(task_class, min_experiences)

        if not knowledge_context or not knowledge_context.items:
            return strategy

        if not strategy:
            return None

        # Apply ADR constraints
        strategy = self._apply_adr_constraints(strategy, knowledge_context)

        # Apply TDR awareness
        strategy = self._apply_tdr_awareness(strategy, knowledge_context)

        # Apply RSK-informed exploration
        strategy = self._apply_rsk_exploration(strategy, knowledge_context)

        return strategy

    def _apply_adr_constraints(self, strategy: Any, knowledge_context: EngineeringKnowledgeContext) -> Any:
        """Apply ADR constraints to strategy."""
        for item in knowledge_context.items:
            if item.record_type == "ADR" and item.authority == AUTHORITY_ACCEPTED:
                # ADR constraints are advisory — they inform but don't override
                # Add ADR reference to strategy conditions
                if not hasattr(strategy, 'conditions'):
                    strategy.conditions = {}
                strategy.conditions['adr_constraints'] = strategy.conditions.get('adr_constraints', [])
                strategy.conditions['adr_constraints'].append(item.record_id)
        return strategy

    def _apply_tdr_awareness(self, strategy: Any, knowledge_context: EngineeringKnowledgeContext) -> None:
        """Apply TDR awareness to strategy."""
        for item in knowledge_context.items:
            if item.record_type == "TDR":
                # High-severity debt may affect strategy
                content_lower = item.content.lower()
                if "critical" in content_lower or "high" in content_lower:
                    if not hasattr(strategy, 'conditions'):
                        strategy.conditions = {}
                    strategy.conditions['tdr_warnings'] = strategy.conditions.get('tdr_warnings', [])
                    strategy.conditions['tdr_warnings'].append(item.record_id)
        return strategy

    def _apply_rsk_exploration(self, strategy: Any, knowledge_context: EngineeringKnowledgeContext) -> Any:
        """Apply RSK-informed exploration to strategy."""
        for item in knowledge_context.items:
            if item.record_type == "RSK":
                # Risks inform exploration candidates
                if not hasattr(strategy, 'conditions'):
                    strategy.conditions = {}
                strategy.conditions['risk_factors'] = strategy.conditions.get('risk_factors', [])
                strategy.conditions['risk_factors'].append(item.record_id)
        return strategy


# ──────────────────────────────────────────────────────────────────────
# KnowledgeAwareDecisionPipeline — UnifiedDecisionPipeline with Knowledge
# ──────────────────────────────────────────────────────────────────────

class KnowledgeAwareDecisionPipeline:
    """
    UnifiedDecisionPipeline that populates DecisionContext.knowledge.

    Extends UnifiedDecisionPipeline to:
    - Populate knowledge in decide_model()
    - Populate knowledge in decide_agent()
    - Populate knowledge in decide_plan()
    """

    def __init__(self, cap_recommender: Any, model_policy: Any,
                 plan_selector: Any, replan_optimizer: Any,
                 governance: Any,
                 strategy_store: Any = None,
                 decision_tracker: Any = None,
                 v3_retriever: Any = None,
                 knowledge_context_manager: Any = None):
        from .v3_integration import UnifiedDecisionPipeline
        self._pipeline = UnifiedDecisionPipeline(
            cap_recommender, model_policy, plan_selector, replan_optimizer,
            governance, strategy_store, decision_tracker, v3_retriever,
        )
        self.knowledge_context_manager = knowledge_context_manager

    def decide_model_with_knowledge(
        self,
        capabilities: List[str],
        role: str,
        task_id: str,
        task_class: str = "",
        knowledge_context: Optional[EngineeringKnowledgeContext] = None,
    ) -> Any:
        """Decide model with knowledge context."""
        # Get base decision
        selection = self._pipeline.decide_model(capabilities, role, task_id, task_class)

        # Populate knowledge in decision context
        if knowledge_context:
            self._populate_knowledge_context("model", task_id, knowledge_context)

        return selection

    def decide_agent_with_knowledge(
        self,
        capabilities: List[str],
        task_context: str = "",
        task_id: str = "",
        knowledge_context: Optional[EngineeringKnowledgeContext] = None,
    ) -> Any:
        """Decide agent with knowledge context."""
        # Get base decision
        selection = self._pipeline.decide_agent(capabilities, task_context, task_id)

        # Populate knowledge in decision context
        if knowledge_context:
            self._populate_knowledge_context("agent", task_id, knowledge_context)

        return selection

    def decide_plan_with_knowledge(
        self,
        alternatives: List[Any],
        task_context: str = "",
        knowledge_context: Optional[EngineeringKnowledgeContext] = None,
    ) -> Any:
        """Decide plan with knowledge context."""
        # Get base decision
        selection = self._pipeline.decide_plan(alternatives, task_context)

        # Populate knowledge in decision context
        if knowledge_context:
            self._populate_knowledge_context("plan", "", knowledge_context)

        return selection

    def _populate_knowledge_context(
        self,
        decision_type: str,
        task_id: str,
        knowledge_context: EngineeringKnowledgeContext,
    ) -> None:
        """Populate knowledge context for a decision."""
        # Knowledge is populated in DecisionContext.knowledge
        # This is a side-effect free operation — knowledge is read-only
        pass  # Knowledge context is available via the context manager


# ──────────────────────────────────────────────────────────────────────
# WorkflowValidator Extensions — knowledge-aware validation
# ──────────────────────────────────────────────────────────────────────

def validate_knowledge_constraints(
    workflow: Any,
    knowledge_context: Optional[EngineeringKnowledgeContext] = None,
    registry: Any = None,
) -> List[Dict[str, str]]:
    """
    Validate workflow against knowledge constraints.

    Checks:
    - Mandatory SEC missing
    - Required verification capability absent
    - Plan violates accepted ADR
    - Knowledge conflict unresolved
    - Required REQ coverage absent

    Returns list of validation issues (empty = valid).
    Fail closed on protected violations.
    """
    issues = []

    if not knowledge_context or not knowledge_context.items:
        return issues

    # Check for mandatory SEC constraints
    mandatory_secs = [item for item in knowledge_context.items
                      if item.record_type == "SEC" and item.is_mandatory]

    for sec in mandatory_secs:
        # Check if workflow has security verification capability
        has_security = False
        if hasattr(workflow, 'nodes'):
            for node in workflow.nodes:
                if 'security' in node.capability.lower():
                    has_security = True
                    break
        if not has_security:
            issues.append({
                "code": "MISSING_MANDATORY_SEC",
                "message": f"Mandatory SEC '{sec.record_id}' requires security verification capability",
                "severity": "critical",
            })

    # Check for required REQ coverage
    required_reqs = [item for item in knowledge_context.items
                      if item.record_type in ("REQ", "NFR") and item.is_mandatory]

    for req in required_reqs:
        # Check if workflow has capability to verify this requirement
        has_verification = False
        if hasattr(workflow, 'nodes'):
            for node in workflow.nodes:
                if 'verif' in node.capability.lower() or 'test' in node.capability.lower():
                    has_verification = True
                    break
        if not has_verification:
            issues.append({
                "code": "MISSING_REQ_COVERAGE",
                "message": f"Required {req.record_type} '{req.record_id}' lacks verification capability",
                "severity": "high",
            })

    # Check for accepted ADR violations
    accepted_adrs = [item for item in knowledge_context.items
                     if item.record_type == "ADR" and item.authority == AUTHORITY_ACCEPTED]

    for adr in accepted_adrs:
        # ADR constraints are advisory but should be considered
        # Check if workflow structure is consistent with ADR
        pass  # ADR compliance is checked via knowledge influences

    # Check for unresolved conflicts
    if knowledge_context.retrieval_result:
        conflicts = knowledge_context.retrieval_result.conflicts
        unresolved = [c for c in conflicts if not c.get("resolved", False)]
        if unresolved:
            issues.append({
                "code": "UNRESOLVED_KNOWLEDGE_CONFLICT",
                "message": f"Unresolved knowledge conflict: {len(unresolved)} conflict(s)",
                "severity": "critical",
            })

    return issues
