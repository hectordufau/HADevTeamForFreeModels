# harness/knowledge/traceability.py — V3.3 Phase 10: Traceability
"""
Cross-domain traceability: derived projection from canonical domain objects.

Phase 10 Implementation — TASK-064 (Evidence Traceability Chain)

Architecture:
- Traceability is a derived, read-only projection from canonical domains
- Canonical domains: Engineering Knowledge, Task Contract, Plan, Execution,
  Verification, Evidence, Review
- Traceability does NOT merge storage models
- Deterministic: same canonical state → same logical traceability
- No dependency on Python hash(), set order, filesystem order, or RNG
- Rebuildable: traceability can be rebuilt from canonical state at any time

Cross-domain chain:
  PRD → REQ → NFR → ADR → TASK → PLAN → EXECUTION → VERIFICATION → EVIDENCE → REVIEW
  with TDR/RSK/SEC/RCA as side constraints

Three distinct concepts:
  RETRIEVED → INFLUENCED → COMPLIED/VIOLATED/UNVERIFIED
  Never infer one from another.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


class TraceabilityError(Exception):
    """Raised on traceability errors."""


class BrokenTraceError(TraceabilityError):
    """Raised when a trace references a missing target."""

    def __init__(self, source_id: str, target_id: str, relation: str):
        self.source_id = source_id
        self.target_id = target_id
        self.relation = relation
        super().__init__(
            f"BROKEN_TRACE: {source_id} --[{relation}]--> {target_id} "
            f"(target does not exist)"
        )


# ──────────────────────────────────────────────────────────────────────
# TraceabilityEdge — a single cross-domain trace link
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TraceabilityEdge:
    """
    A single traceability link between two domain objects.

    Uses stable identity: record_id + version where applicable.
    No object pointers, no Python hash(), no temporary indices.
    """
    source_domain: str  # "engineering", "task", "planning", "execution", "verification", "evidence", "review"
    source_id: str
    source_version: Optional[int] = None  # version-aware where applicable
    target_domain: str = ""
    target_id: str = ""
    target_version: Optional[int] = None
    relation: str = ""  # "REQUIRES", "INFLUENCED", "EXECUTED", "VERIFIED", "EVIDENCED", "REVIEWED"
    status: str = ""  # "SATISFIED", "VIOLATED", "UNVERIFIED", "NOT_APPLICABLE"
    evidence_refs: List[str] = field(default_factory=list)  # evidence IDs, not content
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": {
                "domain": self.source_domain,
                "id": self.source_id,
                "version": self.source_version,
            },
            "target": {
                "domain": self.target_domain,
                "id": self.target_id,
                "version": self.target_version,
            },
            "relation": self.relation,
            "status": self.status,
            "evidence_refs": list(self.evidence_refs),
            "provenance": dict(self.provenance),
        }

    def identity_key(self) -> str:
        """Deterministic identity key for this edge."""
        return (
            f"{self.source_domain}:{self.source_id}"
            f"{'@v' + str(self.source_version) if self.source_version else ''}"
            f"--[{self.relation}]-->"
            f"{self.target_domain}:{self.target_id}"
            f"{'@v' + str(self.target_version) if self.target_version else ''}"
        )


# ──────────────────────────────────────────────────────────────────────
# TraceabilityChain — full traceability chain for a task/decision
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TraceabilityChain:
    """
    Full traceability chain from requirements to review.

    Deterministic: same canonical state → same chain → same digest.
    """
    task_id: str
    edges: List[TraceabilityEdge] = field(default_factory=list)
    digest: str = ""
    completeness: str = ""  # "COMPLETE", "PARTIAL", "BROKEN", "UNVERIFIED"
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "edges": [e.to_dict() for e in self.edges],
            "digest": self.digest,
            "completeness": self.completeness,
            "metrics": dict(self.metrics),
        }


# ──────────────────────────────────────────────────────────────────────
# TraceabilityMetrics — deterministic coverage metrics
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TraceabilityMetrics:
    """
    Deterministic traceability coverage metrics.

    All metrics are computed from canonical state, not from plan claims.
    """
    # Requirements
    applicable_requirements: int = 0
    requirements_planned: int = 0
    requirements_verified: int = 0
    requirements_evidenced: int = 0
    requirements_satisfied: int = 0
    requirements_violated: int = 0
    requirements_unverified: int = 0

    # Security (release-critical)
    applicable_sec: int = 0
    sec_reviewed: int = 0
    sec_evidenced: int = 0
    sec_violated: int = 0
    sec_unverified: int = 0

    # Knowledge influences
    material_influences: int = 0
    influences_with_execution_trace: int = 0
    influences_with_verification_trace: int = 0
    influences_with_evidence_trace: int = 0
    influences_with_review_outcome: int = 0

    # Trace health
    orphan_count: int = 0
    broken_trace_count: int = 0

    def to_dict(self) -> dict:
        return {
            "applicable_requirements": self.applicable_requirements,
            "requirements_planned": self.requirements_planned,
            "requirements_verified": self.requirements_verified,
            "requirements_evidenced": self.requirements_evidenced,
            "requirements_satisfied": self.requirements_satisfied,
            "requirements_violated": self.requirements_violated,
            "requirements_unverified": self.requirements_unverified,
            "applicable_sec": self.applicable_sec,
            "sec_reviewed": self.sec_reviewed,
            "sec_evidenced": self.sec_evidenced,
            "sec_violated": self.sec_violated,
            "sec_unverified": self.sec_unverified,
            "material_influences": self.material_influences,
            "influences_with_execution_trace": self.influences_with_execution_trace,
            "influences_with_verification_trace": self.influences_with_verification_trace,
            "influences_with_evidence_trace": self.influences_with_evidence_trace,
            "influences_with_review_outcome": self.influences_with_review_outcome,
            "orphan_count": self.orphan_count,
            "broken_trace_count": self.broken_trace_count,
        }


# ──────────────────────────────────────────────────────────────────────
# TraceabilityService — builds and rebuilds traceability from canonical state
# ──────────────────────────────────────────────────────────────────────

class TraceabilityService:
    """
    Builds deterministic traceability from canonical domain objects.

    Traceability is a derived projection — it does NOT mutate canonical state.
    Rebuild is deterministic: same canonical state → same logical traceability.
    """

    def __init__(self, knowledge_store: Any = None, evidence_store: Any = None):
        self.knowledge_store = knowledge_store
        self.evidence_store = evidence_store

    def build_chain(
        self,
        task_id: str,
        plan: Any = None,
        execution: Any = None,
        verification: Any = None,
        knowledge_context: Any = None,
        review: Any = None,
    ) -> TraceabilityChain:
        """
        Build full traceability chain for a task.

        Args:
            task_id: Stable task identifier
            plan: Plan object with provenance (optional)
            execution: Execution result (optional)
            verification: Verification result (optional)
            knowledge_context: EngineeringKnowledgeContext (optional)
            review: Review result (optional)

        Returns:
            TraceabilityChain with edges, digest, completeness, metrics
        """
        edges: List[TraceabilityEdge] = []

        # 0. Task → Plan edge (always present)
        if plan:
            plan_id = getattr(plan, 'plan_id', 'unknown')
            edges.append(TraceabilityEdge(
                source_domain="task",
                source_id=task_id,
                target_domain="planning",
                target_id=plan_id,
                relation="PLANNED",
                status="UNVERIFIED",
            ))

        # 0b. Task → Execution edge (always present)
        if execution:
            exec_id = getattr(execution, 'execution_id', 'unknown')
            edges.append(TraceabilityEdge(
                source_domain="task",
                source_id=task_id,
                target_domain="execution",
                target_id=exec_id,
                relation="EXECUTED",
                status="UNVERIFIED",
            ))

        # 1. Knowledge → Plan edges (INFLUENCED)
        if knowledge_context and hasattr(knowledge_context, 'items'):
            for item in knowledge_context.items:
                edge = TraceabilityEdge(
                    source_domain="engineering",
                    source_id=item.record_id,
                    source_version=getattr(item, 'version', None),
                    target_domain="planning",
                    target_id=getattr(plan, 'plan_id', 'unknown') if plan else 'unknown',
                    relation="INFLUENCED",
                    status="UNVERIFIED",  # will be updated by reviewer
                    provenance={
                        "authority": item.authority,
                        "record_type": item.record_type,
                    },
                )
                edges.append(edge)

        # 2. Plan → Execution edges (EXECUTED)
        if plan and execution:
            plan_id = getattr(plan, 'plan_id', 'unknown')
            exec_id = getattr(execution, 'execution_id', 'unknown')
            plan_digest = getattr(plan, 'plan_digest', '')

            edge = TraceabilityEdge(
                source_domain="planning",
                source_id=plan_id,
                target_domain="execution",
                target_id=exec_id,
                relation="EXECUTED",
                status="UNVERIFIED",
                provenance={"plan_digest": plan_digest},
            )
            edges.append(edge)

        # 3. Execution → Verification edges (VERIFIED)
        if execution and verification:
            exec_id = getattr(execution, 'execution_id', 'unknown')
            ver_id = getattr(verification, 'verification_id', 'unknown')
            ver_status = getattr(verification, 'status', 'unknown')

            edge = TraceabilityEdge(
                source_domain="execution",
                source_id=exec_id,
                target_domain="verification",
                target_id=ver_id,
                relation="VERIFIED",
                status=ver_status,
            )
            edges.append(edge)

        # 4. Verification → Evidence edges (EVIDENCED)
        if verification and self.evidence_store:
            ver_id = getattr(verification, 'verification_id', 'unknown')
            evidence_ids = self._get_evidence_ids_for_verification(ver_id)
            for ev_id in evidence_ids:
                edge = TraceabilityEdge(
                    source_domain="verification",
                    source_id=ver_id,
                    target_domain="evidence",
                    target_id=ev_id,
                    relation="EVIDENCED",
                    status="UNVERIFIED",
                )
                edges.append(edge)

        # 5. Review edges (REVIEWED)
        if review:
            review_id = getattr(review, 'review_id', 'unknown')
            review_status = getattr(review, 'status', 'unknown')
            edge = TraceabilityEdge(
                source_domain="review",
                source_id=review_id,
                target_domain="task",
                target_id=task_id,
                relation="REVIEWED",
                status=review_status,
            )
            edges.append(edge)

        # Deterministic ordering
        edges.sort(key=lambda e: e.identity_key())

        # Compute completeness
        completeness = self._compute_completeness(edges)

        # Compute metrics
        metrics = self._compute_metrics(edges, knowledge_context)

        # Compute digest
        digest = self._compute_digest(edges)

        return TraceabilityChain(
            task_id=task_id,
            edges=edges,
            digest=digest,
            completeness=completeness,
            metrics=metrics.to_dict(),
        )

    def rebuild_chain(self, task_id: str, **canonical_refs) -> TraceabilityChain:
        """
        Rebuild traceability from canonical state.

        Same canonical state → same logical traceability, independent of
        process restart/order.
        """
        return self.build_chain(task_id, **canonical_refs)

    def detect_orphans(self, edges: List[TraceabilityEdge]) -> List[Dict[str, str]]:
        """
        Detect orphan traces: edges referencing missing targets.

        Returns list of orphan descriptions. Does NOT create placeholders.
        """
        orphans = []
        for edge in edges:
            # Check if target exists in canonical state
            if edge.target_domain == "engineering" and self.knowledge_store:
                try:
                    self.knowledge_store.get(edge.target_id)
                except Exception:
                    orphans.append({
                        "type": "ORPHAN",
                        "source": edge.identity_key(),
                        "missing_target": f"{edge.target_domain}:{edge.target_id}",
                    })
            # Any edge with "unknown" target is also an orphan
            if edge.target_id == "unknown":
                orphans.append({
                    "type": "ORPHAN",
                    "source": edge.identity_key(),
                    "missing_target": f"{edge.target_domain}:unknown",
                })
        return orphans

    def detect_broken_traces(self, edges: List[TraceabilityEdge]) -> List[Dict[str, str]]:
        """
        Detect broken traces: references to missing internal objects.

        Returns list of broken trace descriptions. Does NOT create placeholders.
        """
        broken = []
        for edge in edges:
            # Any edge with "unknown" target is broken
            if edge.target_id == "unknown":
                broken.append({
                    "type": "BROKEN_TRACE",
                    "source": edge.identity_key(),
                    "reason": f"Target ID unknown in domain '{edge.target_domain}' — cannot trace",
                })
        return broken

    def _get_evidence_ids_for_verification(self, verification_id: str) -> List[str]:
        """Get evidence IDs bound to a verification."""
        if not self.evidence_store:
            return []
        try:
            # Query evidence store for this verification
            from ..evidence.unified import EvidenceQuery
            query = EvidenceQuery(task_id=verification_id)
            packages = self.evidence_store.query(query)
            return [pkg.content_hash for pkg in packages]
        except Exception:
            return []

    def _compute_completeness(self, edges: List[TraceabilityEdge]) -> str:
        """
        Compute trace completeness.

        COMPLETE: all applicable requirements have plan + verification + evidence
        PARTIAL: some requirements traced but not all
        BROKEN: broken traces detected
        UNVERIFIED: traces exist but verification status unknown

        Note: COMPLETE does NOT mean PASS. A complete trace can still be FAIL.
        """
        if not edges:
            return "UNVERIFIED"

        # Check for broken traces
        broken = self.detect_broken_traces(edges)
        if broken:
            return "BROKEN"

        # Check for verification
        has_verification = any(
            e.relation == "VERIFIED" for e in edges
        )
        has_evidence = any(
            e.relation == "EVIDENCED" for e in edges
        )

        if not has_verification:
            return "UNVERIFIED"

        if not has_evidence:
            return "PARTIAL"

        # Check if all edges have non-UNVERIFIED status
        all_verified = all(
            e.status not in ("UNVERIFIED", "") for e in edges
        )

        if all_verified:
            return "COMPLETE"

        return "PARTIAL"

    def _compute_metrics(
        self,
        edges: List[TraceabilityEdge],
        knowledge_context: Any = None,
    ) -> TraceabilityMetrics:
        """Compute deterministic traceability metrics."""
        metrics = TraceabilityMetrics()

        # Count SEC edges
        sec_edges = [e for e in edges if e.source_domain == "engineering"
                     and e.provenance.get("record_type") == "SEC"]
        metrics.applicable_sec = len(sec_edges)
        metrics.sec_reviewed = sum(1 for e in sec_edges if e.status != "UNVERIFIED")
        metrics.sec_evidenced = sum(1 for e in sec_edges if e.evidence_refs)
        metrics.sec_violated = sum(1 for e in sec_edges if e.status == "VIOLATED")
        metrics.sec_unverified = sum(1 for e in sec_edges if e.status == "UNVERIFIED")

        # Count requirement edges
        req_edges = [e for e in edges if e.provenance.get("record_type") in ("REQ", "NFR")]
        metrics.applicable_requirements = len(req_edges)
        metrics.requirements_planned = sum(1 for e in req_edges if e.relation == "INFLUENCED")
        metrics.requirements_verified = sum(1 for e in req_edges if e.relation == "VERIFIED")
        metrics.requirements_evidenced = sum(1 for e in req_edges if e.evidence_refs)
        metrics.requirements_satisfied = sum(1 for e in req_edges if e.status == "SATISFIED")
        metrics.requirements_violated = sum(1 for e in req_edges if e.status == "VIOLATED")
        metrics.requirements_unverified = sum(1 for e in req_edges if e.status == "UNVERIFIED")

        # Count knowledge influences
        influence_edges = [e for e in edges if e.relation == "INFLUENCED"]
        metrics.material_influences = len(influence_edges)
        metrics.influences_with_execution_trace = sum(
            1 for e in influence_edges
            if any(ex.relation == "EXECUTED" for ex in edges)
        )
        metrics.influences_with_verification_trace = sum(
            1 for e in influence_edges
            if any(ex.relation == "VERIFIED" for ex in edges)
        )
        metrics.influences_with_evidence_trace = sum(
            1 for e in influence_edges if e.evidence_refs
        )
        metrics.influences_with_review_outcome = sum(
            1 for e in influence_edges
            if any(ex.relation == "REVIEWED" for ex in edges)
        )

        # Orphans and broken traces
        metrics.orphan_count = len(self.detect_orphans(edges))
        metrics.broken_trace_count = len(self.detect_broken_traces(edges))

        return metrics

    def _compute_digest(self, edges: List[TraceabilityEdge]) -> str:
        """
        Compute deterministic traceability digest.

        Sorted canonical trace edges + source versions + target IDs +
        relation/status + evidence bindings → SHA-256.

        No dependency on Python hash(), set order, filesystem order, or RNG.
        """
        # Sort edges deterministically
        sorted_edges = sorted(edges, key=lambda e: e.identity_key())

        # Build canonical representation
        edge_dicts = []
        for edge in sorted_edges:
            edge_dicts.append({
                "source": {
                    "domain": edge.source_domain,
                    "id": edge.source_id,
                    "version": edge.source_version,
                },
                "target": {
                    "domain": edge.target_domain,
                    "id": edge.target_id,
                    "version": edge.target_version,
                },
                "relation": edge.relation,
                "status": edge.status,
                "evidence_refs": sorted(edge.evidence_refs),
            })

        digest_input = json.dumps(
            {"edges": edge_dicts},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
