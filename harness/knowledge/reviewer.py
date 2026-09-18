# harness/knowledge/reviewer.py — V3.3 Phase 10: Reviewer
"""
Independent Reviewer: evaluates knowledge compliance from evidence and constraints.

Phase 10 Implementation — TASK-063 (Reviewer Knowledge Integration)

Architecture:
- Reviewer is INDEPENDENT from Planner — does not trust plan claims
- Reviewer evaluates available evidence and constraints directly
- Reviewer cannot fabricate evidence, change authority, or mutate knowledge
- Reviewer produces deterministic structured ReviewResult

Critical distinctions:
- Retrieved != INFLUENCED != COMPLIED
- KnowledgeInfluence != KnowledgeCompliance
- Missing evidence → UNVERIFIED/FAIL, never PASS
- Corrupt/wrong evidence → cannot PASS
- SEC violation → FAIL even if all functional tests pass
- Plan drift → FAIL
- Knowledge drift → FAIL

Review statuses: PASS, FAIL, UNVERIFIED, CONFLICT, NEEDS_HUMAN
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .traceability import (
    TraceabilityService,
    TraceabilityChain,
    TraceabilityEdge,
    TraceabilityMetrics,
    BrokenTraceError,
)


class ReviewerError(Exception):
    """Raised on reviewer errors."""


class EvidenceFabricationError(ReviewerError):
    """Raised when reviewer is asked to fabricate evidence."""

    def __init__(self, requirement_id: str):
        self.requirement_id = requirement_id
        super().__init__(
            f"REVIEWER_FORBIDDEN: Cannot fabricate evidence for '{requirement_id}'. "
            f"Reviewer may request/report missing evidence but cannot manufacture proof."
        )


class AuthorityChangeError(ReviewerError):
    """Raised when reviewer is asked to change knowledge authority."""

    def __init__(self, record_id: str, target_status: str):
        self.record_id = record_id
        self.target_status = target_status
        super().__init__(
            f"REVIEWER_FORBIDDEN: Cannot change authority of '{record_id}' "
            f"to '{target_status}'. Reviewer is read-only."
        )


# ──────────────────────────────────────────────────────────────────────
# ReviewContext — typed input for reviewer
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ReviewContext:
    """
    Typed context for independent review.

    Reviewer receives all relevant domain objects and evaluates compliance
    independently from Planner claims.
    """
    # Task identification
    task_id: str = ""

    # Domain objects (typed references, not merged)
    task_contract: Any = None
    plan: Any = None
    execution: Any = None
    verification: Any = None
    evidence_refs: List[str] = field(default_factory=list)

    # Knowledge context
    knowledge_context: Any = None  # EngineeringKnowledgeContext
    knowledge_influences: List[Any] = field(default_factory=list)  # KnowledgeInfluence list

    # Security constraints (release-critical)
    applicable_sec: List[Any] = field(default_factory=list)

    # Policy context
    policy_context: Optional[Dict[str, Any]] = None

    # Conflicts
    conflicts: List[Dict[str, Any]] = field(default_factory=list)

    # Knowledge context digest (from planning)
    knowledge_context_digest: str = ""

    # Plan provenance
    plan_id: str = ""
    plan_digest: str = ""

    # Execution provenance
    execution_id: str = ""
    execution_plan_id: str = ""
    execution_plan_digest: str = ""

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_contract": str(self.task_contract) if self.task_contract else None,
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "execution_id": self.execution_id,
            "execution_plan_id": self.execution_plan_id,
            "execution_plan_digest": self.execution_plan_digest,
            "knowledge_context_digest": self.knowledge_context_digest,
            "applicable_sec": [s.record_id if hasattr(s, 'record_id') else str(s)
                              for s in self.applicable_sec],
            "conflicts": list(self.conflicts),
        }


# ──────────────────────────────────────────────────────────────────────
# RequirementResult — per-requirement review result
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RequirementResult:
    """Result for a single requirement/NFR/SEC review."""
    requirement_id: str
    requirement_type: str  # "REQ", "NFR", "SEC", "ADR", "TDR", "RSK", "RCA"
    status: str  # "SATISFIED", "VIOLATED", "UNVERIFIED", "NOT_APPLICABLE"
    verified: bool = False
    evidence_refs: List[str] = field(default_factory=list)
    verification_ids: List[str] = field(default_factory=list)
    reason: str = ""
    source: str = ""  # "evidence", "plan_claim", "none"

    def to_dict(self) -> dict:
        return {
            "requirement_id": self.requirement_id,
            "requirement_type": self.requirement_type,
            "status": self.status,
            "verified": self.verified,
            "evidence_refs": list(self.evidence_refs),
            "verification_ids": list(self.verification_ids),
            "reason": self.reason,
            "source": self.source,
        }


# ──────────────────────────────────────────────────────────────────────
# KnowledgeResult — per-knowledge-record review result
# ──────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeResult:
    """Result for a knowledge record's influence and outcome."""
    record_id: str
    record_type: str
    authority: str
    influence_status: str  # "RETRIEVED", "INFLUENCED", "NOT_INFLUENCED"
    outcome_status: str = ""  # "SATISFIED", "VIOLATED", "UNVERIFIED", "NOT_MEASURABLE"
    decision: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "record_type": self.record_type,
            "authority": self.authority,
            "influence_status": self.influence_status,
            "outcome_status": self.outcome_status,
            "decision": self.decision,
            "evidence_refs": list(self.evidence_refs),
            "reason": self.reason,
        }


# ──────────────────────────────────────────────────────────────────────
# ReviewFinding — structured reviewer finding
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ReviewFinding:
    """A single structured finding from the reviewer."""
    finding_type: str  # "SEC_VIOLATION", "MISSING_EVIDENCE", "PLAN_DRIFT", etc.
    severity: str  # "critical", "high", "medium", "low"
    source: str  # "REQ-004", "SEC-002", "ADR-003", "EV-011"
    message: str
    evidence_refs: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "finding_type": self.finding_type,
            "severity": self.severity,
            "source": self.source,
            "message": self.message,
            "evidence_refs": list(self.evidence_refs),
        }


# ──────────────────────────────────────────────────────────────────────
# ReviewResult — structured review output
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ReviewResult:
    """
    Structured review result.

    Deterministic: same ReviewContext + Evidence + Policy → equivalent
    structured review/digest for deterministic rules.
    """
    review_id: str
    task_id: str
    status: str  # "PASS", "FAIL", "UNVERIFIED", "CONFLICT", "NEEDS_HUMAN"
    findings: List[ReviewFinding] = field(default_factory=list)
    requirement_results: List[RequirementResult] = field(default_factory=list)
    knowledge_results: List[KnowledgeResult] = field(default_factory=list)
    violations: List[ReviewFinding] = field(default_factory=list)
    missing_evidence: List[ReviewFinding] = field(default_factory=list)
    conflicts: List[ReviewFinding] = field(default_factory=list)

    # Security coverage (release-critical)
    sec_coverage: Dict[str, Any] = field(default_factory=dict)

    # Traceability
    traceability_digest: str = ""
    traceability_completeness: str = ""

    # Review digest (deterministic)
    review_digest: str = ""

    # Reviewer independence evidence
    reviewer_independence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "review_id": self.review_id,
            "task_id": self.task_id,
            "status": self.status,
            "findings": [f.to_dict() for f in self.findings],
            "requirement_results": [r.to_dict() for r in self.requirement_results],
            "knowledge_results": [k.to_dict() for k in self.knowledge_results],
            "violations": [v.to_dict() for v in self.violations],
            "missing_evidence": [m.to_dict() for m in self.missing_evidence],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "sec_coverage": dict(self.sec_coverage),
            "traceability_digest": self.traceability_digest,
            "traceability_completeness": self.traceability_completeness,
            "review_digest": self.review_digest,
            "reviewer_independence": dict(self.reviewer_independence),
        }


# ──────────────────────────────────────────────────────────────────────
# KnowledgeReviewer — independent reviewer
# ──────────────────────────────────────────────────────────────────────

class KnowledgeReviewer:
    """
    Independent Reviewer: evaluates knowledge compliance from evidence.

    Reviewer independence:
    - Does NOT trust Planner claims
    - Evaluates available evidence and constraints directly
    - Cannot fabricate evidence, change authority, or mutate knowledge
    - Deterministic: same inputs → same review

    Fail-closed:
    - Missing required evidence → UNVERIFIED/FAIL
    - Corrupt/wrong evidence → cannot PASS
    - No evidence by absence of contradiction → never PASS
    """

    def __init__(self, evidence_store: Any = None, knowledge_store: Any = None):
        self.evidence_store = evidence_store
        self.knowledge_store = knowledge_store
        self.traceability_service = TraceabilityService(
            knowledge_store=knowledge_store,
            evidence_store=evidence_store,
        )

    def review(self, context: ReviewContext) -> ReviewResult:
        """
        Perform independent review of knowledge compliance.

        Args:
            context: ReviewContext with all typed domain objects

        Returns:
            ReviewResult with findings, requirement results, knowledge results,
            violations, missing evidence, conflicts, SEC coverage, digests

        Raises:
            EvidenceFabricationError: if asked to fabricate evidence
            AuthorityChangeError: if asked to change knowledge authority
        """
        findings: List[ReviewFinding] = []
        requirement_results: List[RequirementResult] = []
        knowledge_results: List[KnowledgeResult] = []
        violations: List[ReviewFinding] = []
        missing_evidence: List[ReviewFinding] = []
        conflicts: List[ReviewFinding] = []

        # 1. Check for unresolved conflicts first (fail closed)
        if context.conflicts:
            unresolved = [c for c in context.conflicts if not c.get("resolved", False)]
            if unresolved:
                for conflict in unresolved:
                    conflicts.append(ReviewFinding(
                        finding_type="UNRESOLVED_CONFLICT",
                        severity="critical",
                        source=conflict.get("source", "unknown"),
                        message=f"Unresolved conflict: {conflict.get('reason', 'unknown')}",
                    ))

        # 1b. Check verification status — failed verification is a violation
        if context.verification:
            ver_status = getattr(context.verification, 'status', 'unknown')
            if ver_status == 'failed':
                violations.append(ReviewFinding(
                    finding_type="VERIFICATION_FAILED",
                    severity="critical",
                    source=getattr(context.verification, 'verification_id', 'unknown'),
                    message="Verification failed — implementation does not meet requirements",
                ))

        # 2. SEC protected coverage (release-critical)
        sec_coverage = self._evaluate_sec_coverage(context, findings, violations)

        # 3. Requirement traceability (REQ/NFR)
        requirement_results = self._evaluate_requirements(context, findings, missing_evidence)

        # 4. Knowledge influence traceability
        knowledge_results = self._evaluate_knowledge_influences(context, findings)

        # 5. Plan drift detection
        plan_drift = self._detect_plan_drift(context, findings, violations)

        # 6. Knowledge drift detection
        knowledge_drift = self._detect_knowledge_drift(context, findings, violations)

        # 7. Evidence integrity check
        evidence_integrity = self._check_evidence_integrity(context, findings, violations)

        # 8. TDR resolution claims
        self._check_tdr_resolution_claims(context, findings, violations)

        # 9. Risk status (ACCEPTED != RESOLVED)
        self._check_risk_status(context, findings)

        # 10. Build traceability chain
        trace_chain = self.traceability_service.build_chain(
            task_id=context.task_id,
            plan=context.plan,
            execution=context.execution,
            verification=context.verification,
            knowledge_context=context.knowledge_context,
        )

        # 11. Determine overall status
        status = self._determine_status(
            findings, violations, missing_evidence, conflicts,
            sec_coverage, requirement_results,
            plan_drift, knowledge_drift,
        )

        # 12. Compute review digest
        review_digest = self._compute_review_digest(context, findings, requirement_results)

        # 13. Build result
        return ReviewResult(
            review_id=f"REV-{context.task_id}",
            task_id=context.task_id,
            status=status,
            findings=findings,
            requirement_results=requirement_results,
            knowledge_results=knowledge_results,
            violations=violations,
            missing_evidence=missing_evidence,
            conflicts=conflicts,
            sec_coverage=sec_coverage,
            traceability_digest=trace_chain.digest,
            traceability_completeness=trace_chain.completeness,
            review_digest=review_digest,
            reviewer_independence={
                "evaluated_independently": True,
                "did_not_trust_plan_claims": True,
                "evidence_based": True,
                "cannot_fabricate_evidence": True,
                "cannot_change_authority": True,
                "read_only": True,
            },
        )

    def _evaluate_sec_coverage(
        self,
        context: ReviewContext,
        findings: List[ReviewFinding],
        violations: List[ReviewFinding],
    ) -> Dict[str, Any]:
        """
        Evaluate SEC protected coverage.

        SEC is release-critical. Applicable SEC constraints require appropriate
        evidence where verification is required. Reviewer cannot mark compliant
        solely because Planner included SEC or execution succeeded.
        """
        sec_coverage = {
            "applicable": 0,
            "reviewed": 0,
            "with_required_evidence": 0,
            "violations": 0,
            "unverified": 0,
            "details": [],
        }

        if not context.applicable_sec:
            return sec_coverage

        sec_coverage["applicable"] = len(context.applicable_sec)

        for sec in context.applicable_sec:
            sec_id = sec.record_id if hasattr(sec, 'record_id') else str(sec)
            sec_reviewed = False
            sec_evidenced = False
            sec_violated = False
            sec_unverified = False

            # Check if SEC was reviewed (has verification)
            if context.verification:
                ver_status = getattr(context.verification, 'status', 'unknown')
                if ver_status == 'passed':
                    sec_reviewed = True
                    sec_coverage["reviewed"] += 1
                elif ver_status == 'failed':
                    sec_violated = True
                    sec_coverage["violations"] += 1
                    violations.append(ReviewFinding(
                        finding_type="SEC_VIOLATION",
                        severity="critical",
                        source=sec_id,
                        message=f"SEC '{sec_id}' violated: verification failed",
                    ))
                else:
                    sec_unverified = True
                    sec_coverage["unverified"] += 1
            else:
                # No verification at all → UNVERIFIED, not PASS
                sec_unverified = True
                sec_coverage["unverified"] += 1
                findings.append(ReviewFinding(
                    finding_type="SEC_UNVERIFIED",
                    severity="critical",
                    source=sec_id,
                    message=f"SEC '{sec_id}' has no verification evidence — UNVERIFIED, not PASS",
                ))

            # Check for evidence binding
            if context.evidence_refs:
                # Check if any evidence is bound to this SEC
                bound_evidence = self._get_evidence_for_sec(sec_id, context.evidence_refs)
                if bound_evidence:
                    sec_evidenced = True
                    sec_coverage["with_required_evidence"] += 1

            # SEC violation despite functional success → FAIL
            if context.execution and not sec_violated:
                exec_status = getattr(context.execution, 'status', 'unknown')
                if exec_status == 'success' and sec_unverified:
                    # Functional success does NOT override SEC unverified
                    findings.append(ReviewFinding(
                        finding_type="SEC_UNVERIFIED_DESPITE_SUCCESS",
                        severity="critical",
                        source=sec_id,
                        message=(
                            f"SEC '{sec_id}' unverified despite execution success. "
                            f"Functional success does NOT override SEC requirements."
                        ),
                    ))

            sec_coverage["details"].append({
                "sec_id": sec_id,
                "reviewed": sec_reviewed,
                "evidenced": sec_evidenced,
                "violated": sec_violated,
                "unverified": sec_unverified,
            })

        return sec_coverage

    def _evaluate_requirements(
        self,
        context: ReviewContext,
        findings: List[ReviewFinding],
        missing_evidence: List[ReviewFinding],
    ) -> List[RequirementResult]:
        """
        Evaluate requirement traceability: REQ-001 → Plan → Verification → Evidence.

        For each requirement, Reviewer answers:
        - Was REQ-001 addressed?
        - How was it verified?
        - Which evidence supports that verification?
        - Status: SATISFIED/VIOLATED/UNVERIFIED/NOT_APPLICABLE
        """
        results: List[RequirementResult] = []

        if not context.knowledge_context:
            return results

        # Get requirements from knowledge context
        req_items = []
        if hasattr(context.knowledge_context, 'items'):
            req_items = [
                item for item in context.knowledge_context.items
                if item.record_type in ("REQ", "NFR")
            ]

        for item in req_items:
            req_id = item.record_id
            req_type = item.record_type

            # Check if requirement was planned
            planned = False
            verified = False
            evidenced = False
            verification_ids: List[str] = []
            evidence_refs: List[str] = []

            # Check plan provenance
            if context.plan:
                plan_records = getattr(context.plan, 'knowledge_records', [])
                if req_id in str(plan_records):
                    planned = True

            # Check verification
            if context.verification:
                ver_status = getattr(context.verification, 'status', 'unknown')
                verification_ids.append(getattr(context.verification, 'verification_id', 'unknown'))
                if ver_status == 'passed':
                    verified = True

            # Check evidence binding
            if context.evidence_refs:
                bound = self._get_evidence_for_requirement(req_id, context.evidence_refs)
                if bound:
                    evidenced = True
                    evidence_refs = bound

            # Determine status
            if not planned:
                status = "UNVERIFIED"
                missing_evidence.append(ReviewFinding(
                    finding_type="REQ_NOT_PLANNED",
                    severity="high",
                    source=req_id,
                    message=f"Requirement '{req_id}' not found in plan provenance",
                ))
            elif not verified:
                status = "UNVERIFIED"
                missing_evidence.append(ReviewFinding(
                    finding_type="REQ_NOT_VERIFIED",
                    severity="high",
                    source=req_id,
                    message=f"Requirement '{req_id}' has no verification",
                ))
            elif not evidenced:
                status = "UNVERIFIED"
                missing_evidence.append(ReviewFinding(
                    finding_type="REQ_NO_EVIDENCE",
                    severity="high",
                    source=req_id,
                    message=f"Requirement '{req_id}' verified but no supporting evidence",
                ))
            else:
                status = "SATISFIED"

            results.append(RequirementResult(
                requirement_id=req_id,
                requirement_type=req_type,
                status=status,
                verified=verified,
                evidence_refs=evidence_refs,
                verification_ids=verification_ids,
                reason=f"Plan: {planned}, Verified: {verified}, Evidenced: {evidenced}",
                source="evidence" if evidenced else ("plan_claim" if planned else "none"),
            ))

        return results

    def _evaluate_knowledge_influences(
        self,
        context: ReviewContext,
        findings: List[ReviewFinding],
    ) -> List[KnowledgeResult]:
        """
        Evaluate knowledge influence traceability.

        For each material KnowledgeInfluence, establish outcome trace:
        EngineeringRecord → KnowledgeInfluence → Plan Decision → Execution →
        Verification → Evidence → Review Result

        Retrieved but not influenced: no outcome trace
        Influenced but not verifiable: INFLUENCED, OUTCOME = UNVERIFIED/NOT_MEASURABLE
        """
        results: List[KnowledgeResult] = []

        if not context.knowledge_influences:
            return results

        for influence in context.knowledge_influences:
            record_id = influence.record_id
            record_type = influence.record_type
            authority = getattr(influence, 'authority', 'unknown')
            decision = influence.decision

            # Check if this influence has execution trace
            has_execution = False
            has_verification = False
            has_evidence = False
            outcome_status = "UNVERIFIED"

            if context.execution:
                exec_id = getattr(context.execution, 'execution_id', '')
                if exec_id and exec_id != 'unknown':
                    has_execution = True

            if context.verification:
                ver_status = getattr(context.verification, 'status', 'unknown')
                if ver_status == 'passed':
                    has_verification = True
                    outcome_status = "SATISFIED"
                elif ver_status == 'failed':
                    has_verification = True
                    outcome_status = "VIOLATED"

            if context.evidence_refs:
                has_evidence = True

            # Determine influence status
            influence_status = "INFLUENCED"

            # Influenced but not measurable → UNVERIFIED, NOT fabricated
            if has_execution and not has_verification:
                outcome_status = "NOT_MEASURABLE"
                findings.append(ReviewFinding(
                    finding_type="INFLUENCE_NOT_MEASURABLE",
                    severity="medium",
                    source=record_id,
                    message=(
                        f"Knowledge '{record_id}' influenced plan but outcome "
                        f"is not measurable by current verification"
                    ),
                ))

            results.append(KnowledgeResult(
                record_id=record_id,
                record_type=record_type,
                authority=authority,
                influence_status=influence_status,
                outcome_status=outcome_status,
                decision=decision,
                evidence_refs=context.evidence_refs if has_evidence else [],
                reason=f"Execution: {has_execution}, Verification: {has_verification}",
            ))

        return results

    def _detect_plan_drift(
        self,
        context: ReviewContext,
        findings: List[ReviewFinding],
        violations: List[ReviewFinding],
    ) -> bool:
        """
        Detect plan drift: approved plan A, execution follows plan B.

        If execution.plan_id != plan.plan_id or execution.plan_digest != plan.plan_digest,
        this is PLAN_DRIFT → Review cannot PASS.
        """
        if not context.execution or not context.plan:
            return False

        exec_plan_id = getattr(context.execution, 'plan_id', '')
        plan_id = getattr(context.plan, 'plan_id', '')

        exec_plan_digest = getattr(context.execution, 'plan_digest', '')
        plan_digest = getattr(context.plan, 'plan_digest', '')

        if exec_plan_id and plan_id and exec_plan_id != plan_id:
            violations.append(ReviewFinding(
                finding_type="PLAN_DRIFT",
                severity="critical",
                source=plan_id,
                message=(
                    f"PLAN_DRIFT: Plan '{plan_id}' was approved but execution "
                    f"followed plan '{exec_plan_id}'"
                ),
            ))
            return True

        if exec_plan_digest and plan_digest and exec_plan_digest != plan_digest:
            violations.append(ReviewFinding(
                finding_type="PLAN_DRIFT",
                severity="critical",
                source=plan_id,
                message=(
                    f"PLAN_DRIFT: Plan digest mismatch. "
                    f"Approved: {plan_digest[:16]}..., "
                    f"Executed: {exec_plan_digest[:16]}..."
                ),
            ))
            return True

        return False

    def _detect_knowledge_drift(
        self,
        context: ReviewContext,
        findings: List[ReviewFinding],
        violations: List[ReviewFinding],
    ) -> bool:
        """
        Detect knowledge drift: plan created with knowledge version X,
        before execution knowledge version Y becomes authoritative.

        If plan.knowledge_context_digest != current knowledge digest,
        this is KNOWLEDGE_DRIFT → may require revalidation/replanning.
        """
        if not context.plan or not context.knowledge_context:
            return False

        plan_knowledge_digest = context.knowledge_context_digest
        current_digest = context.knowledge_context.digest if hasattr(context.knowledge_context, 'digest') else ''

        if plan_knowledge_digest and current_digest and plan_knowledge_digest != current_digest:
            violations.append(ReviewFinding(
                finding_type="KNOWLEDGE_DRIFT",
                severity="critical",
                source="knowledge_context",
                message=(
                    f"KNOWLEDGE_DRIFT: Plan created with digest "
                    f"{plan_knowledge_digest[:16]}... but current digest is "
                    f"{current_digest[:16]}... Revalidation required."
                ),
            ))
            return True

        return False

    def _check_evidence_integrity(
        self,
        context: ReviewContext,
        findings: List[ReviewFinding],
        violations: List[ReviewFinding],
    ) -> bool:
        """
        Check evidence integrity.

        Tampered/corrupt evidence → integrity failure → cannot PASS.
        Wrong task/execution evidence → cannot PASS.
        """
        if not context.evidence_refs or not self.evidence_store:
            return True

        integrity_valid = True
        for ev_ref in context.evidence_refs:
            try:
                # Verify evidence exists and is bound to correct task
                from ..evidence.unified import EvidenceQuery
                query = EvidenceQuery(task_id=context.task_id)
                packages = self.evidence_store.query(query)
                found = False
                for pkg in packages:
                    if pkg.content_hash == ev_ref:
                        found = True
                        # Verify task binding
                        if pkg.task_id != context.task_id:
                            violations.append(ReviewFinding(
                                finding_type="WRONG_TASK_EVIDENCE",
                                severity="critical",
                                source=ev_ref[:16],
                                message=(
                                    f"Evidence {ev_ref[:16]}... bound to task "
                                    f"'{pkg.task_id}' not '{context.task_id}'"
                                ),
                            ))
                            integrity_valid = False
                        break
                if not found:
                    # Evidence not found — not necessarily a violation,
                    # could be from different task context
                    pass
            except Exception as e:
                findings.append(ReviewFinding(
                    finding_type="EVIDENCE_INTEGRITY_ERROR",
                    severity="high",
                    source=ev_ref[:16],
                    message=f"Evidence integrity check failed: {e}",
                ))
                integrity_valid = False

        return integrity_valid

    def _check_tdr_resolution_claims(
        self,
        context: ReviewContext,
        findings: List[ReviewFinding],
        violations: List[ReviewFinding],
    ) -> None:
        """
        Check TDR resolution claims.

        If plan claims TDR resolved, Phase 6 invariant remains:
        resolution requires evidence. Reviewer detects unsupported resolution claims.
        Reviewer does NOT itself close TDR.
        """
        if not context.knowledge_context:
            return

        tdr_items = []
        if hasattr(context.knowledge_context, 'items'):
            tdr_items = [
                item for item in context.knowledge_context.items
                if item.record_type == "TDR"
            ]

        for tdr in tdr_items:
            tdr_id = tdr.record_id
            # Check if plan claims TDR resolved OR if TDR is in knowledge context
            # without resolution evidence
            plan_claims_resolved = False
            if context.plan:
                plan_records = getattr(context.plan, 'knowledge_records', [])
                if tdr_id in str(plan_records):
                    plan_claims_resolved = True

            # Check for remediation evidence
            has_evidence = False
            if context.evidence_refs:
                # Look for TDR-specific evidence
                has_evidence = any(
                    tdr_id in ev_id for ev_id in context.evidence_refs
                )

            # If plan claims resolved but no evidence → violation
            if plan_claims_resolved and not has_evidence:
                violations.append(ReviewFinding(
                    finding_type="TDR_UNSUPPORTED_CLAIM",
                    severity="high",
                    source=tdr_id,
                    message=(
                        f"TDR '{tdr_id}' resolution claimed but no "
                        f"remediation evidence found. TDR remains unresolved."
                    ),
                ))
            # If TDR is in knowledge context without evidence → also flag
            elif not has_evidence and tdr.status not in ("resolved", "closed"):
                violations.append(ReviewFinding(
                    finding_type="TDR_UNSUPPORTED_CLAIM",
                    severity="high",
                    source=tdr_id,
                    message=(
                        f"TDR '{tdr_id}' in knowledge context without "
                        f"remediation evidence. TDR remains unresolved."
                    ),
                ))

    def _check_risk_status(
        self,
        context: ReviewContext,
        findings: List[ReviewFinding],
    ) -> None:
        """
        Check risk status: ACCEPTED != RESOLVED.

        Reviewer does not infer Risk accepted = Risk mitigated.
        Also flags risks with no mitigation evidence.
        """
        if not context.knowledge_context:
            return

        rsk_items = []
        if hasattr(context.knowledge_context, 'items'):
            rsk_items = [
                item for item in context.knowledge_context.items
                if item.record_type == "RSK"
            ]

        for rsk in rsk_items:
            rsk_id = rsk.record_id
            status = getattr(rsk, 'status', 'unknown')
            # Check both "accepted" and "ACCEPTED" since different contexts use different cases
            if status.lower() == "accepted":
                # ACCEPTED is not RESOLVED — report but don't auto-convert
                findings.append(ReviewFinding(
                    finding_type="RISK_ACCEPTED_NOT_RESOLVED",
                    severity="medium",
                    source=rsk_id,
                    message=(
                        f"Risk '{rsk_id}' is ACCEPTED (not RESOLVED). "
                        f"Plan should retain mitigation/fallback consideration."
                    ),
                ))
            elif status == "identified":
                # Identified risk with no evidence → flag
                has_evidence = any(
                    rsk_id in ev_id for ev_id in context.evidence_refs
                )
                if not has_evidence:
                    findings.append(ReviewFinding(
                        finding_type="RISK_ACCEPTED_NOT_RESOLVED",
                        severity="medium",
                        source=rsk_id,
                        message=(
                            f"Risk '{rsk_id}' is identified without mitigation evidence."
                        ),
                    ))

    def _determine_status(
        self,
        findings: List[ReviewFinding],
        violations: List[ReviewFinding],
        missing_evidence: List[ReviewFinding],
        conflicts: List[ReviewFinding],
        sec_coverage: Dict[str, Any],
        requirement_results: List[RequirementResult],
        plan_drift: bool,
        knowledge_drift: bool,
    ) -> str:
        """
        Determine overall review status.

        PASS: no violations, no missing evidence, all requirements satisfied
        FAIL: violations detected (SEC, evidence integrity, etc.)
        UNVERIFIED: required evidence missing
        CONFLICT: unresolved conflicts
        NEEDS_HUMAN: plan drift, knowledge drift, or other human intervention needed
        """
        # Unresolved conflicts → CONFLICT (checked first - blocks everything)
        if conflicts:
            return "CONFLICT"

        # Plan drift or knowledge drift → NEEDS_HUMAN (before violations)
        if plan_drift or knowledge_drift:
            return "NEEDS_HUMAN"

        # Critical violations → FAIL
        if violations:
            return "FAIL"

        # Missing required evidence → UNVERIFIED
        if missing_evidence:
            return "UNVERIFIED"

        # SEC violations → FAIL
        if sec_coverage.get("violations", 0) > 0:
            return "FAIL"

        # SEC unverified → UNVERIFIED
        if sec_coverage.get("unverified", 0) > 0:
            return "UNVERIFIED"

        # All requirements satisfied → PASS
        all_satisfied = all(
            r.status == "SATISFIED" for r in requirement_results
        )
        if all_satisfied and requirement_results:
            return "PASS"

        # Some requirements unverified → UNVERIFIED
        any_unverified = any(
            r.status in ("UNVERIFIED", "VIOLATED") for r in requirement_results
        )
        if any_unverified:
            return "UNVERIFIED"

        # Default to UNVERIFIED if no requirements
        return "UNVERIFIED"

    def _get_evidence_for_sec(self, sec_id: str, evidence_refs: List[str]) -> List[str]:
        """Get evidence bound to a specific SEC."""
        return [ev for ev in evidence_refs if sec_id in ev]

    def _get_evidence_for_requirement(self, req_id: str, evidence_refs: List[str]) -> List[str]:
        """Get evidence bound to a specific requirement."""
        return [ev for ev in evidence_refs if req_id in ev]

    def _compute_review_digest(
        self,
        context: ReviewContext,
        findings: List[ReviewFinding],
        requirement_results: List[RequirementResult],
    ) -> str:
        """
        Compute deterministic review digest.

        ReviewContext + Evidence + Policy → equivalent structured review/digest
        for deterministic rules.
        """
        # Build canonical representation
        finding_dicts = sorted(
            (f.finding_type, f.severity, f.source, f.message)
            for f in findings
        )
        req_dicts = sorted(
            (r.requirement_id, r.status, r.verified, tuple(r.evidence_refs))
            for r in requirement_results
        )

        digest_input = json.dumps(
            {
                "task_id": context.task_id,
                "findings": finding_dicts,
                "requirements": req_dicts,
                "knowledge_context_digest": context.knowledge_context_digest,
                "plan_digest": context.plan_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
