# harness/knowledge/consistency.py — V3.3 Phase 11: Bidirectional Consistency Check
"""
Consistency checking between Engineering Knowledge and Learning.

Phase 11 Implementation — TASK-066 (Bidirectional Consistency Check)

Architecture:
- ConsistencyChecker detects contradictions between knowledge and learning
- Does NOT mutate canonical state
- Flags inconsistencies for human review
- Knowledge is advisory; Learning is empirically derived
- When they conflict, the conflict is flagged, not silently resolved

Critical: Learning cannot override Engineering Knowledge.
If learning-derived suggestion conflicts with existing accepted knowledge:
existing knowledge wins. Suggestion remains CONFLICTING/BLOCKED/NEEDS_REVIEW.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .learning_bridge import (
    EngineeringKnowledgeSuggestion,
    SUGGESTION_STATUS_CONFLICTING,
    SUGGESTION_STATUS_BLOCKED,
    SUGGESTION_STATUS_NEEDS_REVIEW,
)
from .provenance import AUTHORITY_ACCEPTED


class ConsistencyError(Exception):
    """Raised on consistency check errors."""


@dataclass
class ConsistencyFinding:
    """A single consistency finding."""
    finding_type: str  # "KNOWLEDGE_LEARNING_CONTRADICTION", "SUGGESTION_CONFLICT", etc.
    severity: str  # "critical", "high", "medium", "low"
    source: str  # what triggered the finding
    message: str
    conflicting_records: List[str] = field(default_factory=list)
    resolution: str = ""

    def to_dict(self) -> dict:
        return {
            "finding_type": self.finding_type,
            "severity": self.severity,
            "source": self.source,
            "message": self.message,
            "conflicting_records": list(self.conflicting_records),
            "resolution": self.resolution,
        }


@dataclass
class ConsistencyReport:
    """Result of a consistency check between knowledge and learning."""
    findings: List[ConsistencyFinding] = field(default_factory=list)
    digest: str = ""
    status: str = ""  # "CONSISTENT", "INCONSISTENT", "NEEDS_REVIEW"

    def to_dict(self) -> dict:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "digest": self.digest,
            "status": self.status,
        }


class ConsistencyChecker:
    """
    Checks consistency between Engineering Knowledge and Learning.

    Does NOT mutate canonical state.
    Flags inconsistencies for human review.
    """

    def __init__(self, knowledge_store: Any = None):
        """
        Initialize the consistency checker.

        Args:
            knowledge_store: KnowledgeStore instance
        """
        self.knowledge_store = knowledge_store

    def check(
        self,
        knowledge_context: Any = None,
        learning_artifacts: Optional[List[Any]] = None,
        suggestions: Optional[List[EngineeringKnowledgeSuggestion]] = None,
    ) -> ConsistencyReport:
        """
        Check consistency between knowledge and learning.

        Args:
            knowledge_context: EngineeringKnowledgeContext
            learning_artifacts: List of learning artifacts (experiences, strategies)
            suggestions: List of learning-derived suggestions

        Returns:
            ConsistencyReport with findings and status
        """
        findings: List[ConsistencyFinding] = []

        # Check 1: Suggestions vs existing knowledge
        if suggestions:
            for suggestion in suggestions:
                suggestion_findings = self._check_suggestion_consistency(suggestion)
                findings.extend(suggestion_findings)

        # Check 2: Learning artifacts vs knowledge
        if learning_artifacts and knowledge_context:
            artifact_findings = self._check_artifact_consistency(
                learning_artifacts, knowledge_context
            )
            findings.extend(artifact_findings)

        # Check 3: Knowledge context internal consistency
        if knowledge_context:
            context_findings = self._check_knowledge_context_consistency(knowledge_context)
            findings.extend(context_findings)

        # Determine status
        if any(f.severity == "critical" for f in findings):
            status = "INCONSISTENT"
        elif findings:
            status = "NEEDS_REVIEW"
        else:
            status = "CONSISTENT"

        # Compute digest
        digest = self._compute_digest(findings)

        return ConsistencyReport(
            findings=findings,
            digest=digest,
            status=status,
        )

    def _check_suggestion_consistency(
        self,
        suggestion: EngineeringKnowledgeSuggestion,
    ) -> List[ConsistencyFinding]:
        """Check a suggestion against existing knowledge."""
        findings = []

        if suggestion.status == SUGGESTION_STATUS_CONFLICTING:
            findings.append(ConsistencyFinding(
                finding_type="SUGGESTION_CONFLICT",
                severity="high",
                source=suggestion.suggestion_id,
                message=(
                    f"Suggestion '{suggestion.suggestion_id}' conflicts with "
                    f"existing knowledge '{suggestion.conflict_with}'"
                ),
                conflicting_records=[suggestion.conflict_with or ""],
                resolution="Existing accepted knowledge wins. Suggestion requires human review.",
            ))

        return findings

    def _check_artifact_consistency(
        self,
        learning_artifacts: List[Any],
        knowledge_context: Any,
    ) -> List[ConsistencyFinding]:
        """Check learning artifacts against knowledge context."""
        findings = []

        for artifact in learning_artifacts:
            # Check if artifact contradicts knowledge contradictions
            knowledge_conflicts = getattr(knowledge_context, 'retrieval_result', None)
            if knowledge_conflicts and hasattr(knowledge_conflicts, 'conflicts'):
                conflicts = knowledge_conflicts.conflicts
                unresolved = [c for c in conflicts if not c.get("resolved", False)]
                if unresolved:
                    findings.append(ConsistencyFinding(
                        finding_type="KNOWLEDGE_CONFLICT_UNRESOLVED",
                        severity="critical",
                        source=getattr(artifact, 'strategy_id', str(id(artifact))),
                        message="Learning artifact built on knowledge context with unresolved conflicts",
                        conflicting_records=[c.get("records", []) for c in unresolved],
                        resolution="Resolve knowledge conflicts before using context for learning.",
                    ))

        return findings

    def _check_knowledge_context_consistency(
        self,
        knowledge_context: Any,
    ) -> List[ConsistencyFinding]:
        """Check internal consistency of knowledge context."""
        findings = []

        # Check for superseded records presented as current
        items = getattr(knowledge_context, 'items', [])
        for item in items:
            if getattr(item, 'is_historical', False):
                findings.append(ConsistencyFinding(
                    finding_type="SUPERSEDED_AS_CURRENT",
                    severity="medium",
                    source=item.record_id,
                    message=f"Record '{item.record_id}' is superseded/deprecated but in context",
                    conflicting_records=[item.record_id],
                    resolution="Use current version of the record.",
                ))

        return findings

    def _compute_digest(self, findings: List[ConsistencyFinding]) -> str:
        """Compute deterministic digest of findings."""
        finding_dicts = sorted(
            (f.finding_type, f.severity, f.source, f.message)
            for f in findings
        )
        digest_input = json.dumps(
            {"findings": finding_dicts},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()

    def flag_inconsistency(
        self,
        finding: ConsistencyFinding,
    ) -> Dict[str, Any]:
        """
        Flag an inconsistency for human review.

        Returns a structured flag record.
        """
        return {
            "type": "INCONSISTENCY_FLAG",
            "finding": finding.to_dict(),
            "requires_human_review": True,
            "auto_resolved": False,
        }
