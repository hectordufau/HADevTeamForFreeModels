# harness/knowledge/context.py — V3.3 Phase 8: KnowledgeContextManager
"""
Bounded context assembly integrating Engineering Knowledge into ContextManager.

Phase 8 Implementation — TASK-060 (ContextManager Integration)

Architecture:
- Extends ContextManager so Engineering Knowledge is a distinct context layer
- ContextManager does NOT own knowledge — KnowledgeStore remains canonical
- Engineering Knowledge context is separate from Learning context
- Evidence references are preserved as IDs, never copied
- Deterministic ordering: Governance → Security Authority → Authoritative Eng → Accepted Eng → Task Requirements → Other Eng → Learning → Exploration

Three-domain separation:
- EngineeringKnowledgeContext: PRD, NFR, ADR, TDR, RSK, SEC, RCA
- LearningContext: StructuredExperience, FailureLesson, Strategy, DecisionImpact
- EvidenceContext: EvidencePackage references (by ID only)
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .retrieval import (
    KnowledgeRetriever,
    KnowledgeQuery,
    RetrievalResult,
    RetrievalItem,
    RetrievalMetrics,
    ContextOverflowError,
    KnowledgeConflictError,
)
from .records import EngineeringRecord
from .risk_security import (
    PRECEDENCE_GOVERNANCE,
    PRECEDENCE_SECURITY_AUTHORITY,
    PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE,
    PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE,
    PRECEDENCE_TASK_REQUIREMENTS,
    PRECEDENCE_LEARNED_KNOWLEDGE,
    PRECEDENCE_EXPLORATION,
    AuthorityPrecedenceResolver,
)


class KnowledgeContextError(Exception):
    """Raised on knowledge context errors."""


# ──────────────────────────────────────────────────────────────────────
# ContextItem — a single item in the knowledge context
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ContextItem:
    """
    A single item in the knowledge context.

    Every item retains provenance: source domain, record ID, record type,
    authority, status, retrieval reason.
    """

    record_id: str
    record_type: str
    authority: str
    status: str
    content: str  # bounded projection or summary
    source_domain: str  # "engineering", "learning", "evidence"
    retrieval_reason: str
    relevance: float = 0.0
    precedence: int = 0  # from PRECEDENCE_* constants
    is_mandatory: bool = False
    is_historical: bool = False
    evidence_refs: List[str] = field(default_factory=list)  # evidence IDs, not content
    relationships: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "record_type": self.record_type,
            "authority": self.authority,
            "status": self.status,
            "content": self.content,
            "source_domain": self.source_domain,
            "retrieval_reason": self.retrieval_reason,
            "relevance": round(self.relevance, 4),
            "precedence": self.precedence,
            "is_mandatory": self.is_mandatory,
            "is_historical": self.is_historical,
            "evidence_refs": list(self.evidence_refs),
            "relationships": list(self.relationships),
        }


# ──────────────────────────────────────────────────────────────────────
# EngineeringKnowledgeContext — bounded Engineering Knowledge context
# ──────────────────────────────────────────────────────────────────────

@dataclass
class EngineeringKnowledgeContext:
    """
    Bounded Engineering Knowledge context layer.

    Contains retrieved Engineering Records with provenance.
    Separate from LearningContext and EvidenceContext.
    """

    items: List[ContextItem] = field(default=list)
    retrieval_result: Optional[RetrievalResult] = None
    digest: str = ""

    def to_dict(self) -> dict:
        return {
            "items": [item.to_dict() for item in self.items],
            "retrieval_result": self.retrieval_result.to_dict() if self.retrieval_result else None,
            "digest": self.digest,
        }

    def compute_digest(self) -> str:
        """Compute deterministic SHA-256 digest."""
        items_sorted = sorted(
            (item.record_id, item.record_type, item.authority, item.status,
             item.source_domain, item.retrieval_reason, item.precedence)
            for item in self.items
        )
        digest_input = json.dumps(
            {"items": items_sorted},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# LearningContext — Learning Knowledge context (separate from Engineering)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class LearningContext:
    """
    Learning Knowledge context layer.

    Contains StructuredExperience, FailureLesson, Strategy, DecisionImpact.
    Separate from EngineeringKnowledgeContext.
    """

    items: List[ContextItem] = field(default=list)
    digest: str = ""

    def to_dict(self) -> dict:
        return {
            "items": [item.to_dict() for item in self.items],
            "digest": self.digest,
        }

    def compute_digest(self) -> str:
        """Compute deterministic SHA-256 digest."""
        items_sorted = sorted(
            (item.record_id, item.record_type, item.authority, item.status,
             item.source_domain, item.retrieval_reason)
            for item in self.items
        )
        digest_input = json.dumps(
            {"items": items_sorted},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# EvidenceContext — Evidence context (by reference, never copied)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class EvidenceContext:
    """
    Evidence context layer.

    Evidence is referenced by ID, never copied into Engineering Knowledge.
    Maintains three-domain separation.
    """

    evidence_refs: List[str] = field(default_factory=list)
    digest: str = ""

    def to_dict(self) -> dict:
        return {
            "evidence_refs": list(self.evidence_refs),
            "digest": self.digest,
        }

    def compute_digest(self) -> str:
        """Compute deterministic SHA-256 digest."""
        refs_sorted = sorted(self.evidence_refs)
        digest_input = json.dumps(
            {"evidence_refs": refs_sorted},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# ContextConflictMetadata — structured conflict metadata
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ContextConflictMetadata:
    """Structured conflict metadata for context."""

    conflict_type: str
    records: List[str]
    reason: str
    resolved: bool
    resolution: str

    def to_dict(self) -> dict:
        return {
            "conflict_type": self.conflict_type,
            "records": list(self.records),
            "reason": self.reason,
            "resolved": self.resolved,
            "resolution": self.resolution,
        }


# ──────────────────────────────────────────────────────────────────────
# KnowledgeContextManager — bounded context assembly
# ──────────────────────────────────────────────────────────────────────

class KnowledgeContextManager:
    """
    Bounded context assembly integrating Engineering Knowledge.

    Extends ContextManager so Engineering Knowledge is a distinct context layer:
    - Task Context
    - Engineering Knowledge Context
    - Learning Context
    - Evidence Context (by reference)

    Deterministic ordering:
    Governance → Security Authority → Authoritative Eng → Accepted Eng →
    Task Requirements → Other Eng → Learning → Exploration

    No timestamp precedence. No role precedence. No model precedence.
    """

    def __init__(self, retriever: KnowledgeRetriever):
        """
        Initialize KnowledgeContextManager.

        Args:
            retriever: KnowledgeRetriever instance
        """
        self.retriever = retriever
        self._precedence_resolver = AuthorityPrecedenceResolver()

    def build_knowledge_context(
        self,
        query: KnowledgeQuery,
        learning_context: Optional[LearningContext] = None,
        evidence_context: Optional[EvidenceContext] = None,
    ) -> EngineeringKnowledgeContext:
        """
        Build bounded Engineering Knowledge context for a task/query.

        Args:
            query: KnowledgeQuery specification
            learning_context: Optional LearningContext (kept separate)
            evidence_context: Optional EvidenceContext (by reference)

        Returns:
            EngineeringKnowledgeContext with items, provenance, digest

        Raises:
            ContextOverflowError: if mandatory knowledge exceeds budget
            KnowledgeConflictError: if unresolved conflicts in selected context
        """
        # Retrieve Engineering Knowledge
        retrieval_result = self.retriever.retrieve(query)

        # Convert RetrievalItems to ContextItems
        context_items = []
        for item in retrieval_result.items:
            context_item = self._retrieval_item_to_context_item(item)
            context_items.append(context_item)

        # Deterministic ordering
        context_items = self._order_context_items(context_items)

        # Build context
        context = EngineeringKnowledgeContext(
            items=context_items,
            retrieval_result=retrieval_result,
        )
        context.digest = context.compute_digest()

        return context

    def build_full_context(
        self,
        query: KnowledgeQuery,
        learning_context: Optional[LearningContext] = None,
        evidence_context: Optional[EvidenceContext] = None,
    ) -> Dict[str, Any]:
        """
        Build full context with all layers.

        Returns dict with:
        - engineering_context: EngineeringKnowledgeContext
        - learning_context: LearningContext (if provided)
        - evidence_context: EvidenceContext (if provided)
        - conflicts: List of ContextConflictMetadata
        - digest: Overall deterministic digest
        """
        # Build Engineering Knowledge context
        eng_context = self.build_knowledge_context(query, learning_context, evidence_context)

        # Collect conflicts
        conflicts = []
        if eng_context.retrieval_result:
            for conflict in eng_context.retrieval_result.conflicts:
                conflicts.append(ContextConflictMetadata(
                    conflict_type=conflict.get("type", "UNKNOWN"),
                    records=conflict.get("records", []),
                    reason=conflict.get("reason", ""),
                    resolved=conflict.get("resolved", False),
                    resolution=conflict.get("resolution", ""),
                ))

        # Deterministic ordering of conflicts
        conflicts.sort(key=lambda c: (c.conflict_type, c.records))

        # Compute overall digest
        overall_digest = self._compute_overall_digest(eng_context, learning_context, evidence_context)

        return {
            "engineering_context": eng_context.to_dict(),
            "learning_context": learning_context.to_dict() if learning_context else None,
            "evidence_context": evidence_context.to_dict() if evidence_context else None,
            "conflicts": [c.to_dict() for c in conflicts],
            "digest": overall_digest,
        }

    def get_knowledge_for_task(
        self,
        task_description: str,
        record_types: Optional[List[str]] = None,
        budget: int = 10,
    ) -> EngineeringKnowledgeContext:
        """
        Get Engineering Knowledge context for a task.

        Args:
            task_description: Task description
            record_types: Optional record type filter
            budget: Maximum records to return

        Returns:
            EngineeringKnowledgeContext
        """
        query = KnowledgeQuery(
            task_description=task_description,
            record_types=record_types or [],
            budget=budget,
        )
        return self.build_knowledge_context(query)

    def get_knowledge_for_decision(
        self,
        decision_id: str,
        budget: int = 10,
    ) -> EngineeringKnowledgeContext:
        """
        Get Engineering Knowledge context for a decision.

        Traces decision back to requirements and evidence.

        Args:
            decision_id: Decision record ID
            budget: Maximum records to return

        Returns:
            EngineeringKnowledgeContext
        """
        # First, trace the decision
        trace_items = self.retriever.trace(decision_id)

        # Build query from trace
        record_ids = [item.record_id for item in trace_items]
        query = KnowledgeQuery(
            record_ids=record_ids,
            budget=budget,
        )

        return self.build_knowledge_context(query)

    def get_knowledge_for_planning(
        self,
        task_description: str,
        components: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
        budget: int = 15,
    ) -> EngineeringKnowledgeContext:
        """
        Get Engineering Knowledge context for planning.

        Retrieves relevant ADR, TDR, RSK, SEC, NFR for planning.

        Args:
            task_description: Task description
            components: Optional component filter
            capabilities: Optional capability filter
            budget: Maximum records to return

        Returns:
            EngineeringKnowledgeContext
        """
        query = KnowledgeQuery(
            task_description=task_description,
            components=components or [],
            capabilities=capabilities or [],
            record_types=["ADR", "TDR", "RSK", "SEC", "NFR", "REQ"],
            budget=budget,
            max_graph_depth=2,
        )
        return self.build_knowledge_context(query)

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _retrieval_item_to_context_item(self, item: RetrievalItem) -> ContextItem:
        """Convert a RetrievalItem to a ContextItem."""
        # Determine precedence
        precedence = self._get_precedence(item)

        # Build content (bounded projection)
        content = self._build_content_projection(item)

        # Get evidence references (by ID, not content)
        evidence_refs = []
        try:
            record = self.retriever.store.get(item.record_id)
            if record.provenance and record.provenance.evidence_refs:
                evidence_refs = list(record.provenance.evidence_refs)
        except Exception:
            pass

        # Build retrieval reason
        reason = self._build_retrieval_reason(item)

        return ContextItem(
            record_id=item.record_id,
            record_type=item.record_type,
            authority=item.authority,
            status=item.status,
            content=content,
            source_domain="engineering",
            retrieval_reason=reason,
            relevance=item.relevance,
            precedence=precedence,
            is_mandatory=item.is_mandatory,
            is_historical=item.is_historical,
            evidence_refs=evidence_refs,
            relationships=item.relationships,
        )

    def _get_precedence(self, item: RetrievalItem) -> int:
        """Get precedence level for a context item."""
        if item.record_type == "SEC" and item.authority in {"AUTHORITATIVE", "ACCEPTED"}:
            return PRECEDENCE_SECURITY_AUTHORITY
        if item.record_type in ("ADR", "DR") and item.authority == "accepted":
            return PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE
        if item.authority == "accepted":
            return PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE
        if item.source == "task":
            return PRECEDENCE_TASK_REQUIREMENTS
        if item.source == "learning":
            return PRECEDENCE_LEARNED_KNOWLEDGE
        return PRECEDENCE_EXPLORATION

    def _build_content_projection(self, item: RetrievalItem) -> str:
        """Build bounded content projection for a context item."""
        try:
            record = self.retriever.store.get(item.record_id)
            # Bounded projection: title + description summary
            content = f"[{item.record_type}] {record.title}\n{record.description[:200]}"
            if item.is_historical:
                content += "\n[HISTORICAL — superseded/deprecated]"
            return content
        except Exception:
            return f"[{item.record_type}] {item.record_id}"

    def _build_retrieval_reason(self, item: RetrievalItem) -> str:
        """Build deterministic retrieval reason."""
        reasons = []
        if item.source == "explicit":
            reasons.append("explicitly requested")
        elif item.source == "graph_expansion":
            reasons.append(f"graph distance {item.graph_distance}")
        elif item.source == "type_match":
            reasons.append("record type match")
        elif item.source == "tag_match":
            reasons.append("tag match")
        if item.is_mandatory:
            reasons.append("mandatory (SEC/authoritative)")
        if item.is_historical:
            reasons.append("historical")
        return "; ".join(reasons) if reasons else "retrieved"

    def _order_context_items(self, items: List[ContextItem]) -> List[ContextItem]:
        """
        Deterministic ordering of context items.

        Governance → Security Authority → Authoritative Eng → Accepted Eng →
        Task Requirements → Other Eng → Learning → Exploration

        No timestamp precedence. No role precedence. No model precedence.
        """
        def sort_key(item: ContextItem) -> tuple:
            return (
                -item.precedence,  # higher precedence first
                -int(item.is_mandatory),  # mandatory first
                -item.relevance,  # higher relevance first
                item.record_id,  # deterministic tiebreaker
            )

        items.sort(key=sort_key)
        return items

    def _compute_overall_digest(
        self,
        eng_context: EngineeringKnowledgeContext,
        learning_context: Optional[LearningContext],
        evidence_context: Optional[EvidenceContext],
    ) -> str:
        """Compute overall deterministic digest."""
        eng_digest = eng_context.digest if eng_context else ""
        learning_digest = learning_context.digest if learning_context else ""
        evidence_digest = evidence_context.digest if evidence_context else ""

        digest_input = json.dumps(
            {
                "engineering": eng_digest,
                "learning": learning_digest,
                "evidence": evidence_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
