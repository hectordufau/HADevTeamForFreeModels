# harness/knowledge/retrieval.py — V3.3 Phase 8: KnowledgeRetriever
"""
Deterministic, bounded, authority-aware Engineering Knowledge retrieval.

Phase 8 Implementation — TASK-059 (KnowledgeRetriever)

Architecture:
- KnowledgeRetriever is a selection mechanism, NOT a canonical store
- KnowledgeStore remains canonical for Engineering Knowledge
- EngineeringKnowledgeGraph remains derived structural relationships
- Retrieval is read-only: no status updates, no authority changes, no mutations
- Deterministic: same store + graph + query → equivalent ordering across restarts
- Bounded: context_budget limits returned records; never returns complete KB
- Explainable: every returned item has deterministic reasons
- Authority-aware: SEC protected knowledge survives ordinary ranking/budget
- Security precedence: Governance > Security Authority > Authoritative Eng > Accepted Eng > Task Requirements > Learned > Exploration

Three-domain separation:
- Retrieval observes Engineering Knowledge only
- Does NOT touch Learning domain (Experience, Strategy, FailureLesson)
- Does NOT touch Evidence domain (EvidencePackage, hash chains)
- Evidence references are preserved as IDs, never copied
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .records import (
    EngineeringRecord,
    VALID_RECORD_TYPES,
    RELATIONSHIP_CONTRADICTS,
    RELATIONSHIP_SUPERSEDES,
)
from .graph import (
    EngineeringKnowledgeGraph,
    TraversalOptions,
    GraphError,
)
from .contradiction import ContradictionDetector, IntegrityConflict
from .risk_security import (
    SEC_AUTHORITY_RANK,
    VALID_SEC_AUTHORITIES,
    PRECEDENCE_GOVERNANCE,
    PRECEDENCE_SECURITY_AUTHORITY,
    PRECEDENCE_AUTHORITATIVE_ENG_KNOWLEDGE,
    PRECEDENCE_ACCEPTED_ENG_KNOWLEDGE,
    PRECEDENCE_TASK_REQUIREMENTS,
    PRECEDENCE_LEARNED_KNOWLEDGE,
    PRECEDENCE_EXPLORATION,
    AuthorityPrecedenceResolver,
)
from .provenance import (
    AUTHORITY_PROPOSED,
    AUTHORITY_ACCEPTED,
    AUTHORITY_DEPRECATED,
    AUTHORITY_ARCHIVED,
    VALID_AUTHORITY_LEVELS,
)


class RetrievalError(Exception):
    """Raised on retrieval errors."""


class ContextOverflowError(RetrievalError):
    """Raised when mandatory knowledge alone exceeds hard context limit."""

    def __init__(self, mandatory_count: int, budget: int):
        self.mandatory_count = mandatory_count
        self.budget = budget
        super().__init__(
            f"Mandatory knowledge ({mandatory_count} records) exceeds "
            f"hard context limit ({budget}). CONTEXT_OVERFLOW / NEEDS_HUMAN."
        )


class KnowledgeConflictError(RetrievalError):
    """Raised when selected context contains unresolved conflicts."""

    def __init__(self, conflicts: List[Dict[str, Any]]):
        self.conflicts = conflicts
        super().__init__(
            f"KNOWLEDGE_CONFLICT: {len(conflicts)} unresolved conflict(s) "
            f"in selected context. Fail closed."
        )


# ──────────────────────────────────────────────────────────────────────
# KnowledgeQuery — retrieval query specification
# ──────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeQuery:
    """
    Specification for a knowledge retrieval query.

    All fields are optional. A query with no constraints returns
    an empty result (never the complete knowledge base by default).

    Deterministic: same query → same results across restarts.
    No LLM preference. No vector database. No semantic mutation.
    """

    # Explicit seed records (highest priority)
    record_ids: List[str] = field(default_factory=list)

    # Task/project context
    project: str = ""
    task: str = ""
    task_description: str = ""

    # Capability/component context (NOT agent roles)
    capabilities: List[str] = field(default_factory=list)
    components: List[str] = field(default_factory=list)

    # Filter constraints
    record_types: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    authority_requirements: List[str] = field(default_factory=list)
    statuses: List[str] = field(default_factory=list)

    # Relationship constraints
    relationship_constraints: Dict[str, Any] = field(default_factory=dict)

    # Budget
    budget: int = 20  # max records to return (default bounded)
    include_historical: bool = False  # include superseded/deprecated

    # Graph expansion
    max_graph_depth: int = 2
    max_graph_nodes: int = 50
    allowed_relationships: Optional[Set[str]] = None

    # Authority minimum
    min_authority: str = ""  # e.g., AUTHORITY_ACCEPTED

    def __post_init__(self):
        # Validate record_types
        for rt in self.record_types:
            if rt not in VALID_RECORD_TYPES:
                raise RetrievalError(
                    f"Invalid record_type '{rt}'. "
                    f"Must be one of: {', '.join(sorted(VALID_RECORD_TYPES))}"
                )

        # Validate authority_requirements
        for auth in self.authority_requirements:
            if auth not in VALID_AUTHORITY_LEVELS:
                raise RetrievalError(
                    f"Invalid authority '{auth}'. "
                    f"Must be one of: {', '.join(sorted(VALID_AUTHORITY_LEVELS))}"
                )

        # Validate budget
        if not isinstance(self.budget, int) or self.budget < 1:
            raise RetrievalError("budget must be a positive integer (>= 1)")

        # Validate graph bounds
        if not isinstance(self.max_graph_depth, int) or self.max_graph_depth < 0:
            raise RetrievalError("max_graph_depth must be a non-negative integer")
        if not isinstance(self.max_graph_nodes, int) or self.max_graph_nodes < 1:
            raise RetrievalError("max_graph_nodes must be a positive integer")

    def to_dict(self) -> dict:
        return {
            "record_ids": list(self.record_ids),
            "project": self.project,
            "task": self.task,
            "task_description": self.task_description,
            "capabilities": list(self.capabilities),
            "components": list(self.components),
            "record_types": list(self.record_types),
            "tags": list(self.tags),
            "authority_requirements": list(self.authority_requirements),
            "statuses": list(self.statuses),
            "relationship_constraints": dict(self.relationship_constraints),
            "budget": self.budget,
            "include_historical": self.include_historical,
            "max_graph_depth": self.max_graph_depth,
            "max_graph_nodes": self.max_graph_nodes,
            "allowed_relationships": (
                sorted(self.allowed_relationships)
                if self.allowed_relationships is not None
                else None
            ),
            "min_authority": self.min_authority,
        }


# ──────────────────────────────────────────────────────────────────────
# RetrievalItem — a single retrieved record with explanation
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalItem:
    """
    A single retrieved record with deterministic explanation.

    Every returned item is explainable. No opaque score.
    Authority ≠ relevance. A highly authoritative record may be unrelated.
    A highly relevant record may have low authority.
    """

    record_id: str
    record_type: str
    authority: str
    status: str
    relevance: float  # 0.0 to 1.0, deterministic
    reasons: List[str]  # deterministic explanation
    relationships: List[Dict[str, str]]  # related record IDs with relation types
    source: str  # retrieval source: "explicit", "graph_expansion", "type_match", "tag_match", "scope_match"
    is_mandatory: bool = False  # SEC protected, governance, authoritative requirement
    is_historical: bool = False  # superseded/deprecated
    graph_distance: int = -1  # -1 = not from graph, 0 = seed, 1 = direct, 2 = indirect
    estimated_cost: int = 0  # deterministic estimated context cost

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "record_type": self.record_type,
            "authority": self.authority,
            "status": self.status,
            "relevance": round(self.relevance, 4),
            "reasons": list(self.reasons),
            "relationships": list(self.relationships),
            "source": self.source,
            "is_mandatory": self.is_mandatory,
            "is_historical": self.is_historical,
            "graph_distance": self.graph_distance,
            "estimated_cost": self.estimated_cost,
        }


# ──────────────────────────────────────────────────────────────────────
# RetrievalMetrics — deterministic retrieval metrics
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    """Deterministic metrics for a retrieval operation."""

    candidate_count: int = 0
    selected_count: int = 0
    mandatory_count: int = 0
    optional_count: int = 0
    budget_used: int = 0
    budget_remaining: int = 0
    records_dropped_by_budget: int = 0
    conflict_count: int = 0
    graph_expansion_count: int = 0

    def to_dict(self) -> dict:
        return {
            "candidate_count": self.candidate_count,
            "selected_count": self.selected_count,
            "mandatory_count": self.mandatory_count,
            "optional_count": self.optional_count,
            "budget_used": self.budget_used,
            "budget_remaining": self.budget_remaining,
            "records_dropped_by_budget": self.records_dropped_by_budget,
            "conflict_count": self.conflict_count,
            "graph_expansion_count": self.graph_expansion_count,
        }


# ──────────────────────────────────────────────────────────────────────
# RetrievalResult — complete retrieval result
# ──────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    Complete result of a knowledge retrieval operation.

    Contains selected items, metrics, conflicts, and digest.
    Deterministic: same store + graph + query → equivalent result.
    """

    items: List[RetrievalItem] = field(default_factory=list)
    metrics: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    digest: str = ""  # SHA-256 for reproducibility
    query: Optional[KnowledgeQuery] = None

    def to_dict(self) -> dict:
        return {
            "items": [item.to_dict() for item in self.items],
            "metrics": self.metrics.to_dict(),
            "conflicts": list(self.conflicts),
            "digest": self.digest,
            "query": self.query.to_dict() if self.query else None,
        }

    def compute_digest(self) -> str:
        """
        Compute deterministic SHA-256 digest for reproducibility.

        Sorted items + metrics → SHA-256.
        No Python hash(). No filesystem or process-local state.
        """
        items_sorted = sorted(
            (item.record_id, item.record_type, item.authority, item.status,
             round(item.relevance, 4), item.source, item.is_mandatory,
             item.is_historical, item.graph_distance, item.estimated_cost)
            for item in self.items
        )
        metrics_dict = self.metrics.to_dict()
        digest_input = json.dumps(
            {
                "items": items_sorted,
                "metrics": metrics_dict,
                "conflicts": sorted(
                    (c.get("type", ""), tuple(c.get("records", [])))
                    for c in self.conflicts
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# KnowledgeRetriever — deterministic, bounded, explainable retrieval
# ──────────────────────────────────────────────────────────────────────

class KnowledgeRetriever:
    """
    Deterministic, bounded, authority-aware Engineering Knowledge retriever.

    Selection mechanism only — does NOT own knowledge.
    KnowledgeStore remains canonical. EngineeringKnowledgeGraph remains derived.

    Usage:
        store = KnowledgeStore("knowledge.db")
        graph = EngineeringKnowledgeGraph.build_from_store(store)
        retriever = KnowledgeRetriever(store, graph)
        result = retriever.retrieve(KnowledgeQuery(
            record_ids=["REQ-001"],
            record_types=["ADR", "SEC", "NFR"],
            budget=10,
        ))

    Deterministic ranking signals (in order):
    1. Explicit reference (seed record_ids)
    2. Graph distance (0=seed, 1=direct, 2=indirect)
    3. Record type relevance (task-driven, NOT role-driven)
    4. Scope match (project, component, capability)
    5. Authority (accepted > proposed; SEC authority separate)
    6. Status (current > historical)
    7. Structured tag/component match

    Authority ≠ relevance. SEC protected knowledge cannot be dropped
    by ordinary ranking/budget competition.
    """

    # Record type relevance by task context (NOT agent role)
    # These are capability/task-driven priorities, not role-based
    TASK_TYPE_RELEVANCE = {
        "architecture": {"ADR": 1.0, "NFR": 0.9, "SEC": 0.8, "REQ": 0.7, "TDR": 0.6, "RSK": 0.5, "RCA": 0.4, "DR": 0.9, "PRD": 0.6},
        "security": {"SEC": 1.0, "NFR": 0.8, "ADR": 0.7, "RSK": 0.6, "REQ": 0.5, "TDR": 0.4, "RCA": 0.3, "DR": 0.6, "PRD": 0.4},
        "implementation": {"TDR": 1.0, "ADR": 0.9, "SEC": 0.8, "REQ": 0.7, "NFR": 0.6, "RSK": 0.5, "RCA": 0.4, "DR": 0.8, "PRD": 0.5},
        "testing": {"RCA": 1.0, "TDR": 0.9, "RSK": 0.8, "SEC": 0.7, "REQ": 0.6, "ADR": 0.5, "NFR": 0.4, "DR": 0.5, "PRD": 0.4},
        "default": {"ADR": 0.8, "SEC": 0.8, "NFR": 0.7, "REQ": 0.7, "TDR": 0.6, "RSK": 0.6, "RCA": 0.5, "DR": 0.7, "PRD": 0.5},
    }

    # Current statuses (not historical)
    CURRENT_STATUSES = {"accepted", "active", "identified", "draft", "proposed", "defined", "verified", "implemented", "review", "assessed", "mitigated", "acknowledged", "remediation_planned", "in_progress", "corrective_action", "preventive_action", "waived", "expired"}

    def __init__(self, store, graph: EngineeringKnowledgeGraph):
        """
        Initialize KnowledgeRetriever.

        Args:
            store: KnowledgeStore (canonical Engineering Knowledge)
            graph: EngineeringKnowledgeGraph (derived relationships)
        """
        self.store = store
        self.graph = graph
        self._contradiction_detector = ContradictionDetector(graph)
        self._precedence_resolver = AuthorityPrecedenceResolver()

    def retrieve(self, query: KnowledgeQuery) -> RetrievalResult:
        """
        Retrieve records relevant to a query.

        Deterministic: same store + graph + query → equivalent ordering.
        Bounded: never returns more than query.budget records.
        Explainable: every item has deterministic reasons.
        Authority-aware: SEC protected knowledge survives ranking/budget.

        Args:
            query: KnowledgeQuery specification

        Returns:
            RetrievalResult with items, metrics, conflicts, digest

        Raises:
            ContextOverflowError: if mandatory knowledge alone exceeds budget
            KnowledgeConflictError: if selected context has unresolved conflicts
        """
        if not isinstance(query, KnowledgeQuery):
            raise RetrievalError("query must be a KnowledgeQuery instance")

        # Phase 1: Collect candidates
        candidates: Dict[str, RetrievalItem] = {}
        graph_expansion_count = 0

        # 1a. Explicit seed records (highest priority)
        for record_id in query.record_ids:
            try:
                record = self.store.get(record_id)
                item = self._record_to_item(record, source="explicit", graph_distance=0)
                item.reasons.append(f"explicitly requested (seed record_id {record_id})")
                candidates[record_id] = item
            except Exception:
                pass  # Skip nonexistent records

        # 1b. Graph expansion from seeds
        if query.max_graph_depth > 0 and query.max_graph_nodes > 0:
            seed_ids = list(candidates.keys())
            for seed_id in seed_ids:
                traversal = TraversalOptions(
                    max_depth=query.max_graph_depth,
                    max_nodes=query.max_graph_nodes,
                    allowed_relations=query.allowed_relationships,
                )
                related_ids = self.graph.traverse(seed_id, traversal)
                for related_id in related_ids:
                    if related_id == seed_id:
                        continue
                    if related_id not in candidates:
                        try:
                            record = self.store.get(related_id)
                            distance = self._compute_graph_distance(seed_id, related_id)
                            item = self._record_to_item(
                                record, source="graph_expansion", graph_distance=distance
                            )
                            item.reasons.append(
                                f"graph distance {distance} from seed {seed_id}"
                            )
                            candidates[related_id] = item
                            graph_expansion_count += 1
                        except Exception:
                            pass

        # 1c. Type-based candidates
        if query.record_types:
            for record_type in query.record_types:
                try:
                    records = self.store.find_by_type(record_type)
                    for record in records:
                        if record.record_id not in candidates:
                            item = self._record_to_item(record, source="type_match")
                            item.reasons.append(f"record type match: {record_type}")
                            candidates[record.record_id] = item
                except Exception:
                    pass

        # 1d. Tag-based candidates
        if query.tags:
            for tag in query.tags:
                try:
                    records = self.store.find_by_tag(tag)
                    for record in records:
                        if record.record_id not in candidates:
                            item = self._record_to_item(record, source="tag_match")
                            item.reasons.append(f"tag match: {tag}")
                            candidates[record.record_id] = item
                except Exception:
                    pass

        # Phase 2: Filter by status (unless include_historical)
        if not query.include_historical:
            filtered = {}
            for rid, item in candidates.items():
                if item.status in self.CURRENT_STATUSES:
                    filtered[rid] = item
                else:
                    # Mark as historical but keep if explicitly requested
                    if rid in query.record_ids:
                        item.is_historical = True
                        filtered[rid] = item
            candidates = filtered

        # Phase 3: Filter by authority requirements
        if query.authority_requirements:
            filtered = {}
            for rid, item in candidates.items():
                if item.authority in query.authority_requirements:
                    filtered[rid] = item
            candidates = filtered

        # Phase 4: Filter by min_authority
        if query.min_authority:
            filtered = {}
            for rid, item in candidates.items():
                if self._authority_meets_minimum(item.authority, query.min_authority):
                    filtered[rid] = item
            candidates = filtered

        # Phase 5: Mark mandatory items (SEC protected, governance, authoritative requirements)
        for rid, item in candidates.items():
            item.is_mandatory = self._is_mandatory(item)

        # Phase 6: Compute relevance scores
        task_context = self._infer_task_context(query)
        for rid, item in candidates.items():
            item.relevance = self._compute_relevance(item, query, task_context)

        # Phase 7: Deterministic ranking
        ranked = self._deterministic_rank(candidates, query)

        # Phase 8: Budget enforcement
        selected, dropped = self._enforce_budget(ranked, query.budget)

        # Phase 9: Check for mandatory overflow
        mandatory_items = [item for item in selected if item.is_mandatory]
        if len(mandatory_items) > query.budget:
            raise ContextOverflowError(len(mandatory_items), query.budget)

        # Phase 10: Contradiction awareness
        conflicts = self._detect_conflicts(selected)

        # Phase 11: Check for unresolved conflicts among selected
        unresolved = [c for c in conflicts if not c.get("resolved", False)]
        if unresolved:
            # Fail closed: do not silently return contradictory records as coherent truth
            raise KnowledgeConflictError(unresolved)

        # Phase 12: Compute metrics
        metrics = RetrievalMetrics(
            candidate_count=len(candidates),
            selected_count=len(selected),
            mandatory_count=len(mandatory_items),
            optional_count=len(selected) - len(mandatory_items),
            budget_used=sum(item.estimated_cost for item in selected),
            budget_remaining=query.budget - len(selected),
            records_dropped_by_budget=dropped,
            conflict_count=len(conflicts),
            graph_expansion_count=graph_expansion_count,
        )

        # Phase 13: Build result
        result = RetrievalResult(
            items=selected,
            metrics=metrics,
            conflicts=[c.to_dict() if hasattr(c, "to_dict") else c for c in conflicts],
            query=query,
        )
        result.digest = result.compute_digest()

        return result

    def get_related(self, record_id: str, direction: str = "both") -> List[RetrievalItem]:
        """
        Get records related to a given record via graph traversal.

        Args:
            record_id: Starting record ID
            direction: "outgoing", "incoming", or "both"

        Returns:
            List of RetrievalItem for related records
        """
        if record_id not in self.graph.nodes:
            return []

        related_ids = self.graph.neighbors(record_id, direction=direction)
        items = []
        for rid in related_ids:
            try:
                record = self.store.get(rid)
                item = self._record_to_item(record, source="graph_expansion", graph_distance=1)
                items.append(item)
            except Exception:
                pass

        # Deterministic ordering
        items.sort(key=lambda x: (x.graph_distance, x.record_id))
        return items

    def trace(self, decision_id: str) -> List[RetrievalItem]:
        """
        Trace a decision back to requirements and evidence.

        Builds a traceability chain: Decision → ADR → PRD → NFR → Evidence.

        Args:
            decision_id: The decision record ID to trace

        Returns:
            List of RetrievalItem forming the traceability chain
        """
        try:
            record = self.store.get(decision_id)
        except Exception:
            return []

        items = []
        item = self._record_to_item(record, source="explicit", graph_distance=0)
        item.reasons.append(f"Trace root: {decision_id}")
        items.append(item)

        # Trace to requirements
        for rel in record.get_relationships():
            if rel.target_id.startswith(("REQ-", "NFR-", "PRD-")):
                try:
                    req_record = self.store.get(rel.target_id)
                    req_item = self._record_to_item(
                        req_record, source="graph_expansion", graph_distance=1
                    )
                    req_item.reasons.append(
                        f"Traced from {decision_id} via {rel.relation_type}"
                    )
                    items.append(req_item)
                except Exception:
                    pass

        # Trace to evidence
        if record.provenance and record.provenance.evidence_refs:
            for ev_id in record.provenance.evidence_refs:
                # Evidence is referenced by ID, not embedded
                ev_item = RetrievalItem(
                    record_id=ev_id,
                    record_type="EVIDENCE",
                    authority="N/A",
                    status="N/A",
                    relevance=0.5,
                    reasons=[f"Evidence reference from {decision_id}"],
                    relationships=[],
                    source="evidence_reference",
                    graph_distance=2,
                )
                items.append(ev_item)

        # Deterministic ordering
        items.sort(key=lambda x: (x.graph_distance, x.record_id))
        return items

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _record_to_item(
        self,
        record: EngineeringRecord,
        source: str,
        graph_distance: int = -1,
    ) -> RetrievalItem:
        """Convert an EngineeringRecord to a RetrievalItem."""
        # Compute deterministic estimated cost
        record_size = len(record.to_json())
        summary_size = len(record.title) + len(record.description)
        rel_metadata = sum(len(str(r)) for r in record.get_relationships())
        estimated_cost = record_size + summary_size + rel_metadata

        # Build relationships list
        relationships = []
        for rel in record.get_relationships():
            relationships.append({
                "target_id": rel.target_id,
                "relation_type": rel.relation_type,
            })

        # Determine if historical
        is_historical = record.status not in self.CURRENT_STATUSES

        return RetrievalItem(
            record_id=record.record_id,
            record_type=record.record_type,
            authority=record.authority,
            status=record.status,
            relevance=0.0,  # computed later
            reasons=[],
            relationships=relationships,
            source=source,
            is_mandatory=False,  # computed later
            is_historical=is_historical,
            graph_distance=graph_distance,
            estimated_cost=estimated_cost,
        )

    def _compute_graph_distance(self, seed_id: str, target_id: str) -> int:
        """Compute graph distance from seed to target."""
        if seed_id == target_id:
            return 0
        # Direct neighbors
        if target_id in self.graph.neighbors(seed_id):
            return 1
        # Indirect (2-hop)
        for neighbor in self.graph.neighbors(seed_id):
            if target_id in self.graph.neighbors(neighbor):
                return 2
        return 3  # Default for non-adjacent

    def _infer_task_context(self, query: KnowledgeQuery) -> str:
        """Infer task context from query (NOT agent role)."""
        if query.task:
            return query.task.lower()
        if query.task_description:
            desc = query.task_description.lower()
            if "architect" in desc or "design" in desc:
                return "architecture"
            if "security" in desc or "auth" in desc:
                return "security"
            if "implement" in desc or "code" in desc:
                return "implementation"
            if "test" in desc or "verify" in desc:
                return "testing"
        return "default"

    def _compute_relevance(
        self,
        item: RetrievalItem,
        query: KnowledgeQuery,
        task_context: str,
    ) -> float:
        """
        Compute deterministic relevance score.

        Authority ≠ relevance. A highly authoritative record may be unrelated.
        A highly relevant record may have low authority.
        """
        scores = []

        # 1. Explicit reference (highest signal)
        if item.record_id in query.record_ids:
            scores.append(1.0)

        # 2. Graph distance (closer = more relevant)
        if item.graph_distance == 0:
            scores.append(0.9)
        elif item.graph_distance == 1:
            scores.append(0.7)
        elif item.graph_distance == 2:
            scores.append(0.5)

        # 3. Record type relevance (task-driven, NOT role-driven)
        type_relevance = self.TASK_TYPE_RELEVANCE.get(
            task_context, self.TASK_TYPE_RELEVANCE["default"]
        )
        scores.append(type_relevance.get(item.record_type, 0.3))

        # 4. Scope match (project, component, capability)
        scope_score = 0.0
        if query.project and item.record_id.startswith(query.project):
            scope_score += 0.3
        if query.components:
            for comp in query.components:
                if comp.lower() in item.record_id.lower():
                    scope_score += 0.2
        if query.capabilities:
            for cap in query.capabilities:
                if cap.lower() in item.record_id.lower():
                    scope_score += 0.1
        scores.append(min(scope_score, 0.5))

        # 5. Authority (accepted > proposed; SEC authority separate)
        if item.authority == AUTHORITY_ACCEPTED:
            scores.append(0.6)
        elif item.authority == AUTHORITY_PROPOSED:
            scores.append(0.3)
        elif item.authority in VALID_SEC_AUTHORITIES:
            sec_rank = SEC_AUTHORITY_RANK.get(item.authority, 0)
            scores.append(0.4 + sec_rank * 0.1)

        # 6. Status (current > historical)
        if item.status in self.CURRENT_STATUSES:
            scores.append(0.5)
        else:
            scores.append(0.1)

        # 7. Tag match
        if query.tags:
            # Check if record has matching tags (from store)
            try:
                record = self.store.get(item.record_id)
                matching_tags = set(record.tags) & set(query.tags)
                if matching_tags:
                    scores.append(0.4)
            except Exception:
                pass

        # Weighted combination (deterministic)
        weights = [0.25, 0.20, 0.15, 0.10, 0.15, 0.10, 0.05]
        if len(scores) < len(weights):
            weights = weights[:len(scores)]
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        return round(weighted_sum / total_weight, 4)

    def _deterministic_rank(
        self,
        candidates: Dict[str, RetrievalItem],
        query: KnowledgeQuery,
    ) -> List[RetrievalItem]:
        """
        Deterministic ranking of candidates.

        Ranking signals (in priority order):
        1. Explicit reference (seed record_ids)
        2. Graph distance (0=seed, 1=direct, 2=indirect)
        3. Record type relevance (task-driven)
        4. Scope match
        5. Authority (accepted > proposed; SEC separate)
        6. Status (current > historical)
        7. Structured tag/component match

        No Python hash. No set order. No SQLite row order.
        No filesystem order. No global RNG. No LLM preference.
        """
        items = list(candidates.values())

        def sort_key(item: RetrievalItem) -> tuple:
            # Explicit reference first
            is_explicit = item.record_id in query.record_ids
            # Graph distance (lower = better)
            distance = item.graph_distance if item.graph_distance >= 0 else 999
            # Authority score (higher = better)
            auth_score = self._authority_score(item.authority)
            # Status score (current = 1, historical = 0)
            status_score = 1 if item.status in self.CURRENT_STATUSES else 0
            # Relevance (higher = better)
            relevance = item.relevance
            # Mandatory first
            is_mandatory = 1 if item.is_mandatory else 0

            return (
                -int(is_explicit),  # explicit first
                -is_mandatory,  # mandatory first
                distance,  # closer first
                -relevance,  # higher relevance first
                -auth_score,  # higher authority first
                -status_score,  # current first
                item.record_id,  # deterministic tiebreaker
            )

        items.sort(key=sort_key)
        return items

    def _authority_score(self, authority: str) -> int:
        """Get deterministic authority score."""
        if authority == AUTHORITY_ACCEPTED:
            return 4
        if authority == AUTHORITY_PROPOSED:
            return 2
        if authority in VALID_SEC_AUTHORITIES:
            return SEC_AUTHORITY_RANK.get(authority, 0) + 3
        return 0

    def _authority_meets_minimum(self, authority: str, minimum: str) -> bool:
        """Check if authority meets minimum requirement."""
        if minimum == AUTHORITY_ACCEPTED:
            return authority == AUTHORITY_ACCEPTED
        if minimum == AUTHORITY_PROPOSED:
            return authority in (AUTHORITY_PROPOSED, AUTHORITY_ACCEPTED)
        return True

    def _is_mandatory(self, item: RetrievalItem) -> bool:
        """
        Check if an item is mandatory (protected knowledge).

        Mandatory items:
        - SEC records with ACCEPTED or AUTHORITATIVE authority
        - Governance constraints
        - Authoritative requirements (accepted ADR/DR that constrain the task)

        These cannot be dropped by ordinary ranking/budget competition.
        """
        # SEC with accepted or higher authority
        if item.record_type == "SEC":
            if item.authority in VALID_SEC_AUTHORITIES:
                sec_rank = SEC_AUTHORITY_RANK.get(item.authority, 0)
                if sec_rank >= SEC_AUTHORITY_RANK.get("ACCEPTED", 4):
                    return True

        # Accepted ADR/DR (authoritative engineering knowledge)
        if item.record_type in ("ADR", "DR") and item.authority == AUTHORITY_ACCEPTED:
            return True

        # Accepted requirements that constrain the task
        if item.record_type in ("REQ", "NFR") and item.authority == AUTHORITY_ACCEPTED:
            return True

        return False

    def _enforce_budget(
        self,
        ranked: List[RetrievalItem],
        budget: int,
    ) -> Tuple[List[RetrievalItem], int]:
        """
        Enforce budget constraint.

        Mandatory items are always included (protected knowledge).
        Optional items are included until budget is exhausted.

        Returns:
            (selected_items, dropped_count)
        """
        selected = []
        mandatory = []
        optional = []

        for item in ranked:
            if item.is_mandatory:
                mandatory.append(item)
            else:
                optional.append(item)

        # Mandatory items always included
        selected.extend(mandatory)

        # Optional items fill remaining budget
        remaining = budget - len(mandatory)
        if remaining < 0:
            # Mandatory alone exceeds budget — will raise ContextOverflowError
            return mandatory, len(optional)

        for item in optional:
            if len(selected) < budget:
                selected.append(item)
            else:
                break

        dropped = len(ranked) - len(selected)
        return selected, dropped

    def _detect_conflicts(self, items: List[RetrievalItem]) -> List[Dict[str, Any]]:
        """
        Detect contradictions among selected items.

        Uses ContradictionDetector for structural contradictions.
        Does not silently return contradictory records as coherent truth.
        """
        conflicts = []

        # Check for CONTRADICTS relationships among selected
        selected_ids = {item.record_id for item in items}
        for item in items:
            for rel in item.relationships:
                if rel["relation_type"] == RELATIONSHIP_CONTRADICTS:
                    if rel["target_id"] in selected_ids:
                        conflicts.append({
                            "type": "EXPLICIT_CONTRADICTION",
                            "records": sorted([item.record_id, rel["target_id"]]),
                            "reason": f"{item.record_id} CONTRADICTS {rel['target_id']}",
                            "resolved": False,
                            "resolution": "",
                        })

        # Check for superseded records presented as current
        for item in items:
            if item.is_historical and item.record_id not in selected_ids:
                conflicts.append({
                    "type": "SUPERSEDED_AS_CURRENT",
                    "records": [item.record_id],
                    "reason": f"{item.record_id} is superseded/deprecated but presented as current",
                    "resolved": False,
                    "resolution": "",
                })

        # Use ContradictionDetector for graph-level conflicts
        try:
            graph_conflicts = self._contradiction_detector.detect_all()
            for gc in graph_conflicts:
                conflicts.append({
                    "type": gc.conflict_type,
                    "records": gc.records,
                    "reason": gc.reason,
                    "resolved": False,
                    "resolution": "",
                })
        except Exception:
            pass

        # Sort for deterministic output
        conflicts.sort(key=lambda c: (c["type"], c["records"]))
        return conflicts
