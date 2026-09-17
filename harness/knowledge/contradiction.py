# harness/knowledge/contradiction.py — V3.3 Structural Contradiction Detection
"""
Detect structural contradictions in the EngineeringKnowledgeGraph.

Phase 3 Implementation — TASK-045 (Contradiction Detection)

This module detects deterministic structural contradictions only.
It does NOT attempt broad LLM semantic contradiction detection.

Contradiction types detected:
- DANGLING_REFERENCE: relationship references nonexistent record
- INVALID_RELATION: relation type not recognized
- FORBIDDEN_CYCLE: cycle in SUPERSEDES or CAUSED_BY
- SUPERSESSION_CONFLICT: mutual supersession, branching ambiguity
- EXPLICIT_CONTRADICTION: CONTRADICTS edge exists
- STATUS_CONFLICT: superseded record still marked active/accepted

Domain separation:
- ContradictionDetector belongs to Engineering Knowledge
- Does NOT touch Learning or Evidence domains
- Integrates with EngineeringKnowledgeGraph for structural analysis
"""

from typing import Any, Dict, List, Optional, Set

from .records import (
    EngineeringRecord,
    RELATIONSHIP_SUPERSEDES,
    RELATIONSHIP_CONTRADICTS,
    RELATIONSHIP_CAUSED_BY,
    RELATIONSHIP_RESOLVED_BY,
)
from .graph import (
    EngineeringKnowledgeGraph,
    GraphEdge,
    IntegrityConflict,
    GraphError,
)


class ContradictionError(Exception):
    """Raised on contradiction detection errors."""


# ──────────────────────────────────────────────────────────────────────
# Structured conflict representation
# ──────────────────────────────────────────────────────────────────────

CONFLICT_TYPE_DANGLING_REFERENCE = "DANGLING_REFERENCE"
CONFLICT_TYPE_INVALID_RELATION = "INVALID_RELATION"
CONFLICT_TYPE_FORBIDDEN_CYCLE = "FORBIDDEN_CYCLE"
CONFLICT_TYPE_SUPERSESSION_CONFLICT = "SUPERSESSION_CONFLICT"
CONFLICT_TYPE_EXPLICIT_CONTRADICTION = "EXPLICIT_CONTRADICTION"
CONFLICT_TYPE_STATUS_CONFLICT = "STATUS_CONFLICT"

VALID_CONFLICT_TYPES = {
    CONFLICT_TYPE_DANGLING_REFERENCE,
    CONFLICT_TYPE_INVALID_RELATION,
    CONFLICT_TYPE_FORBIDDEN_CYCLE,
    CONFLICT_TYPE_SUPERSESSION_CONFLICT,
    CONFLICT_TYPE_EXPLICIT_CONTRADICTION,
    CONFLICT_TYPE_STATUS_CONFLICT,
}


# ──────────────────────────────────────────────────────────────────────
# ContradictionDetector
# ──────────────────────────────────────────────────────────────────────

class ContradictionDetector:
    """
    Detects structural contradictions in an EngineeringKnowledgeGraph.

    Deterministic: same graph → same contradictions, always.
    No LLM inference. No semantic analysis. Structural only.

    Usage:
        graph = EngineeringKnowledgeGraph.build_from_store(store)
        detector = ContradictionDetector(graph)
        conflicts = detector.detect_all()
        for c in conflicts:
            print(c.conflict_type, c.records, c.reason)

    Fail-closed: A graph with unresolved integrity violations is NOT
    presented as clean knowledge.
    """

    def __init__(self, graph: EngineeringKnowledgeGraph):
        self.graph = graph

    def detect_all(self) -> List[IntegrityConflict]:
        """
        Detect all structural contradictions in the graph.

        Returns a sorted, deduplicated list of IntegrityConflict objects.
        Empty list means the graph is structurally clean.
        """
        conflicts: List[IntegrityConflict] = []

        # Dangling references (edges pointing to nonexistent nodes)
        conflicts.extend(self._detect_dangling_references())

        # Invalid relation types (should not happen with GraphEdge validation)
        conflicts.extend(self._detect_invalid_relations())

        # Forbidden cycles
        conflicts.extend(self._detect_forbidden_cycles())

        # Supersession conflicts
        conflicts.extend(self._detect_supersession_conflicts())

        # Explicit contradictions (CONTRADICTS edges)
        conflicts.extend(self._detect_explicit_contradictions())

        # Status conflicts (superseded record still active)
        conflicts.extend(self._detect_status_conflicts())

        # Deduplicate by (type, records) — keep first occurrence
        seen: Set[tuple] = set()
        unique: List[IntegrityConflict] = []
        for c in conflicts:
            key = (c.conflict_type, tuple(c.records))
            if key not in seen:
                seen.add(key)
                unique.append(c)

        # Sort for deterministic output
        unique.sort(key=lambda c: (c.conflict_type, c.records))
        return unique

    def is_clean(self) -> bool:
        """Return True if the graph has no structural contradictions."""
        return len(self.detect_all()) == 0

    def _detect_dangling_references(self) -> List[IntegrityConflict]:
        """Detect edges referencing nonexistent canonical records."""
        conflicts: List[IntegrityConflict] = []
        known_nodes = self.graph.nodes

        for edge in self.graph.get_edges():
            if edge.target_id not in known_nodes:
                conflicts.append(
                    IntegrityConflict(
                        conflict_type=CONFLICT_TYPE_DANGLING_REFERENCE,
                        records=[edge.source_id, edge.target_id],
                        relations=[{
                            "source_id": edge.source_id,
                            "relation_type": edge.relation_type,
                            "target_id": edge.target_id,
                        }],
                        severity="HIGH",
                        reason=(
                            f"Dangling reference: {edge.source_id} "
                            f"—{edge.relation_type}→ {edge.target_id} "
                            f"but {edge.target_id} does not exist in graph"
                        ),
                    )
                )
            if edge.source_id not in known_nodes:
                conflicts.append(
                    IntegrityConflict(
                        conflict_type=CONFLICT_TYPE_DANGLING_REFERENCE,
                        records=[edge.source_id, edge.target_id],
                        relations=[{
                            "source_id": edge.source_id,
                            "relation_type": edge.relation_type,
                            "target_id": edge.target_id,
                        }],
                        severity="HIGH",
                        reason=(
                            f"Dangling reference: {edge.source_id} "
                            f"—{edge.relation_type}→ {edge.target_id} "
                            f"but {edge.source_id} does not exist in graph"
                        ),
                    )
                )

        return conflicts

    def _detect_invalid_relations(self) -> List[IntegrityConflict]:
        """Detect edges with invalid relation types."""
        conflicts: List[IntegrityConflict] = []
        # GraphEdge constructor already validates, but double-check
        for edge in self.graph.get_edges():
            if edge.relation_type not in {
                "REQUIRES", "SATISFIES", "DECIDED_BY", "CONSTRAINED_BY",
                "INTRODUCES", "MITIGATES", "SUPERSEDES", "CAUSED_BY",
                "RESOLVED_BY", "VERIFIED_BY", "SUPPORTED_BY", "CONTRADICTS",
                "RELATES_TO", "DERIVED_FROM",
            }:
                conflicts.append(
                    IntegrityConflict(
                        conflict_type=CONFLICT_TYPE_INVALID_RELATION,
                        records=[edge.source_id, edge.target_id],
                        relations=[{
                            "source_id": edge.source_id,
                            "relation_type": edge.relation_type,
                            "target_id": edge.target_id,
                        }],
                        severity="HIGH",
                        reason=f"Invalid relation type: {edge.relation_type}",
                    )
                )
        return conflicts

    def _detect_forbidden_cycles(self) -> List[IntegrityConflict]:
        """Detect cycles in SUPERSEDES, CAUSED_BY, or RESOLVED_BY."""
        conflicts: List[IntegrityConflict] = []
        forbidden_types = {
            RELATIONSEDES := RELATIONSHIP_SUPERSEDES,
            RELATIONSHIP_CAUSED_BY,
            RELATIONSHIP_RESOLVED_BY,
        }

        for rtype in sorted(forbidden_types):
            has_cycle, cycle_path = self.graph.has_cycle(relation_type=rtype)
            if has_cycle:
                relations = []
                for i in range(len(cycle_path) - 1):
                    relations.append({
                        "source_id": cycle_path[i],
                        "relation_type": rtype,
                        "target_id": cycle_path[i + 1],
                    })
                # Close the cycle
                relations.append({
                    "source_id": cycle_path[-1],
                    "relation_type": rtype,
                    "target_id": cycle_path[0],
                })
                conflicts.append(
                    IntegrityConflict(
                        conflict_type=CONFLICT_TYPE_FORBIDDEN_CYCLE,
                        records=cycle_path,
                        relations=relations,
                        severity="HIGH",
                        reason=(
                            f"Forbidden cycle in {rtype}: "
                            f"{' → '.join(cycle_path)} → {cycle_path[0]}"
                        ),
                    )
                )

        return conflicts

    def _detect_supersession_conflicts(self) -> List[IntegrityConflict]:
        """Detect mutual supersession and branching ambiguity."""
        conflicts: List[IntegrityConflict] = []

        # Mutual supersession: A supersedes B AND B supersedes A
        for edge in self.graph.get_edges(relation_type=RELATIONSHIP_SUPERSEDES):
            reverse = self.graph.has_edge(
                edge.target_id, edge.source_id, RELATIONSHIP_SUPERSEDES
            )
            if reverse:
                records = sorted([edge.source_id, edge.target_id])
                conflicts.append(
                    IntegrityConflict(
                        conflict_type=CONFLICT_TYPE_SUPERSESSION_CONFLICT,
                        records=records,
                        relations=[
                            {
                                "source_id": edge.source_id,
                                "relation_type": "SUPERSEDES",
                                "target_id": edge.target_id,
                            },
                            {
                                "source_id": edge.target_id,
                                "relation_type": "SUPERSEDES",
                                "target_id": edge.source_id,
                            },
                        ],
                        severity="HIGH",
                        reason=(
                            f"Mutual supersession: {edge.source_id} and "
                            f"{edge.target_id} each supersede the other"
                        ),
                    )
                )

        # Branching ambiguity: one record superseded by multiple
        outgoing: Dict[str, List[str]] = {}
        for edge in self.graph.get_edges(relation_type=RELATIONSHIP_SUPERSEDES):
            outgoing.setdefault(edge.source_id, []).append(edge.target_id)

        for source_id, targets in sorted(outgoing.items()):
            if len(targets) > 1:
                conflicts.append(
                    IntegrityConflict(
                        conflict_type=CONFLICT_TYPE_SUPERSESSION_CONFLICT,
                        records=[source_id] + sorted(targets),
                        relations=[
                            {
                                "source_id": source_id,
                                "relation_type": "SUPERSEDES",
                                "target_id": t,
                            }
                            for t in sorted(targets)
                        ],
                        severity="MEDIUM",
                        reason=(
                            f"Ambiguous supersession: {source_id} supersedes "
                            f"multiple records: {sorted(targets)}"
                        ),
                    )
                )

        return conflicts

    def _detect_explicit_contradictions(self) -> List[IntegrityConflict]:
        """Detect explicit CONTRADICTS edges (structural, not semantic)."""
        conflicts: List[IntegrityConflict] = []

        for edge in self.graph.get_edges(relation_type=RELATIONSHIP_CONTRADICTS):
            conflicts.append(
                IntegrityConflict(
                    conflict_type=CONFLICT_TYPE_EXPLICIT_CONTRADICTION,
                    records=sorted([edge.source_id, edge.target_id]),
                    relations=[{
                        "source_id": edge.source_id,
                        "relation_type": "CONTRADICTS",
                        "target_id": edge.target_id,
                    }],
                    severity="MEDIUM",
                    reason=(
                        f"Explicit contradiction: {edge.source_id} "
                        f"CONTRADICTS {edge.target_id}"
                    ),
                )
            )

        return conflicts

    def _detect_status_conflicts(self) -> List[IntegrityConflict]:
        """
        Detect records that are superseded but still marked active/accepted.

        This requires record metadata (status, authority) which is not stored
        in the graph itself. When records are available via the store, this
        check is performed by validate_integrity(store).
        """
        # Graph-level status conflicts are detected during validate_integrity()
        # with store access. This method is a placeholder for graph-only checks.
        return []

    def detect_with_records(
        self, records: List[EngineeringRecord]
    ) -> List[IntegrityConflict]:
        """
        DetectContradictions enriched with record metadata.

        Adds STATUS_CONFLICT detection: superseded records that are still
        active/accepted in the canonical store.
        """
        conflicts = self.detect_all()

        # Build record lookup
        record_map: Dict[str, EngineeringRecord] = {
            r.record_id: r for r in records
        }

        # Check for superseded-but-active records
        for edge in self.graph.get_edges(relation_type=RELATIONSHIP_SUPERSEDES):
            source = record_map.get(edge.source_id)
            if source is not None:
                # Superseded record should not be 'active' or 'accepted'
                active_statuses = {"active", "accepted", "proposed", "identified", "draft"}
                if source.status in active_statuses:
                    conflicts.append(
                        IntegrityConflict(
                            conflict_type=CONFLICT_TYPE_STATUS_CONFLICT,
                            records=[edge.source_id],
                            relations=[{
                                "source_id": edge.source_id,
                                "relation_type": "SUPERSEDES",
                                "target_id": edge.target_id,
                            }],
                            severity="MEDIUM",
                            reason=(
                                f"Record {edge.source_id} is superseded by "
                                f"{edge.target_id} but still has active status "
                                f"'{source.status}'"
                            ),
                        )
                    )

        # Re-deduplicate and sort
        seen: Set[tuple] = set()
        unique: List[IntegrityConflict] = []
        for c in conflicts:
            key = (c.conflict_type, tuple(c.records))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        unique.sort(key=lambda c: (c.conflict_type, c.records))
        return unique
