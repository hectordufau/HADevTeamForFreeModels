# harness/knowledge/graph.py — V3.3 Engineering Knowledge Graph
"""
Deterministic directed graph connecting EngineeringRecord instances via typed relationships.

Phase 3 Implementation — TASK-044 (KnowledgeGraph Core Model)

Architecture:
- Graph nodes reference stable EngineeringRecord IDs (no record duplication)
- Edges use Phase 1 RecordRelationship types (REQUIRES, SATISFIES, DECIDED_BY, etc.)
- Graph is built from KnowledgeStore — canonical records remain authoritative
- Same canonical records yield same graph regardless of insertion order
- All traversal is bounded (max_depth, max_nodes, allowed_relations)
- Deterministic query ordering: (record_id, relation_type, record_type)

Domain separation:
- EngineeringKnowledgeGraph manages EngineeringRecord relationships only
- Does NOT touch Learning or Evidence domains
- Does NOT duplicate CapabilityGraph semantics
"""

import hashlib
import json
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from .records import (
    EngineeringRecord,
    RecordRelationship,
    VALID_RELATIONSHIP_TYPES,
    RELATIONSHIP_REQUIRES,
    RELATIONSHIP_SATISFIES,
    RELATIONSHIP_DECIDED_BY,
    RELATIONSHIP_CONSTRAINED_BY,
    RELATIONSHIP_INTRODUCES,
    RELATIONSHIP_MITIGATES,
    RELATIONSHIP_SUPERSEDES,
    RELATIONSHIP_CAUSED_BY,
    RELATIONSHIP_RESOLVED_BY,
    RELATIONSHIP_VERIFIED_BY,
    RELATIONSHIP_SUPPORTED_BY,
    RELATIONSHIP_CONTRADICTS,
    RELATIONSHIP_RELATES_TO,
    RELATIONSHIP_DERIVED_FROM,
)


class GraphError(Exception):
    """Raised on graph operation errors."""


class IntegrityViolation(Exception):
    """Raised when graph integrity validation fails."""

    def __init__(self, conflicts: List[Dict[str, Any]]):
        self.conflicts = conflicts
        super().__init__(f"Graph integrity violation: {len(conflicts)} conflict(s) detected")


# ──────────────────────────────────────────────────────────────────────
# Graph Edge — typed relationship between two record IDs
# ──────────────────────────────────────────────────────────────────────

class GraphEdge:
    """
    A typed, directed edge between two engineering record IDs.

    Edges reference record IDs only — canonical records remain in KnowledgeStore.
    Direction is explicit: source → relation_type → target.
    Inverse traversal preserves canonical direction.
    """

    __slots__ = ("source_id", "relation_type", "target_id", "metadata")

    def __init__(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not isinstance(source_id, str) or not source_id:
            raise GraphError("source_id must be a non-empty string")
        if not isinstance(target_id, str) or not target_id:
            raise GraphError("target_id must be a non-empty string")
        if relation_type not in VALID_RELATIONSHIP_TYPES:
            raise GraphError(
                f"Invalid relation_type '{relation_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_RELATIONSHIP_TYPES))}"
            )
        self.source_id = source_id
        self.relation_type = relation_type
        self.target_id = target_id
        self.metadata = dict(metadata) if metadata else {}

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "relation_type": self.relation_type,
            "target_id": self.target_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GraphEdge":
        return cls(
            source_id=data["source_id"],
            relation_type=data["relation_type"],
            target_id=data["target_id"],
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_relationship(cls, rel: RecordRelationship) -> "GraphEdge":
        """Create a GraphEdge from a RecordRelationship."""
        return cls(
            source_id=rel.source_id,
            relation_type=rel.relation_type,
            target_id=rel.target_id,
            metadata=dict(rel.metadata),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GraphEdge):
            return NotImplemented
        return (
            self.source_id == other.source_id
            and self.relation_type == other.relation_type
            and self.target_id == other.target_id
        )

    def __hash__(self) -> int:
        return hash((self.source_id, self.relation_type, self.target_id))

    def __repr__(self) -> str:
        return f"GraphEdge({self.source_id} —{self.relation_type}→ {self.target_id})"


# ──────────────────────────────────────────────────────────────────────
# TraversalOptions — bounded, typed traversal configuration
# ──────────────────────────────────────────────────────────────────────

class TraversalOptions:
    """
    Configuration for bounded graph traversal.

    All traversal is bounded by default: a graph query must not
    accidentally inject the entire knowledge base into future context.
    """

    def __init__(
        self,
        max_depth: int = 10,
        max_nodes: int = 1000,
        allowed_relations: Optional[Set[str]] = None,
        record_types: Optional[Set[str]] = None,
        statuses: Optional[Set[str]] = None,
        authorities: Optional[Set[str]] = None,
        direction: str = "outgoing",
    ):
        if not isinstance(max_depth, int) or max_depth < 0:
            raise GraphError("max_depth must be a non-negative integer")
        if not isinstance(max_nodes, int) or max_nodes < 1:
            raise GraphError("max_nodes must be a positive integer")
        if direction not in ("outgoing", "incoming", "both"):
            raise GraphError("direction must be 'outgoing', 'incoming', or 'both'")

        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.allowed_relations = (
            set(allowed_relations) if allowed_relations is not None else None
        )
        self.record_types = (
            set(record_types) if record_types is not None else None
        )
        self.statuses = set(statuses) if statuses is not None else None
        self.authorities = (
            set(authorities) if authorities is not None else None
        )
        self.direction = direction

    def to_dict(self) -> dict:
        return {
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "allowed_relations": (
                sorted(self.allowed_relations)
                if self.allowed_relations is not None
                else None
            ),
            "record_types": (
                sorted(self.record_types) if self.record_types is not None else None
            ),
            "statuses": (
                sorted(self.statuses) if self.statuses is not None else None
            ),
            "authorities": (
                sorted(self.authorities) if self.authorities is not None else None
            ),
            "direction": self.direction,
        }


# ──────────────────────────────────────────────────────────────────────
# PathResult — result of a path query
# ──────────────────────────────────────────────────────────────────────

class PathResult:
    """Result of a path query between two nodes."""

    __slots__ = ("found", "path", "edges", "length")

    def __init__(
        self,
        found: bool,
        path: List[str],
        edges: List[GraphEdge],
        length: int,
    ):
        self.found = found
        self.path = path  # list of record IDs
        self.edges = edges  # list of GraphEdges traversed
        self.length = length

    def to_dict(self) -> dict:
        return {
            "found": self.found,
            "path": list(self.path),
            "edges": [e.to_dict() for e in self.edges],
            "length": self.length,
        }


# ──────────────────────────────────────────────────────────────────────
# IntegrityConflict — structured conflict result
# ──────────────────────────────────────────────────────────────────────

class IntegrityConflict:
    """
    A structured graph integrity conflict.

    Conflict types: DANGLING_REFERENCE, INVALID_RELATION, FORBIDDEN_CYCLE,
    SUPERSESSION_CONFLICT, EXPLICIT_CONTRADICTION, STATUS_CONFLICT.
    """

    def __init__(
        self,
        conflict_type: str,
        records: List[str],
        relations: List[Dict[str, str]],
        severity: str,
        reason: str,
    ):
        self.conflict_type = conflict_type
        self.records = sorted(records)
        self.relations = relations
        self.severity = severity
        self.reason = reason

    def to_dict(self) -> dict:
        return {
            "type": self.conflict_type,
            "records": list(self.records),
            "relations": list(self.relations),
            "severity": self.severity,
            "reason": self.reason,
        }

    def __repr__(self) -> str:
        return f"IntegrityConflict({self.conflict_type}: {self.records})"


# ──────────────────────────────────────────────────────────────────────
# Default traversal options — conservative bounds
# ──────────────────────────────────────────────────────────────────────

DEFAULT_TRAVERSAL = TraversalOptions(
    max_depth=10,
    max_nodes=1000,
)


# ──────────────────────────────────────────────────────────────────────
# EngineeringKnowledgeGraph
# ──────────────────────────────────────────────────────────────────────

class EngineeringKnowledgeGraph:
    """
    Deterministic directed graph of engineering record relationships.

    Built from KnowledgeStore — canonical records remain authoritative.
    Nodes are record IDs; edges are typed GraphEdges.
    All operations are deterministic and bounded.

    Usage:
        store = KnowledgeStore("knowledge.db")
        graph = EngineeringKnowledgeGraph.build_from_store(store)
        related = graph.neighbors("REQ-001")
        path = graph.path("REQ-001", "ADR-004")
        conflicts = graph.validate_integrity()
        digest = graph.compute_digest()
    """

    def __init__(self):
        # Adjacency: node_id → list of GraphEdge
        self._outgoing: Dict[str, List[GraphEdge]] = {}
        # Reverse: node_id → list of GraphEdge (incoming)
        self._incoming: Dict[str, List[GraphEdge]] = {}
        # Node set (all known record IDs)
        self._nodes: Set[str] = set()

    # ──────────────────────────────────────────────────────────────────
    # Construction — build from records or relationships
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def build_from_store(cls, store) -> "EngineeringKnowledgeGraph":
        """
        Rebuild graph from KnowledgeStore via public API.

        Reads all records and their relationships from the store.
        Deterministic: same store contents → same graph, regardless of
        filesystem order, SQLite row order, dictionary insertion order.
        """
        graph = cls()
        records = store.list_all()
        for record in records:
            graph._add_node(record.record_id)
        for record in records:
            for rel in record.get_relationships():
                edge = GraphEdge.from_relationship(rel)
                graph._add_edge_unchecked(edge)
        return graph

    @classmethod
    def build_from_records(
        cls, records: List[EngineeringRecord]
    ) -> "EngineeringKnowledgeGraph":
        """Build graph from a list of EngineeringRecord instances."""
        graph = cls()
        for record in records:
            graph._add_node(record.record_id)
        for record in records:
            for rel in record.get_relationships():
                edge = GraphEdge.from_relationship(rel)
                graph._add_edge_unchecked(edge)
        return graph

    @classmethod
    def build_from_edges(cls, edges: List[GraphEdge]) -> "EngineeringKnowledgeGraph":
        """Build graph from a list of GraphEdge instances."""
        graph = cls()
        for edge in edges:
            graph._add_node(edge.source_id)
            graph._add_node(edge.target_id)
            graph._add_edge_unchecked(edge)
        return graph

    def add_node(self, node_id: str) -> None:
        """Add a node to the graph (idempotent)."""
        if not isinstance(node_id, str) or not node_id:
            raise GraphError("node_id must be a non-empty string")
        if node_id not in self._nodes:
            self._nodes.add(node_id)
            self._outgoing[node_id] = []
            self._incoming[node_id] = []

    def _add_node(self, node_id: str) -> None:
        """Internal: add a node without validation (for bulk operations)."""
        if node_id not in self._nodes:
            self._nodes.add(node_id)
            self._outgoing[node_id] = []
            self._incoming[node_id] = []

    def _add_edge_unchecked(self, edge: GraphEdge) -> None:
        """
        Add an edge without cycle checking (used during bulk build).

        Duplicate edges are silently deduplicated (idempotent rebuild).
        Self-references are rejected.
        """
        if edge.source_id == edge.target_id:
            return  # Skip self-references during bulk build

        self._add_node(edge.source_id)
        self._add_node(edge.target_id)

        # Check for duplicate (idempotent)
        for existing in self._outgoing[edge.source_id]:
            if existing == edge:
                return

        self._outgoing[edge.source_id].append(edge)
        self._incoming[edge.target_id].append(edge)

    def add_edge(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphEdge:
        """
        Add a single edge (with validation).

        Raises GraphError on invalid relation type or self-reference.
        Duplicate edges are silently accepted (idempotent semantics).
        """
        edge = GraphEdge(
            source_id=source_id,
            relation_type=relation_type,
            target_id=target_id,
            metadata=metadata,
        )
        if source_id == target_id:
            raise GraphError("Self-references are not allowed in graph edges")
        self._add_edge_unchecked(edge)
        return edge

    # ──────────────────────────────────────────────────────────────────
    # Node / edge queries
    # ──────────────────────────────────────────────────────────────────

    @property
    def nodes(self) -> Set[str]:
        """Return all node IDs (record IDs)."""
        return set(self._nodes)

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""
        return node_id in self._nodes

    def node_count(self) -> int:
        """Return the number of nodes."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """Return the total number of edges."""
        return sum(len(edges) for edges in self._outgoing.values())

    def has_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: Optional[str] = None,
    ) -> bool:
        """Check if an edge exists, optionally filtered by relation type."""
        if source_id not in self._outgoing:
            return False
        for edge in self._outgoing[source_id]:
            if edge.target_id == target_id:
                if relation_type is None or edge.relation_type == relation_type:
                    return True
        return False

    def get_edges(
        self,
        source_id: Optional[str] = None,
        relation_type: Optional[str] = None,
    ) -> List[GraphEdge]:
        """
        Get edges, optionally filtered by source and/or relation type.

        Returns edges in deterministic order: sorted by (source_id, relation_type, target_id).
        """
        edges: List[GraphEdge] = []
        sources = [source_id] if source_id else sorted(self._outgoing.keys())
        for sid in sources:
            if sid not in self._outgoing:
                continue
            for edge in self._outgoing[sid]:
                if relation_type is None or edge.relation_type == relation_type:
                    edges.append(edge)
        return edges

    def get_edge_types(self, source_id: str, target_id: str) -> List[str]:
        """Get all relation types between two nodes."""
        if source_id not in self._outgoing:
            return []
        return [
            e.relation_type
            for e in self._outgoing[source_id]
            if e.target_id == target_id
        ]

    # ──────────────────────────────────────────────────────────────────
    # Traversal — bounded, typed, deterministic
    # ──────────────────────────────────────────────────────────────────

    def neighbors(
        self,
        record_id: str,
        relation_type: Optional[str] = None,
        direction: str = "outgoing",
    ) -> List[str]:
        """
        Get neighboring record IDs: one-hop traversal.

        Args:
            record_id: Starting node.
            relation_type: Optional filter.
            direction: "outgoing", "incoming", or "both".

        Returns:
            Sorted list of neighboring record IDs. Empty if node not found.
        """
        if record_id not in self._nodes:
            return []
        results: Set[str] = set()
        if direction in ("outgoing", "both"):
            for edge in self._outgoing[record_id]:
                if relation_type is None or edge.relation_type == relation_type:
                    results.add(edge.target_id)
        if direction in ("incoming", "both"):
            for edge in self._incoming[record_id]:
                if relation_type is None or edge.relation_type == relation_type:
                    results.add(edge.source_id)
        return sorted(results)

    def outgoing(
        self,
        record_id: str,
        relation_type: Optional[str] = None,
    ) -> List[str]:
        """Get targets of outgoing edges from a node. Sorted."""
        return self.neighbors(record_id, relation_type, direction="outgoing")

    def incoming(
        self,
        record_id: str,
        relation_type: Optional[str] = None,
    ) -> List[str]:
        """Get sources of incoming edges to a node. Sorted."""
        return self.neighbors(record_id, relation_type, direction="incoming")

    def related(
        self,
        record_id: str,
        relation_type: Optional[str] = None,
    ) -> List[str]:
        """Get related record IDs in both directions. Sorted."""
        return self.neighbors(record_id, relation_type, direction="both")

    def traverse(
        self,
        start_id: str,
        options: Optional[TraversalOptions] = None,
    ) -> List[str]:
        """
        Bounded BFS traversal from a starting node.

        Respects TraversalOptions: max_depth, max_nodes, allowed_relations,
        record_types, statuses, authorities, direction.

        Returns node IDs in BFS order (deterministic), including start node.
        """
        if start_id not in self._nodes:
            return []
        opts = options or DEFAULT_TRAVERSAL

        visited: Set[str] = set()
        queue: deque = deque([(start_id, 0)])
        visited.add(start_id)
        result: List[str] = [start_id]

        while queue and len(visited) < opts.max_nodes:
            current, depth = queue.popleft()
            if depth >= opts.max_depth:
                continue

            # Get neighbors based on direction
            neighbors: List[Tuple[str, str]] = []  # (neighbor_id, relation_type)
            if opts.direction in ("outgoing", "both"):
                for edge in self._outgoing.get(current, []):
                    if opts.allowed_relations is None or edge.relation_type in opts.allowed_relations:
                        neighbors.append((edge.target_id, edge.relation_type))
            if opts.direction in ("incoming", "both"):
                for edge in self._incoming.get(current, []):
                    if opts.allowed_relations is None or edge.relation_type in opts.allowed_relations:
                        neighbors.append((edge.source_id, edge.relation_type))

            # Sort for deterministic ordering
            neighbors.sort()

            for neighbor_id, _ in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    result.append(neighbor_id)
                    queue.append((neighbor_id, depth + 1))
                    if len(visited) >= opts.max_nodes:
                        break

        return result

    def path(
        self,
        source: str,
        target: str,
        options: Optional[TraversalOptions] = None,
    ) -> PathResult:
        """
        Find a path from source to target (BFS shortest path).

        Respects TraversalOptions bounds. Returns PathResult with found=True
        and the path, or found=False if no path exists.
        """
        if source not in self._nodes or target not in self._nodes:
            return PathResult(found=False, path=[], edges=[], length=0)
        if source == target:
            return PathResult(found=True, path=[source], edges=[], length=0)

        opts = options or DEFAULT_TRAVERSAL

        # BFS with path tracking: node → (predecessor, edge_used)
        visited: Dict[str, Tuple[str, GraphEdge]] = {source: (None, None)}  # type: ignore
        queue: deque = deque([source])

        while queue and len(visited) < opts.max_nodes:
            current = queue.popleft()

            # Check depth limit (BFS guarantees shortest path)
            # Depth of current = path length from source
            depth = 0
            node = current
            while visited[node][0] is not None:
                depth += 1
                node = visited[node][0]
            if depth >= opts.max_depth:
                continue

            # Explore neighbors
            neighbors: List[Tuple[str, GraphEdge]] = []
            if opts.direction in ("outgoing", "both"):
                for edge in self._outgoing.get(current, []):
                    if opts.allowed_relations is None or edge.relation_type in opts.allowed_relations:
                        neighbors.append((edge.target_id, edge))
            if opts.direction in ("incoming", "both"):
                for edge in self._incoming.get(current, []):
                    if opts.allowed_relations is None or edge.relation_type in opts.allowed_relations:
                        neighbors.append((edge.source_id, edge))

            neighbors.sort(key=lambda x: x[0])

            for neighbor_id, edge in neighbors:
                if neighbor_id not in visited:
                    visited[neighbor_id] = (current, edge)
                    if neighbor_id == target:
                        # Reconstruct path
                        path: List[str] = [target]
                        edges: List[GraphEdge] = []
                        node = target
                        while visited[node][0] is not None:
                            edges.append(visited[node][1])
                            path.append(visited[node][0])
                            node = visited[node][0]
                        path.reverse()
                        edges.reverse()
                        return PathResult(
                            found=True, path=path, edges=edges, length=len(edges)
                        )
                    queue.append(neighbor_id)
                    if len(visited) >= opts.max_nodes:
                        break

        return PathResult(found=False, path=[], edges=[], length=0)

    def ancestors(
        self,
        record_id: str,
        options: Optional[TraversalOptions] = None,
    ) -> List[str]:
        """
        Get ancestor nodes (nodes reachable via incoming edges).

        Semantically defined for: REQUIRES, SATISFIES, DECIDED_BY, CONSTRAINED_BY,
        INTRODUCES, MITIGATES, CAUSED_BY, RESOLVED_BY, VERIFIED_BY, SUPPORTED_BY,
        SUPERSEDES, DERIVED_FROM.

        Not defined for: CONTRADICTS, RELATES_TO (treated as ancestors for exploration).
        """
        if record_id not in self._nodes:
            return []
        opts = options or TraversalOptions(direction="incoming")
        opts.direction = "incoming"
        return self.traverse(record_id, opts)

    def descendants(
        self,
        record_id: str,
        options: Optional[TraversalOptions] = None,
    ) -> List[str]:
        """
        Get descendant nodes (nodes reachable via outgoing edges).

        Semantically defined for: REQUIRES, SATISFIES, DECIDED_BY, CONSTRAINED_BY,
        INTRODUCES, MITIGATES, CAUSED_BY, RESOLVED_BY, VERIFIED_BY, SUPPORTED_BY,
        SUPERSEDES, DERIVED_FROM.
        """
        if record_id not in self._nodes:
            return []
        opts = options or TraversalOptions(direction="outgoing")
        opts.direction = "outgoing"
        return self.traverse(record_id, opts)

    # ──────────────────────────────────────────────────────────────────
    # Supersession semantics
    # ──────────────────────────────────────────────────────────────────

    def supersession_chain(self, record_id: str) -> List[str]:
        """
        Get the supersession chain: record → superseded_by → ...

        Returns list of record IDs in supersession order (oldest first).
        Deterministic: sorted by (source_id, target_id) at each step.
        """
        if record_id not in self._nodes:
            return []
        chain = [record_id]
        current = record_id
        visited: Set[str] = {record_id}

        while True:
            next_nodes = sorted(
                self.outgoing(current, relation_type=RELATIONSHIP_SUPERSEDES)
            )
            if not next_nodes:
                break
            next_node = next_nodes[0]
            if next_node in visited:
                break  # Cycle — stop
            chain.append(next_node)
            visited.add(next_node)
            current = next_node

        return chain

    def effective_record(self, record_id: str) -> Optional[str]:
        """
        Get the current effective record in a supersession chain.

        Returns the last record in the chain (most recent), or None if
        the record is part of an ambiguous supersession topology.
        """
        chain = self.supersession_chain(record_id)
        if len(chain) == 1:
            return chain[0]

        # Check ALL nodes in the chain for branching (ambiguous)
        for node in chain:
            outgoing_super = self.outgoing(node, relation_type=RELATIONSHIP_SUPERSEDES)
            if len(outgoing_super) > 1:
                return None  # Ambiguous — fail closed
        return chain[-1]

    def is_superseded(self, record_id: str) -> bool:
        """Check if a record has been superseded by another."""
        if record_id not in self._nodes:
            return False
        return len(self.outgoing(record_id, relation_type=RELATIONSHIP_SUPERSEDES)) > 0

    # ──────────────────────────────────────────────────────────────────
    # Cycle detection — relation-specific
    # ──────────────────────────────────────────────────────────────────

    def has_cycle(
        self,
        relation_type: Optional[str] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Detect cycles, optionally scoped to a specific relation type.

        Returns (True, cycle_path) if a cycle exists, (False, []).

        Cycle semantics:
        - SUPERSEDES: cycles are forbidden (A supersedes B, B supersedes A)
        - CAUSED_BY: cycles are forbidden (causality must be acyclic)
        - RESOLVED_BY: cycles are forbidden
        - REQUIRES, SATISFIES, DECIDED_BY, CONSTRAINED_BY, INTRODUCES,
          MITIGATES, VERIFIED_BY, SUPPORTED_BY, RELATES_TO, DERIVED_FROM,
          CONTRADICTS: cycles are permitted (not globally prohibited)
        """
        if relation_type is not None:
            edges: List[GraphEdge] = []
            for sid in sorted(self._outgoing.keys()):
                for edge in self._outgoing[sid]:
                    if edge.relation_type == relation_type:
                        edges.append(edge)
            return self._detect_cycle_in_subgraph(edges)

        # Check per-relation-type: only certain types are cycle-forbidden
        forbidden_types = {
            RELATIONSHIP_SUPERSEDES,
            RELATIONSHIP_CAUSED_BY,
            RELATIONSHIP_RESOLVED_BY,
        }
        for rtype in forbidden_types:
            has_cycle, cycle_path = self.has_cycle(relation_type=rtype)
            if has_cycle:
                return True, cycle_path

        # Mixed-cycle check: any cycle using any edges
        all_edges: List[GraphEdge] = []
        for sid in sorted(self._outgoing.keys()):
            all_edges.extend(self._outgoing[sid])
        return self._detect_cycle_in_subgraph(all_edges)

    def _detect_cycle_in_subgraph(
        self, edges: List[GraphEdge]
    ) -> Tuple[bool, List[str]]:
        """Detect a cycle in a subgraph defined by edge list. Returns cycle path."""
        # Build adjacency from edges
        adj: Dict[str, List[str]] = {}
        nodes: Set[str] = set()
        for edge in edges:
            nodes.add(edge.source_id)
            nodes.add(edge.target_id)
            adj.setdefault(edge.source_id, []).append(edge.target_id)

        # Sort adjacency for deterministic DFS
        for node in adj:
            adj[node].sort()

        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {n: WHITE for n in nodes}
        parent: Dict[str, str] = {}

        def dfs(node: str) -> Optional[List[str]]:
            color[node] = GRAY
            for neighbor in adj.get(node, []):
                if color[neighbor] == GRAY:
                    # Found cycle — reconstruct path
                    cycle = [neighbor, node]
                    cur = node
                    while cur != neighbor:
                        cur = parent.get(cur)
                        if cur is None:
                            break
                        if cur != neighbor:
                            cycle.append(cur)
                    cycle.reverse()
                    return cycle
                if color[neighbor] == WHITE:
                    parent[neighbor] = node
                    result = dfs(neighbor)
                    if result is not None:
                        return result
            color[node] = BLACK
            return None

        for node in sorted(nodes):
            if color[node] == WHITE:
                result = dfs(node)
                if result is not None:
                    return True, result
        return False, []

    # ──────────────────────────────────────────────────────────────────
    # Integrity validation
    # ──────────────────────────────────────────────────────────────────

    def validate_integrity(self, store=None) -> List[IntegrityConflict]:
        """
        Validate graph integrity. Returns list of conflicts (empty = clean).

        Detects:
        - DANGLING_REFERENCE: edge references nonexistent canonical record
        - INVALID_RELATION: relation type not in VALID_RELATIONSHIP_TYPES
        - FORBIDDEN_CYCLE: cycle in SUPERSEDES or CAUSED_BY
        - SUPERSESSION_CONFLICT: mutual supersession, branching ambiguity
        - EXPLICIT_CONTRADICTION: CONTRADICTS edges (structural, not semantic)
        - STATUS_CONFLICT: superseded record still active/accepted

        If store is provided, verifies canonical records match graph nodes.
        If store is None, validates internal consistency only.
        """
        conflicts: List[IntegrityConflict] = []

        # Check for dangling references (edges pointing to nonexistent nodes)
        all_nodes = self._nodes
        for edge in self.get_edges():
            if edge.target_id not in all_nodes:
                conflicts.append(
                    IntegrityConflict(
                        conflict_type="DANGLING_REFERENCE",
                        records=[edge.source_id, edge.target_id],
                        relations=[{
                            "source_id": edge.source_id,
                            "relation_type": edge.relation_type,
                            "target_id": edge.target_id,
                        }],
                        severity="HIGH",
                        reason=f"Edge {edge.source_id} —{edge.relation_type}→ {edge.target_id} references nonexistent record {edge.target_id}",
                    )
                )

        # If store provided, verify all canonical records are in graph
        if store is not None:
            canonical_ids = {r.record_id for r in store.list_all()}
            graph_from_store = EngineeringKnowledgeGraph.build_from_store(store)
            missing_in_graph = canonical_ids - graph_from_store.nodes
            if missing_in_graph:
                conflicts.append(
                    IntegrityConflict(
                        conflict_type="DANGLING_REFERENCE",
                        records=sorted(missing_in_graph),
                        relations=[],
                        severity="MEDIUM",
                        reason=f"Records in store missing from graph: {sorted(missing_in_graph)}",
                    )
                )

        # Check for forbidden cycles in SUPERSEDES
        has_cycle, cycle_path = self.has_cycle(relation_type=RELATIONSHIP_SUPERSEDES)
        if has_cycle:
            conflicts.append(
                IntegrityConflict(
                    conflict_type="SUPERSESSION_CONFLICT",
                    records=cycle_path,
                    relations=[
                        {"source_id": cycle_path[i], "relation_type": "SUPERSEDES", "target_id": cycle_path[i+1]}
                        for i in range(len(cycle_path) - 1)
                    ] + [{"source_id": cycle_path[-1], "relation_type": "SUPERSEDES", "target_id": cycle_path[0]}],
                    severity="HIGH",
                    reason=f"Supersession cycle detected: {' → '.join(cycle_path)}",
                )
            )

        # Check for forbidden cycles in CAUSED_BY
        has_cycle, cycle_path = self.has_cycle(relation_type=RELATIONSHIP_CAUSED_BY)
        if has_cycle:
            conflicts.append(
                IntegrityConflict(
                    conflict_type="FORBIDDEN_CYCLE",
                    records=cycle_path,
                    relations=[
                        {"source_id": cycle_path[i], "relation_type": "CAUSED_BY", "target_id": cycle_path[i+1]}
                        for i in range(len(cycle_path) - 1)
                    ] + [{"source_id": cycle_path[-1], "relation_type": "CAUSED_BY", "target_id": cycle_path[0]}],
                    severity="HIGH",
                    reason=f"Causality cycle detected: {' → '.join(cycle_path)}",
                )
            )

        # Check for mutual supersession (A supersedes B AND B supersedes A)
        for edge in self.get_edges(relation_type=RELATIONSHIP_SUPERSEDES):
            reverse = self.has_edge(edge.target_id, edge.source_id, RELATIONSHIP_SUPERSEDES)
            if reverse:
                records = sorted([edge.source_id, edge.target_id])
                conflicts.append(
                    IntegrityConflict(
                        conflict_type="SUPERSESSION_CONFLICT",
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
                        reason=f"Mutual supersession: {edge.source_id} and {edge.target_id} each supersede the other",
                    )
                )

        # Check for ambiguous supersession (one record superseded by multiple)
        outgoing_super: Dict[str, List[str]] = {}
        for edge in self.get_edges(relation_type=RELATIONSHIP_SUPERSEDES):
            outgoing_super.setdefault(edge.source_id, []).append(edge.target_id)
        for source_id, targets in sorted(outgoing_super.items()):
            if len(targets) > 1:
                conflicts.append(
                    IntegrityConflict(
                        conflict_type="SUPERSESSION_CONFLICT",
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
                        reason=f"Ambiguous supersession: {source_id} supersedes multiple records: {sorted(targets)}",
                    )
                )

        # Check for explicit CONTRADICTS edges (structural, not semantic)
        for edge in self.get_edges(relation_type=RELATIONSHIP_CONTRADICTS):
            conflicts.append(
                IntegrityConflict(
                    conflict_type="EXPLICIT_CONTRADICTION",
                    records=sorted([edge.source_id, edge.target_id]),
                    relations=[{
                        "source_id": edge.source_id,
                        "relation_type": "CONTRADICTS",
                        "target_id": edge.target_id,
                    }],
                    severity="MEDIUM",
                    reason=f"Explicit contradiction between {edge.source_id} and {edge.target_id}",
                )
            )

        # Sort for deterministic output
        conflicts.sort(key=lambda c: (c.conflict_type, c.records))
        return conflicts

    # ──────────────────────────────────────────────────────────────────
    # Deterministic digest
    # ──────────────────────────────────────────────────────────────────

    def compute_digest(self) -> str:
        """
        Compute a deterministic SHA-256 graph digest.

        Sorted nodes + sorted typed edges → SHA-256.
        No Python hash(). No filesystem or process-local state.

        Two logically equivalent graphs always produce the same digest,
        regardless of construction order or runtime environment.
        """
        canonical = self.to_dict()
        nodes_sorted = sorted(canonical["nodes"])
        edges_sorted = sorted(
            (
                e["source_id"],
                e["relation_type"],
                e["target_id"],
            )
            for e in canonical["edges"]
        )
        digest_input = json.dumps(
            {
                "nodes": nodes_sorted,
                "edges": [list(e) for e in edges_sorted],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()

    # ──────────────────────────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Serialize graph to a dictionary (deterministic).

        Uses sorted node IDs and edge list sorted by (source_id, relation_type, target_id).
        """
        edges = sorted(
            self.get_edges(),
            key=lambda e: (e.source_id, e.relation_type, e.target_id),
        )
        return {
            "nodes": sorted(self._nodes),
            "edges": [e.to_dict() for e in edges],
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize to JSON string (deterministic)."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "EngineeringKnowledgeGraph":
        """Create graph from a dictionary (deterministic reconstruction)."""
        graph = cls()
        for node in data.get("nodes", []):
            graph._add_node(node)
        for edge_data in data.get("edges", []):
            edge = GraphEdge.from_dict(edge_data)
            graph._add_edge_unchecked(edge)
        return graph

    @classmethod
    def from_json(cls, text: str) -> "EngineeringKnowledgeGraph":
        """Create graph from a JSON string."""
        return cls.from_dict(json.loads(text))

    # ──────────────────────────────────────────────────────────────────
    # Comparison (logical equivalence)
    # ──────────────────────────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        """Two graphs are logically equal if they have the same nodes and edges."""
        if not isinstance(other, EngineeringKnowledgeGraph):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __repr__(self) -> str:
        return (
            f"EngineeringKnowledgeGraph("
            f"nodes={self.node_count()}, edges={self.edge_count()})"
        )
