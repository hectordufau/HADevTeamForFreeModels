# harness/knowledge/graph.py — V3.3 Engineering Knowledge Graph
"""
Directed graph connecting engineering records via typed relationships.

Provides:
- Node and edge management with typed relationships
- BFS/DFS traversal with depth and type filtering
- Cycle detection (graph must remain a DAG)
- Persistence to JSONL (append-only edge log) and JSON (full snapshot)
- O(1) node index for fast lookup

Architecture:
- Adjacency list storage (node -> set of (target, edge_type) tuples)
- Bidirectional index for efficient incoming-edge lookups
- All edge types are validated against VALID_EDGE_TYPES
- Graph MUST remain acyclic (DAG invariant)

Domain separation:
- KnowledgeGraph manages EngineeringRecord relationships only
- Does NOT touch Learning or Evidence domains
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple
from collections import deque


class GraphError(Exception):
    """Raised on graph operation errors."""


# ──────────────────────────────────────────────────────────────────────
# Edge types (relationship types between engineering records)
# ──────────────────────────────────────────────────────────────────────

EDGE_TYPE_GOVERNS = "governs"           # PRD → NFR, PRD → REQ
EDGE_TYPE_INFORMS = "informs"           # PRD → ADR, evidence → decision
EDGE_TYPE_CONSTRAINS = "constrains"     # NFR → ADR, SEC → ADR
EDGE_TYPE_ADDRESSES = "addresses"       # DR → PRD, ADR → requirement
EDGE_TYPE_SATISFIES = "satisfies"       # SEC → NFR, implementation → REQ
EDGE_TYPE_CREATES = "creates"           # ADR → TDR, change → debt
EDGE_TYPE_INTRODUCES = "introduces"     # ADR → RSK, decision → risk
EDGE_TYPE_ENFORCES = "enforces"         # ADR → SEC, policy → control
EDGE_TYPE_MOTIVATES = "motivates"       # RCA → ADR, failure → decision
EDGE_TYPE_IDENTIFIES = "identifies"     # RCA → TDR, analysis → debt

EDGE_TYPES = [
    EDGE_TYPE_GOVERNS,
    EDGE_TYPE_INFORMS,
    EDGE_TYPE_CONSTRAINS,
    EDGE_TYPE_ADDRESSES,
    EDGE_TYPE_SATISFIES,
    EDGE_TYPE_CREATES,
    EDGE_TYPE_INTRODUCES,
    EDGE_TYPE_ENFORCES,
    EDGE_TYPE_MOTIVATES,
    EDGE_TYPE_IDENTIFIES,
]

# For deserialization: map string → string (edge types are strings, not enums,
# but we expose EdgeType as constants for API clarity)
VALID_EDGE_TYPES = set(EDGE_TYPES)


# ──────────────────────────────────────────────────────────────────────
# KnowledgeGraph
# ──────────────────────────────────────────────────────────────────────


class KnowledgeGraph:
    """
    Directed acyclic graph of engineering record relationships.

    Usage:
        graph = KnowledgeGraph()
        graph.add_edge("PRD-001", "NFR-001", "governs")
        related = graph.get_related("PRD-001")
        path = graph.traverse("PRD-001", method="bfs")
    """

    def __init__(self):
        # Adjacency list: node_id → list of (target_id, edge_type)
        self._outgoing: Dict[str, List[Tuple[str, str]]] = {}
        # Reverse index: node_id → list of (source_id, edge_type)
        self._incoming: Dict[str, List[Tuple[str, str]]] = {}
        # Node index for O(1) lookup: node_id → True
        self._node_index: Dict[str, bool] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Node management
    # ──────────────────────────────────────────────────────────────────────

    def add_node(self, node_id: str) -> None:
        """Add a node to the graph. Idempotent."""
        if not isinstance(node_id, str) or not node_id:
            raise GraphError("node_id must be a non-empty string")
        if node_id not in self._node_index:
            self._node_index[node_id] = True
            self._outgoing[node_id] = []
            self._incoming[node_id] = []

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self._node_index

    def get_nodes(self) -> List[str]:
        """Return all node IDs in the graph."""
        return list(self._node_index.keys())

    def node_count(self) -> int:
        """Return the number of nodes."""
        return len(self._node_index)

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node and all its connected edges.
        Raises GraphError if node does not exist.
        """
        if node_id not in self._node_index:
            raise GraphError(f"Node '{node_id}' does not exist")

        # Remove outgoing edges
        for target_id, _ in self._outgoing[node_id]:
            self._incoming[target_id] = [
                (s, t) for s, t in self._incoming[target_id] if s != node_id
            ]

        # Remove incoming edges
        for source_id, _ in self._incoming[node_id]:
            self._outgoing[source_id] = [
                (t, e) for t, e in self._outgoing[source_id] if t != node_id
            ]

        del self._outgoing[node_id]
        del self._incoming[node_id]
        del self._node_index[node_id]

    # ──────────────────────────────────────────────────────────────────────
    # Edge management
    # ──────────────────────────────────────────────────────────────────────

    def add_edge(self, source_id: str, target_id: str, edge_type: str) -> None:
        """
        Add a directed edge from source to target with the given type.

        Raises GraphError if:
        - edge_type is invalid
        - source == target (self-reference)
        - adding the edge would create a cycle (source reachable from target)
        """
        if not isinstance(edge_type, str) or edge_type not in VALID_EDGE_TYPES:
            raise GraphError(
                f"Invalid edge type '{edge_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_EDGE_TYPES))}"
            )

        if source_id == target_id:
            raise GraphError("Self-references are not allowed")

        # Ensure both nodes exist
        self.add_node(source_id)
        self.add_node(target_id)

        # Check for duplicate
        for t, e in self._outgoing[source_id]:
            if t == target_id and e == edge_type:
                return  # Already exists

        # Cycle check: if target can reach source, adding source→target creates cycle
        if self._is_reachable(target_id, source_id):
            raise GraphError(
                f"Adding edge '{source_id} → {target_id}' would create a cycle"
            )

        self._outgoing[source_id].append((target_id, edge_type))
        self._incoming[target_id].append((source_id, edge_type))

    def has_edge(self, source_id: str, target_id: str, edge_type: Optional[str] = None) -> bool:
        """Check if an edge exists, optionally filtered by type."""
        if source_id not in self._outgoing:
            return False
        for t, e in self._outgoing[source_id]:
            if t == target_id:
                if edge_type is None or e == edge_type:
                    return True
        return False

    def get_edge_types(self, source_id: str, target_id: str) -> List[str]:
        """Get all edge types between two nodes."""
        if source_id not in self._outgoing:
            return []
        return [e for t, e in self._outgoing[source_id] if t == target_id]

    def edge_count(self) -> int:
        """Return the total number of edges."""
        return sum(len(edges) for edges in self._outgoing.values())

    def get_edges(self) -> List[Dict]:
        """Return all edges as a list of dicts."""
        edges = []
        for source_id in sorted(self._outgoing.keys()):
            for target_id, edge_type in self._outgoing[source_id]:
                edges.append({
                    "source_id": source_id,
                    "target_id": target_id,
                    "relation_type": edge_type,
                })
        return edges

    # ──────────────────────────────────────────────────────────────────────
    # Traversal & queries
    # ──────────────────────────────────────────────────────────────────────

    def get_related(
        self,
        node_id: str,
        direction: str = "outgoing",
        edge_type: Optional[str] = None,
    ) -> List[str]:
        """
        Get related node IDs.

        Args:
            node_id: The node to query.
            direction: "outgoing", "incoming", or "both".
            edge_type: Optional filter by edge type.

        Returns:
            List of related node IDs (may contain duplicates if multiple
            edge types connect the same pair).

        Raises GraphError on invalid direction.
        """
        if direction not in ("outgoing", "incoming", "both"):
            raise GraphError(
                f"Invalid direction '{direction}'. "
                "Must be 'outgoing', 'incoming', or 'both'"
            )

        if node_id not in self._node_index:
            return []

        results = []
        seen = set()

        if direction in ("outgoing", "both"):
            for target_id, e in self._outgoing[node_id]:
                if edge_type is None or e == edge_type:
                    if target_id not in seen:
                        results.append(target_id)
                        seen.add(target_id)

        if direction in ("incoming", "both"):
            for source_id, e in self._incoming[node_id]:
                if edge_type is None or e == edge_type:
                    if source_id not in seen:
                        results.append(source_id)
                        seen.add(source_id)

        return results

    def traverse(
        self,
        start_id: str,
        method: str = "bfs",
        max_depth: Optional[int] = None,
        edge_type: Optional[str] = None,
    ) -> List[str]:
        """
        Traverse the graph from a starting node.

        Args:
            start_id: Node to start from.
            method: "bfs" or "dfs".
            max_depth: Maximum traversal depth (None = unlimited).
            edge_type: Only follow edges of this type.

        Returns:
            List of visited node IDs in traversal order (start node first).

        Raises GraphError on invalid method.
        """
        if method not in ("bfs", "dfs"):
            raise GraphError(f"Invalid traversal method '{method}'. Must be 'bfs' or 'dfs'")

        if start_id not in self._node_index:
            return []

        visited = {start_id}
        order = [start_id]

        if method == "bfs":
            queue = deque([(start_id, 0)])
            while queue:
                current, depth = queue.popleft()
                if max_depth is not None and depth >= max_depth:
                    continue
                for neighbor, e in self._outgoing[current]:
                    if edge_type is not None and e != edge_type:
                        continue
                    if neighbor not in visited:
                        visited.add(neighbor)
                        order.append(neighbor)
                        queue.append((neighbor, depth + 1))
        else:  # dfs
            stack = [(start_id, 0)]
            while stack:
                current, depth = stack.pop()
                if max_depth is not None and depth >= max_depth:
                    continue
                for neighbor, e in reversed(self._outgoing[current]):
                    if edge_type is not None and e != edge_type:
                        continue
                    if neighbor not in visited:
                        visited.add(neighbor)
                        order.append(neighbor)
                        stack.append((neighbor, depth + 1))

        return order

    def _is_reachable(self, start: str, target: str, visited: Optional[Set[str]] = None) -> bool:
        """Check if target is reachable from start via BFS."""
        if visited is None:
            visited = set()
        if start == target:
            return True
        if start in visited:
            return False
        visited.add(start)
        for neighbor, _ in self._outgoing.get(start, []):
            if self._is_reachable(neighbor, target, visited):
                return True
        return False

    def has_cycle(self) -> bool:
        """Check if the graph contains any cycle (DFS-based)."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {node: WHITE for node in self._node_index}

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for neighbor, _ in self._outgoing.get(node, []):
                if color[neighbor] == GRAY:
                    return True
                if color[neighbor] == WHITE and dfs(neighbor):
                    return True
            color[node] = BLACK
            return False

        for node in self._node_index:
            if color[node] == WHITE:
                if dfs(node):
                    return True
        return False

    # ──────────────────────────────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize graph to a dictionary (deterministic)."""
        return {
            "nodes": sorted(self._node_index.keys()),
            "edges": self.get_edges(),
        }

    def save(self, path: str, format: str = "jsonl") -> None:
        """
        Persist the graph to disk.

        Args:
            path: File path to save to.
            format: "jsonl" (append-only edge log) or "json" (full snapshot).

        Raises GraphError on invalid format or write failure.
        """
        if format not in ("jsonl", "json"):
            raise GraphError(f"Invalid format '{format}'. Must be 'jsonl' or 'json'")

        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        try:
            if format == "jsonl":
                with open(path, "w", encoding="utf-8") as f:
                    for edge in self.get_edges():
                        f.write(json.dumps(edge, sort_keys=True) + "\n")
            else:  # json
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.to_dict(), f, sort_keys=True, indent=2)
        except OSError as e:
            raise GraphError(f"Failed to save graph: {e}")

    @classmethod
    def load(cls, path: str, format: str = "jsonl") -> "KnowledgeGraph":
        """
        Load a graph from disk.

        Args:
            path: File path to load from.
            format: "jsonl" or "json" (must match the save format).

        Returns:
            Reconstructed KnowledgeGraph.

        Raises GraphError on invalid format or read failure.
        """
        if format not in ("jsonl", "json"):
            raise GraphError(f"Invalid format '{format}'. Must be 'jsonl' or 'json'")

        if not os.path.exists(path):
            raise GraphError(f"Graph file not found: {path}")

        graph = cls()

        try:
            if format == "jsonl":
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        edge = json.loads(line)
                        graph.add_edge(
                            edge["source_id"],
                            edge["target_id"],
                            edge["relation_type"],
                        )
            else:  # json
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for node in data.get("nodes", []):
                    graph.add_node(node)
                for edge in data.get("edges", []):
                    graph.add_edge(
                        edge["source_id"],
                        edge["target_id"],
                        edge["relation_type"],
                    )
        except (OSError, json.JSONDecodeError) as e:
            raise GraphError(f"Failed to load graph: {e}")

        return graph

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        """Create a KnowledgeGraph from a dictionary."""
        graph = cls()
        for node in data.get("nodes", []):
            graph.add_node(node)
        for edge in data.get("edges", []):
            graph.add_edge(
                edge["source_id"],
                edge["target_id"],
                edge["relation_type"],
            )
        return graph


# ──────────────────────────────────────────────────────────────────────
# EdgeType registry (for API clarity — these are the canonical edge types)
# ──────────────────────────────────────────────────────────────────────

class EdgeType:
    """Canonical edge type constants for the knowledge graph."""
    GOVERNS = EDGE_TYPE_GOVERNS
    INFORMS = EDGE_TYPE_INFORMS
    CONSTRAINS = EDGE_TYPE_CONSTRAINS
    ADDRESSES = EDGE_TYPE_ADDRESSES
    SATISFIES = EDGE_TYPE_SATISFIES
    CREATES = EDGE_TYPE_CREATES
    INTRODUCES = EDGE_TYPE_INTRODUCES
    ENFORCES = EDGE_TYPE_ENFORCES
    MOTIVATES = EDGE_TYPE_MOTIVATES
    IDENTIFIES = EDGE_TYPE_IDENTIFIES
