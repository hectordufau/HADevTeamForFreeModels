# harness/capabilities/graph.py — CapabilityGraph (V2.2)
"""
DAG of capability nodes with dependency resolution.
Supports cycle detection, topological sort, and ready-node computation.
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field


class CapabilityGraphError(Exception):
    """Raised on capability graph errors."""


@dataclass
class CapabilityNode:
    """A node in the capability dependency graph."""
    id: str
    capability: str
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed, skipped
    result: Optional[dict] = None


class CapabilityGraph:
    """DAG of capability nodes with dependency resolution."""

    def __init__(self):
        self._nodes: Dict[str, CapabilityNode] = {}
        self._edges: Dict[str, Set[str]] = {}  # node_id -> set of dependency node_ids

    def add_node(self, node: CapabilityNode):
        """Add a node to the graph."""
        if node.id in self._nodes:
            raise CapabilityGraphError(f"Node '{node.id}' already exists")
        self._nodes[node.id] = node
        self._edges[node.id] = set(node.dependencies)

    def get_node(self, node_id: str) -> Optional[CapabilityNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_ready_nodes(self) -> List[CapabilityNode]:
        """Nodes whose dependencies are all satisfied (completed or skipped)."""
        ready = []
        for node_id, node in self._nodes.items():
            if node.status != "pending":
                continue
            deps = self._edges.get(node_id, set())
            if all(
                dep_id in self._nodes
                and self._nodes[dep_id].status in ("completed", "skipped")
                for dep_id in deps
            ):
                ready.append(node)
        return ready

    def has_cycle(self) -> bool:
        """Detect cycles using DFS (White-Gray-Black)."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in self._nodes}

        def dfs(node_id):
            if color[node_id] == GRAY:
                return True  # back edge -> cycle
            if color[node_id] == BLACK:
                return False
            color[node_id] = GRAY
            for dep in self._edges.get(node_id, set()):
                if dep in self._nodes and dfs(dep):
                    return True
            color[node_id] = BLACK
            return False

        for nid in self._nodes:
            if dfs(nid):
                return True
        return False

    def topological_sort(self) -> List[CapabilityNode]:
        """Return nodes in execution order (Kahn's algorithm)."""
        if self.has_cycle():
            raise CapabilityGraphError("Cannot sort: graph contains a cycle")

        # Compute in-degree
        in_degree: Dict[str, int] = {}
        for nid in self._nodes:
            in_degree.setdefault(nid, 0)
            for dep in self._edges.get(nid, set()):
                if dep in self._nodes:
                    in_degree[nid] = in_degree.get(nid, 0) + 1

        # Queue nodes with no dependencies
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        sorted_nodes = []

        while queue:
            nid = queue.pop(0)
            sorted_nodes.append(self._nodes[nid])
            # Reduce in-degree for nodes that depend on this one
            for other_id, other_node in self._nodes.items():
                if nid in self._edges.get(other_id, set()):
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)

        if len(sorted_nodes) != len(self._nodes):
            raise CapabilityGraphError("Graph is not a DAG (cannot fully sort)")

        return sorted_nodes

    def mark_completed(self, node_id: str, result: Optional[dict] = None):
        """Mark a node as completed."""
        if node_id not in self._nodes:
            raise CapabilityGraphError(f"Node '{node_id}' not found")
        self._nodes[node_id].status = "completed"
        if result:
            self._nodes[node_id].result = result

    def mark_failed(self, node_id: str, result: Optional[dict] = None):
        """Mark a node as failed."""
        if node_id not in self._nodes:
            raise CapabilityGraphError(f"Node '{node_id}' not found")
        self._nodes[node_id].status = "failed"
        if result:
            self._nodes[node_id].result = result

    def mark_skipped(self, node_id: str, result: Optional[dict] = None):
        """Mark a node as skipped."""
        if node_id not in self._nodes:
            raise CapabilityGraphError(f"Node '{node_id}' not found")
        self._nodes[node_id].status = "skipped"
        if result:
            self._nodes[node_id].result = result

    def get_all_nodes(self) -> List[CapabilityNode]:
        """Return all nodes."""
        return list(self._nodes.values())

    def get_node_count(self) -> int:
        return len(self._nodes)

    def to_dict(self) -> dict:
        """Serialize graph to dict."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "capability": n.capability,
                    "dependencies": n.dependencies,
                    "status": n.status,
                    "result": n.result,
                }
                for n in self._nodes.values()
            ],
            "edges": {nid: list(deps) for nid, deps in self._edges.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CapabilityGraph":
        """Deserialize graph from dict."""
        graph = cls()
        for nd in data.get("nodes", []):
            graph.add_node(CapabilityNode(
                id=nd["id"],
                capability=nd["capability"],
                dependencies=nd.get("dependencies", []),
                status=nd.get("status", "pending"),
                result=nd.get("result"),
            ))
        return graph
