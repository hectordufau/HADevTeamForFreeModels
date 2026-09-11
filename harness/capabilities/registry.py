# harness/capabilities/registry.py — CapabilityRegistry (V2.2)
"""
Maps agents to their declared capabilities.
Supports scoring: exact=1.0, child=0.9, parent=0.7, sibling=0.4.
"""

from typing import Any, Dict, List, Optional, Tuple


class CapabilityRegistryError(Exception):
    """Raised on capability registry errors."""


class CapabilityRegistry:
    """Maps agents to their declared capabilities and supports lookup/scoring."""

    def __init__(self):
        # role -> list of capability strings
        self._agents: Dict[str, List[str]] = {}
        # capability -> set of roles
        self._capability_index: Dict[str, set] = {}

    def register_agent(self, role: str, capabilities: List[str]):
        """Register an agent with its capabilities."""
        if not role:
            raise CapabilityRegistryError("Role cannot be empty")
        self._agents[role] = list(capabilities)
        for cap in capabilities:
            if cap not in self._capability_index:
                self._capability_index[cap] = set()
            self._capability_index[cap].add(role)

    def get_capabilities(self, role: str) -> List[str]:
        """Get capabilities for a registered agent."""
        if role not in self._agents:
            raise CapabilityRegistryError(f"Agent '{role}' not registered")
        return self._agents[role]

    def find_agents_for(self, capability: str) -> List[str]:
        """Find agents that declare this capability (exact match)."""
        return sorted(self._capability_index.get(capability, set()))

    def find_best_agent(self, required_capabilities: List[str],
                        taxonomy: Any = None,
                        scoring: Optional[Dict[str, float]] = None) -> Tuple[Optional[str], float]:
        """
        Find the agent best matching a set of required capabilities.

        Uses scoring weights: exact=1.0, child=0.9, parent=0.7, sibling=0.4.
        If taxonomy is provided, also considers hierarchical matches.

        Returns (agent_role, score) or (None, 0.0) if no match.
        """
        if not self._agents:
            return None, 0.0

        if scoring is None:
            scoring = {"exact": 1.0, "child": 0.9, "parent": 0.7, "sibling": 0.4}

        best_role = None
        best_score = -1.0

        for role, agent_caps in self._agents.items():
            score = self._score_agent(
                role, agent_caps, required_capabilities, taxonomy, scoring
            )
            if score > best_score:
                best_score = score
                best_role = role

        return best_role, best_score

    def score_agent_for_capabilities(self, role: str,
                                     required_capabilities: List[str],
                                     taxonomy: Any = None,
                                     scoring: Optional[Dict[str, float]] = None) -> float:
        """Score a single agent against required capabilities."""
        if role not in self._agents:
            return 0.0
        return self._score_agent(
            role, self._agents[role], required_capabilities, taxonomy, scoring
        )

    def _score_agent(self, role: str, agent_caps: List[str],
                     required_caps: List[str],
                     taxonomy: Any,
                     scoring: Optional[Dict[str, float]]) -> float:
        """Internal scoring logic."""
        if not required_caps:
            return 0.0

        if scoring is None:
            scoring = {"exact": 1.0, "child": 0.9, "parent": 0.7, "sibling": 0.4}

        set_caps = set(agent_caps)
        total = 0.0

        for req in required_caps:
            # Exact match
            if req in set_caps:
                total += scoring["exact"]
                continue

            # Taxonomic matches if taxonomy is provided
            if taxonomy:
                # Child match: agent declares a sub-capability
                children = taxonomy.get_children(req) if hasattr(taxonomy, 'get_children') else []
                if any(c in set_caps for c in children):
                    total += scoring["child"]
                    continue

                # Parent match: agent declares a parent capability
                parents = taxonomy.get_parents(req) if hasattr(taxonomy, 'get_parents') else []
                if any(p in set_caps for p in parents):
                    total += scoring["parent"]
                    continue

                # Sibling match: agent declares sibling under same parent
                siblings = taxonomy.get_siblings(req) if hasattr(taxonomy, 'get_siblings') else []
                if any(s in set_caps for s in siblings):
                    total += scoring["sibling"]
                    continue

        # Normalize by number of required capabilities
        return total / len(required_caps)

    def list_agents(self) -> List[str]:
        """List all registered agent roles."""
        return sorted(self._agents.keys())

    def unregister_agent(self, role: str):
        """Remove an agent from the registry."""
        if role in self._agents:
            caps = self._agents.pop(role)
            for cap in caps:
                if cap in self._capability_index:
                    self._capability_index[cap].discard(role)
                    if not self._capability_index[cap]:
                        del self._capability_index[cap]
