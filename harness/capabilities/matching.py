# harness/capabilities/matching.py — CapabilityMatching (V2.2)
"""
Scores agents against required capabilities using taxonomic matching.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass


class CapabilityMatchingError(Exception):
    """Raised on capability matching errors."""


@dataclass
class MatchResult:
    """Result of matching an agent against a capability."""
    agent: str
    capability: str
    score: float          # 0.0 to 1.0
    match_type: str       # exact, child, parent, sibling, none


class CapabilityMatcher:
    """Finds best agent(s) for required capabilities using taxonomic matching."""

    # Scoring weights
    SCORING = {
        "exact": 1.0,
        "child": 0.9,
        "parent": 0.7,
        "sibling": 0.4,
        "none": 0.0,
    }

    def __init__(self, registry: Any, taxonomy: Any):
        self.registry = registry
        self.taxonomy = taxonomy

    def match(self, required: List[str]) -> List[MatchResult]:
        """
        Score all agents against each required capability.

        Returns a flat list of MatchResult (one per capability per agent).
        """
        results = []
        agents = self.registry.list_agents()

        for cap in required:
            for agent in agents:
                match_type, score = self._match_single(agent, cap)
                results.append(MatchResult(
                    agent=agent,
                    capability=cap,
                    score=score,
                    match_type=match_type,
                ))

        return results

    def best_match(self, required: List[str]) -> MatchResult:
        """
        Return single best agent for a set of capabilities.

        Aggregate score across all required capabilities.
        """
        if not required:
            raise CapabilityMatchingError("Required capabilities list is empty")

        agents = self.registry.list_agents()
        if not agents:
            raise CapabilityMatchingError("No agents registered")

        best_agent: str = ""
        best_score = -1.0
        best_type = "none"

        for agent in agents:
            total = 0.0
            overall_type = "exact"
            for cap in required:
                match_type, score = self._match_single(agent, cap)
                total += score
                # Track worst-case match type (most distant)
                type_rank = {"exact": 4, "child": 3, "parent": 2, "sibling": 1, "none": 0}
                if type_rank.get(match_type, 0) < type_rank.get(overall_type, 0):
                    overall_type = match_type

            avg = total / len(required)
            if avg > best_score:
                best_score = avg
                best_agent = agent
                best_type = overall_type

        return MatchResult(
            agent=best_agent,
            capability=", ".join(required),
            score=best_score,
            match_type=best_type,
        )

    def _match_single(self, agent: str, capability: str) -> Tuple[str, float]:
        """Score a single agent against a single capability."""
        try:
            agent_caps = self.registry.get_capabilities(agent)
        except Exception:
            return "none", 0.0

        set_caps = set(agent_caps)

        # Exact match
        if capability in set_caps:
            return "exact", self.SCORING["exact"]

        # Taxonomic matches
        if self.taxonomy:
            # Child match
            children = self.taxonomy.get_children(capability)
            if any(c in set_caps for c in children):
                return "child", self.SCORING["child"]

            # Parent match
            parents = self.taxonomy.get_parents(capability)
            if any(p in set_caps for p in parents):
                return "parent", self.SCORING["parent"]

            # Sibling match
            siblings = self.taxonomy.get_siblings(capability)
            if any(s in set_caps for s in siblings):
                return "sibling", self.SCORING["sibling"]

        return "none", self.SCORING["none"]
