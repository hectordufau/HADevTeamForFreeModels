# harness/capabilities/intelligence.py — Capability Intelligence (Phase G)
"""
Capability gap detection, recommendation, synonym resolution, and quality metrics.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


class CapabilityIntelligenceError(Exception):
    """Raised on capability intelligence errors."""


@dataclass
class CapabilityGap:
    """A detected gap in capability coverage."""
    capability: str
    gap_type: str  # missing, insufficient_coverage, low_quality
    severity: str  # critical, major, minor
    description: str
    affected_scenarios: List[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class CapabilityRecommendation:
    """A recommendation for capability improvement."""
    capability: str
    recommended_action: str
    priority: int  # 1 (highest) to 5 (lowest)
    rationale: str
    expected_impact: str


@dataclass
class CapabilityQualityMetrics:
    """Quality metrics for a capability."""
    capability: str
    usage_count: int
    success_rate: float
    avg_score: float
    agent_count: int
    scenario_count: int
    quality_score: float  # 0.0 to 1.0
    last_used: str = ""


class SynonymResolver:
    """Resolves capability synonyms and near-matches."""

    # Built-in synonym map
    SYNONYMS: Dict[str, List[str]] = {
        "backend_development": ["backend", "server_side", "api_implementation"],
        "frontend_development": ["frontend", "client_side", "ui_development"],
        "testing": ["software_testing", "qa", "quality_assurance"],
        "security_analysis": ["security", "application_security"],
        "refactoring": ["code_refactoring", "restructuring"],
        "code_review": ["review", "peer_review", "code_audit"],
        "system_design": ["architecture", "software_architecture"],
        "domain_modeling": ["domain_driven_design", "modeling"],
        "deployment": ["deploy", "release_management"],
        "ci_cd": ["continuous_integration", "continuous_delivery"],
    }

    @classmethod
    def resolve(cls, capability: str) -> List[str]:
        """Resolve a capability to its canonical form and synonyms."""
        # Check if it's a synonym of a known capability
        for canonical, synonyms in cls.SYNONYMS.items():
            if capability == canonical or capability in synonyms:
                return [canonical] + [s for s in synonyms if s != capability]
        return [capability]

    @classmethod
    def is_synonym(cls, cap_a: str, cap_b: str) -> bool:
        """Check if two capabilities are synonyms."""
        return cap_b in cls.resolve(cap_a) or cap_a in cls.resolve(cap_b)

    @classmethod
    def canonical(cls, capability: str) -> str:
        """Get the canonical form of a capability."""
        for canonical, synonyms in cls.SYNONYMS.items():
            if capability == canonical or capability in synonyms:
                return canonical
        return capability


class CapabilityGapDetector:
    """Detects gaps in capability coverage across agents and scenarios."""

    def __init__(self, registry: Any, taxonomy: Any):
        self.registry = registry
        self.taxonomy = taxonomy

    def detect_gaps(self, required_capabilities: List[str],
                    scenario_tags: Optional[List[str]] = None) -> List[CapabilityGap]:
        """Detect gaps between required capabilities and agent registry."""
        gaps = []
        all_agents = self.registry.list_agents() if self.registry else []
        all_caps = set()
        for agent in all_agents:
            try:
                all_caps.update(self.registry.get_capabilities(agent))
            except Exception:
                continue

        for cap in required_capabilities:
            if cap not in all_caps:
                # Check synonyms
                syns = SynonymResolver.resolve(cap)
                if not any(s in all_caps for s in syns):
                    # Check taxonomy hierarchy
                    if self.taxonomy:
                        children = self.taxonomy.get_children(cap)
                        parents = self.taxonomy.get_parents(cap)
                        if not any(c in all_caps for c in children + parents):
                            gaps.append(CapabilityGap(
                                capability=cap,
                                gap_type="missing",
                                severity="critical",
                                description=f"No agent declares capability '{cap}' or its synonyms",
                                affected_scenarios=scenario_tags or [],
                                recommendation=f"Register an agent that declares '{cap}' or add it to an existing agent's profile",
                            ))
                        else:
                            gaps.append(CapabilityGap(
                                capability=cap,
                                gap_type="insufficient_coverage",
                                severity="major",
                                description=f"Capability '{cap}' has only hierarchical (not exact) agent coverage",
                                affected_scenarios=scenario_tags or [],
                                recommendation=f"Add '{cap}' as an explicit capability to the relevant agent",
                            ))
                    else:
                        gaps.append(CapabilityGap(
                            capability=cap,
                            gap_type="missing",
                            severity="critical",
                            description=f"No agent declares capability '{cap}'",
                            affected_scenarios=scenario_tags or [],
                            recommendation=f"Register a new agent with '{cap}' capability",
                        ))

        return gaps

    def get_coverage_report(self) -> Dict[str, Any]:
        """Get a full coverage report of all capabilities."""
        all_agents = self.registry.list_agents() if self.registry else []
        covered = set()
        for agent in all_agents:
            try:
                covered.update(self.registry.get_capabilities(agent))
            except Exception:
                continue

        uncovered = set()
        if self.taxonomy:
            all_caps = self.taxonomy.get_all_capabilities()
            uncovered = all_caps - covered

        return {
            "total_capabilities": len(covered) + len(uncovered),
            "covered": len(covered),
            "uncovered": len(uncovered),
            "coverage_pct": len(covered) / max(len(covered) + len(uncovered), 1) * 100,
            "agents": len(all_agents),
        }


class CapabilityRecommender:
    """Recommends capabilities based on task context and historical data."""

    def __init__(self, registry: Any, taxonomy: Any,
                 experience_store: Any = None):
        self.registry = registry
        self.taxonomy = taxonomy
        self.experience_store = experience_store

    def recommend(self, task_objective: str,
                  acceptances: List[str],
                  limit: int = 5) -> List[CapabilityRecommendation]:
        """Recommend capabilities for a given task context."""
        recommendations = []

        # Extract keywords from objective
        keywords = set(task_objective.lower().split())
        for ac in acceptances:
            keywords.update(ac.lower().split())

        # Match keywords to capabilities via taxonomy
        matched_caps = set()
        if self.taxonomy:
            for cap in self.taxonomy.get_all_capabilities():
                if any(kw in cap.lower() for kw in keywords) or \
                   any(kw in cap.lower().replace("_", " ") for kw in keywords):
                    matched_caps.add(cap)

        # Score each matched capability
        for cap in matched_caps:
            agents = self.registry.find_agents_for(cap) if self.registry else []
            score = len(agents) / max(len(self.registry.list_agents()), 1) if agents else 0

            recommendations.append(CapabilityRecommendation(
                capability=cap,
                recommended_action="include",
                priority=max(1, int(5 - score * 4)),
                rationale=f"Capability '{cap}' matched keywords in task objective",
                expected_impact="high" if agents else "medium",
            ))

        recommendations.sort(key=lambda r: r.priority)
        return recommendations[:limit]


class CapabilityQualityAnalyzer:
    """Analyzes quality metrics for capabilities based on usage history."""

    def __init__(self, registry: Any, experience_store: Any = None):
        self.registry = registry
        self.experience_store = experience_store

    def analyze(self, capability: str) -> CapabilityQualityMetrics:
        """Analyze quality metrics for a single capability."""
        agents = self.registry.find_agents_for(capability) if self.registry else []

        # Query experience store if available
        usage_count = 0
        success_count = 0
        total_score = 0.0
        scenario_count = 0

        if self.experience_store:
            experiences = self.experience_store.search(capability, min_relevance=0.0)
            usage_count = len(experiences)
            scenario_count = len(set(e.entry_id for e in experiences))

            for exp in experiences:
                if exp.relevance > 0.5:
                    success_count += 1
                total_score += exp.confidence

        success_rate = success_count / max(usage_count, 1)
        avg_score = total_score / max(usage_count, 1)

        # Composite quality score
        quality = (
            min(1.0, usage_count / 10) * 0.3 +
            success_rate * 0.4 +
            min(1.0, len(agents) / 3) * 0.3
        )

        return CapabilityQualityMetrics(
            capability=capability,
            usage_count=usage_count,
            success_rate=success_rate,
            avg_score=avg_score,
            agent_count=len(agents),
            scenario_count=scenario_count,
            quality_score=round(quality, 3),
        )
