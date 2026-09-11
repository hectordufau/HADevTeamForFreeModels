# harness/planner/task_intelligence.py — Task Intelligence V3 (Phase H)
"""
LLM-assisted TaskAnalyzer, task analysis validation, and task classification.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


class TaskIntelligenceError(Exception):
    """Raised on task intelligence errors."""


@dataclass
class TaskClassification:
    """Classification result for a task."""
    task_type: str  # feature, bugfix, refactor, documentation, security, performance, other
    complexity_level: str  # low, medium, high, critical
    estimated_effort: str  # minutes, hours, days
    required_capabilities: List[str] = field(default_factory=list)
    reasoning: str = ""


class LLMTaskAnalyzer:
    """LLM-assisted task analyzer that uses the LLM to propose capabilities.

    The LLM proposes capabilities; the Harness validates against taxonomy and policy.
    Falls back to keyword-based analysis if LLM is unavailable.
    """

    def __init__(self, taxonomy: Any, llm_client: Any = None):
        self.taxonomy = taxonomy
        self.llm_client = llm_client

    def analyze(self, task: Any) -> Dict[str, Any]:
        """Analyze a task using LLM assistance when available."""
        objective = self._get_objective(task)
        requirements = self._get_requirements(task)
        acceptances = self._get_acceptance_criteria(task)

        # Try LLM analysis
        if self.llm_client:
            try:
                return self._llm_analyze(objective, requirements, acceptances)
            except Exception:
                pass

        # Fallback to keyword analysis
        return self._keyword_analyze(objective, requirements, acceptances)

    def _llm_analyze(self, objective: str, requirements: List[str],
                     acceptances: List[str]) -> Dict[str, Any]:
        """Use LLM to propose capabilities."""
        # Structure the prompt
        prompt = self._build_prompt(objective, requirements, acceptances)

        # Call LLM
        response = self.llm_client.chat(prompt)

        # Parse response
        return self._parse_llm_response(response)

    def _keyword_analyze(self, objective: str, requirements: List[str],
                         acceptances: List[str]) -> Dict[str, Any]:
        """Fallback keyword-based analysis."""
        from harness.planner.task_analyzer import TaskAnalyzer as BaseAnalyzer
        base = BaseAnalyzer(taxonomy=self.taxonomy)
        result = base._extract_capabilities(objective, requirements)
        return {"required": list(result)}

    def _build_prompt(self, objective: str, requirements: List[str],
                      acceptances: List[str]) -> str:
        """Build an LLM prompt for task analysis."""
        caps = ""
        if self.taxonomy:
            caps = ", ".join(sorted(self.taxonomy.get_all_capabilities()))

        return (
            f"Analyze this software engineering task and determine the required capabilities.\n\n"
            f"Objective: {objective}\n"
            f"Requirements: {'; '.join(requirements)}\n"
            f"Acceptance Criteria: {'; '.join(acceptances)}\n\n"
            f"Available capabilities: {caps}\n\n"
            f"Return a JSON object with:\n"
            f"- required: list of required capability names\n"
            f"- optional: list of optional capability names\n"
            f"Respond with JSON only."
        )

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract capability proposal."""
        import json
        import re

        # Try JSON extraction
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except (json.JSONDecodeError, ValueError):
                pass

        return {"required": [], "optional": []}

    def _get_objective(self, task: Any) -> str:
        if hasattr(task, 'spec') and hasattr(task.spec, 'objective'):
            return task.spec.objective
        if isinstance(task, dict):
            return task.get('spec', {}).get('objective', '')
        return str(task)

    def _get_requirements(self, task: Any) -> List[str]:
        if hasattr(task, 'spec') and hasattr(task.spec, 'requirements'):
            return task.spec.requirements
        if isinstance(task, dict):
            return task.get('spec', {}).get('requirements', [])
        return []

    def _get_acceptance_criteria(self, task: Any) -> List[str]:
        if hasattr(task, 'spec') and hasattr(task.spec, 'acceptance_criteria'):
            return task.spec.acceptance_criteria
        if isinstance(task, dict):
            return task.get('spec', {}).get('acceptance_criteria', [])
        return []


class TaskValidator:
    """Validates task analysis proposals against taxonomy and policy."""

    def __init__(self, taxonomy: Any):
        self.taxonomy = taxonomy

    def validate(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a capability proposal.

        Returns validated proposal with unknown capabilities removed.
        """
        valid_caps = set()
        if self.taxonomy:
            valid_caps = self.taxonomy.get_all_capabilities()

        required = [c for c in proposal.get("required", []) if c in valid_caps]
        optional = [c for c in proposal.get("optional", []) if c in valid_caps]

        return {
            "required": required,
            "optional": optional,
            "removed": [c for c in proposal.get("required", []) + proposal.get("optional", [])
                       if c not in valid_caps],
        }


class TaskClassifier:
    """Classifies tasks by type, complexity, and effort."""

    CLASSIFICATION_KEYWORDS: Dict[str, List[str]] = {
        "feature": ["implement", "add", "create", "new", "feature", "build"],
        "bugfix": ["fix", "bug", "error", "issue", "broken", "incorrect"],
        "refactor": ["refactor", "clean", "restructure", "improve", "optimize"],
        "documentation": ["document", "docs", "readme", "wiki", "comment"],
        "security": ["security", "vulnerability", "protect", "auth"],
        "performance": ["performance", "speed", "slow", "latency", "optimize"],
    }

    COMPLEXITY_KEYWORDS: Dict[str, List[str]] = {
        "low": ["simple", "basic", "minor", "trivial", "cosmetic"],
        "medium": ["moderate", "standard", "normal"],
        "high": ["complex", "difficult", "challenging", "major"],
        "critical": ["critical", "urgent", "blocker", "system-wide"],
    }

    def classify(self, task: Any) -> TaskClassification:
        """Classify a task by type, complexity, and effort."""
        objective = self._get_text(task)

        task_type = self._classify_type(objective)
        complexity = self._classify_complexity(objective)
        effort = self._estimate_effort(complexity)

        caps = self._infer_capabilities(task_type, objective)

        return TaskClassification(
            task_type=task_type,
            complexity_level=complexity,
            estimated_effort=effort,
            required_capabilities=caps,
            reasoning=f"Classified as {task_type}/{complexity} based on keyword analysis",
        )

    def _classify_type(self, text: str) -> str:
        """Classify task type by keyword matching."""
        text_lower = text.lower()
        best_type = "other"
        best_score = 0
        for task_type, keywords in self.CLASSIFICATION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_type = task_type
        return best_type

    def _classify_complexity(self, text: str) -> str:
        """Classify complexity by keyword matching."""
        text_lower = text.lower()
        best_level = "medium"
        best_score = 0
        for level, keywords in self.COMPLEXITY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_level = level
        return best_level

    def _estimate_effort(self, complexity: str) -> str:
        """Estimate effort based on complexity level."""
        mapping = {
            "low": "minutes",
            "medium": "hours",
            "high": "days",
            "critical": "days",
        }
        return mapping.get(complexity, "hours")

    def _infer_capabilities(self, task_type: str, objective: str) -> List[str]:
        """Infer required capabilities from task type and objective."""
        caps = []
        if task_type == "feature":
            caps.extend(["api_implementation", "testing"])
        elif task_type == "bugfix":
            caps.extend(["debugging", "testing"])
        elif task_type == "refactor":
            caps.extend(["refactoring", "testing"])
        elif task_type == "documentation":
            caps.extend(["technical_writing"])
        elif task_type == "security":
            caps.extend(["security_analysis"])
        elif task_type == "performance":
            caps.extend(["profiling", "optimization"])

        # Always add mandatory caps
        caps.extend(["verification", "evaluation", "review"])
        return list(set(caps))

    def _get_text(self, task: Any) -> str:
        """Extract text content from task for classification."""
        parts = []
        if hasattr(task, 'spec'):
            if hasattr(task.spec, 'objective'):
                parts.append(task.spec.objective)
            if hasattr(task.spec, 'requirements'):
                parts.extend(task.spec.requirements)
            if hasattr(task.spec, 'acceptance_criteria'):
                parts.extend(task.spec.acceptance_criteria)
        elif isinstance(task, dict):
            spec = task.get('spec', {})
            parts.append(spec.get('objective', ''))
        return " ".join(parts)
