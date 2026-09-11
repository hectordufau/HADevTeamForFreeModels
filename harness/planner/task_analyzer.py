# harness/planner/task_analyzer.py — TaskAnalyzer (V2.2)
"""
Analyzes a Task Contract and produces capability requirements.
The LLM proposes capabilities; the Harness validates against taxonomy and policy.
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field


class TaskAnalyzerError(Exception):
    """Raised on task analysis errors."""


@dataclass
class CapabilityRequirements:
    """Structured set of required capabilities from task analysis."""
    required: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=lambda: {"minimum_coverage": 1.0})
    agent_preferences: List[Dict[str, Any]] = field(default_factory=list)

    def validate(self) -> List[str]:
        """Validate the requirements structure."""
        errors = []
        if not self.required:
            errors.append("At least one required capability must be specified")
        if self.constraints.get("minimum_coverage", 1.0) < 0 or self.constraints.get("minimum_coverage", 1.0) > 1.0:
            errors.append("minimum_coverage must be between 0.0 and 1.0")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


# Mandatory capabilities that must always be present
MANDATORY_CAPABILITIES = {"verification", "evaluation", "review"}


class TaskAnalyzer:
    """Analyzes a task contract and produces capability requirements."""

    def __init__(self, taxonomy: Any = None):
        self.taxonomy = taxonomy

    def analyze(self, task: Any) -> CapabilityRequirements:
        """
        Analyze a TaskContract and produce capability requirements.

        Uses keyword-based extraction from the task objective and requirements,
        then validates against the taxonomy and enforces mandatory capabilities.
        """
        if not hasattr(task, 'spec') or not hasattr(task.spec, 'objective'):
            raise TaskAnalyzerError("Task must have spec.objective")

        objective = task.spec.objective.lower()
        requirements = getattr(task.spec, 'requirements', [])

        # Extract capabilities from objective keywords
        extracted = self._extract_capabilities(objective, requirements)

        # Separate required vs optional
        required, optional = self._classify_capabilities(extracted, task)

        # Enforce mandatory capabilities
        for mc in MANDATORY_CAPABILITIES:
            if mc not in required:
                required.append(mc)

        # Build agent preferences from verification requirements
        agent_prefs = self._build_agent_preferences(task)

        return CapabilityRequirements(
            required=required,
            optional=optional,
            constraints={"minimum_coverage": 1.0},
            agent_preferences=agent_prefs,
        )

    def validate_proposal(self, proposal: Dict[str, Any]) -> CapabilityRequirements:
        """
        Validate an LLM-proposed capability set against taxonomy and policy.
        """
        required = proposal.get("required", [])
        optional = proposal.get("optional", [])

        # Validate against taxonomy
        if self.taxonomy:
            valid_caps = self.taxonomy.get_all_capabilities()
            required = [c for c in required if c in valid_caps or c in MANDATORY_CAPABILITIES]
            optional = [c for c in optional if c in valid_caps]

        # Enforce mandatory capabilities
        for mc in MANDATORY_CAPABILITIES:
            if mc not in required:
                required.append(mc)

        constraints = proposal.get("constraints", {"minimum_coverage": 1.0})
        agent_prefs = proposal.get("agent_preferences", [])

        cr = CapabilityRequirements(
            required=required,
            optional=optional,
            constraints=constraints,
            agent_preferences=agent_prefs,
        )

        if not cr.is_valid():
            raise TaskAnalyzerError(f"Invalid proposal: {cr.validate()}")

        return cr

    def _extract_capabilities(self, objective: str,
                              requirements: List[str]) -> Set[str]:
        """Extract capability keywords from task objective and requirements."""
        extracted = set()

        # Keyword -> capability mappings
        keyword_map = {
            "api": "api_design",
            "endpoint": "api_implementation",
            "backend": "backend_development",
            "frontend": "frontend_development",
            "ui": "ui_implementation",
            "database": "database_schema",
            "schema": "database_schema",
            "test": "testing",
            "unit test": "unit_testing",
            "integration": "integration_testing",
            "e2e": "e2e_testing",
            "security": "security_analysis",
            "vulnerability": "vulnerability_assessment",
            "performance": "profiling",
            "optimize": "optimization",
            "document": "technical_writing",
            "documentation": "technical_writing",
            "deploy": "deployment",
            "ci": "ci_cd",
            "cd": "ci_cd",
            "refactor": "refactoring",
            "architecture": "system_design",
            "design": "system_design",
            "domain": "domain_modeling",
            "integration": "integration_design",
        }

        # Check objective
        for keyword, capability in keyword_map.items():
            if keyword in objective:
                extracted.add(capability)

        # Check requirements
        for req in requirements:
            req_lower = req.lower()
            for keyword, capability in keyword_map.items():
                if keyword in req_lower:
                    extracted.add(capability)

        return extracted

    def _classify_capabilities(self, extracted: Set[str],
                                task: Any) -> tuple:
        """Separate extracted capabilities into required vs optional."""
        required = []
        optional = []

        # Acceptance criteria items hint at required capabilities
        ac_text = " ".join(getattr(task.spec, 'acceptance_criteria', []))
        check_required = {"test", "verif", "valid", "deploy", "security", "review"}

        for cap in extracted:
            if any(keyword in cap.lower() for keyword in check_required):
                required.append(cap)
            else:
                optional.append(cap)

        # If no capabilities extracted, add sensible defaults
        if not extracted:
            required = ["backend_development", "testing"]
            optional = ["security_analysis"]

        return required, optional

    def _build_agent_preferences(self, task: Any) -> List[Dict[str, Any]]:
        """Build agent preferences from task verification requirements."""
        prefs = []
        verif = getattr(task.spec, 'verification', None)
        if verif and hasattr(verif, 'required'):
            for check_type in verif.required:
                cap_map = {
                    "unit_tests": "unit_testing",
                    "integration": "integration_testing",
                    "lint": "testing",
                    "build": "backend_development",
                    "security": "security_analysis",
                }
                capability = cap_map.get(check_type)
                if capability:
                    prefs.append({
                        "capability": capability,
                        "required": True,
                    })
        return prefs
