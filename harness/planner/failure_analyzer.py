# harness/planner/failure_analyzer.py — FailureAnalyzer (V2.2)
"""
Classifies execution failures and identifies capability gaps.
Maps error patterns to failure categories and recommends missing capabilities.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


class FailureAnalyzerError(Exception):
    """Raised on failure analysis errors."""


@dataclass
class FailureReport:
    """Detailed failure analysis result."""
    node_id: str
    capability: str
    category: str  # implementation, test, architecture, security, integration, configuration, environment, unknown
    message: str
    missing_capabilities: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    evidence_summary: Optional[str] = None


class FailureAnalyzer:
    """Classifies failures and identifies capability gaps.

    Categories:
        - implementation: Logic error, API misuse
        - test: Test assertion failed
        - architecture: Wrong pattern, coupling issue
        - security: Vulnerability detected
        - integration: API contract mismatch
        - configuration: Missing env var, wrong setting
        - environment: Dependency missing, version conflict
        - unknown: Cannot be classified
    """

    # Error pattern -> category mapping
    ERROR_PATTERNS: Dict[str, str] = {
        # Implementation errors
        "typeerror": "implementation",
        "attributeerror": "implementation",
        "valueerror": "implementation",
        "keyerror": "implementation",
        "indexerror": "implementation",
        "nameerror": "implementation",
        "syntaxerror": "implementation",
        "indentationerror": "implementation",
        "importerror": "implementation",
        "modulenotfounderror": "implementation",
        "zerodivisionerror": "implementation",
        "assertionerror": "implementation",
        "recursionerror": "implementation",
        "stopiteration": "implementation",
        "runtimeerror": "implementation",
        "logical error": "implementation",
        "logic error": "implementation",
        "api misuse": "implementation",
        "invalid syntax": "implementation",

        # Test errors
        "assertionerror": "test",
        "assertion failed": "test",
        "assertionfailed": "test",
        "test failed": "test",
        "test assertion": "test",
        "test failure": "test",
        "test case": "test",

        # Architecture errors
        "circular dependency": "architecture",
        "wrong pattern": "architecture",
        "coupling": "architecture",
        "tight coupling": "architecture",
        "design flaw": "architecture",
        "architecture violation": "architecture",
        "layering violation": "architecture",

        # Security errors
        "vulnerability": "security",
        "injection": "security",
        "xss": "security",
        "csrf": "security",
        "sqli": "security",
        "sql injection": "security",
        "security issue": "security",
        "unauthorized": "security",
        "permission denied": "security",
        "access denied": "security",
        "authentication": "security",

        # Integration errors
        "connection refused": "integration",
        "timeout": "integration",
        "contract mismatch": "integration",
        "api contract": "integration",
        "protocol error": "integration",
        "handshake failed": "integration",

        # Configuration errors
        "missing env": "configuration",
        "environment variable": "configuration",
        "misconfigured": "configuration",
        "configuration error": "configuration",
        "invalid config": "configuration",
        "wrong setting": "configuration",
        "not configured": "configuration",

        # Environment errors
        "dependency missing": "environment",
        "version conflict": "environment",
        "incompatible": "environment",
        "not found": "environment",  # file/command not found
        "no such file": "environment",
        "command not found": "environment",
        "exec format error": "environment",
        "out of memory": "environment",
        "disk full": "environment",
        "connection reset": "environment",
    }

    # Category -> recommended missing capabilities
    CATEGORY_RECOMMENDATIONS: Dict[str, Dict] = {
        "implementation": {
            "missing_capabilities": ["refactoring", "debugging"],
            "recommendations": [
                "Add debugging capability to handle logic errors",
                "Consider adding refactoring capability for code quality",
            ],
        },
        "test": {
            "missing_capabilities": ["unit_testing", "integration_testing"],
            "recommendations": [
                "Add unit testing capability to catch test failures early",
                "Consider integration testing for contract validation",
            ],
        },
        "architecture": {
            "missing_capabilities": ["system_design", "domain_modeling"],
            "recommendations": [
                "Add system design capability for architectural decisions",
                "Consider domain modeling to reduce coupling",
            ],
        },
        "security": {
            "missing_capabilities": ["security_analysis", "vulnerability_assessment"],
            "recommendations": [
                "Add security analysis capability to prevent vulnerabilities",
                "Consider vulnerability assessment for thorough security review",
            ],
        },
        "integration": {
            "missing_capabilities": ["integration_design", "integration_testing"],
            "recommendations": [
                "Add integration design capability for API contracts",
                "Consider integration testing for compatibility verification",
            ],
        },
        "configuration": {
            "missing_capabilities": ["deployment", "ci_cd"],
            "recommendations": [
                "Add deployment capability for configuration management",
                "Consider CI/CD capability for environment consistency",
            ],
        },
        "environment": {
            "missing_capabilities": ["deployment", "backend_development"],
            "recommendations": [
                "Add deployment capability for environment management",
                "Consider backend development for dependency resolution",
            ],
        },
        "unknown": {
            "missing_capabilities": ["debugging"],
            "recommendations": [
                "Add debugging capability for general error handling",
            ],
        },
    }

    def analyze(self, node: Any, result: Any, evidence: Any = None) -> FailureReport:
        """Analyze a failure and produce a FailureReport.

        Args:
            node: The WorkflowNode that failed.
            result: The NodeExecutionResult from execution.
            evidence: Optional evidence data to inform classification.

        Returns:
            A FailureReport with root cause, missing capabilities, and recommendations.
        """
        node_id = self._get_node_id(node)
        capability = self._get_capability(node)
        error = self._get_error(result)
        evidence_text = self._get_evidence_text(evidence)

        # Classify the error
        category = self.classify(error, evidence_text)

        # Get recommendations for the category
        recs = self.CATEGORY_RECOMMENDATIONS.get(category, self.CATEGORY_RECOMMENDATIONS["unknown"])
        missing_caps = list(recs["missing_capabilities"])
        recommendations = list(recs["recommendations"])

        # Add capability-specific recommendations
        if capability:
            recommendations.insert(0, f"Failed in capability '{capability}': review implementation approach")

        return FailureReport(
            node_id=node_id,
            capability=capability,
            category=category,
            message=error or "Unknown error",
            missing_capabilities=missing_caps,
            recommendations=recommendations,
            evidence_summary=evidence_text[:500] if evidence_text else None,
        )

    def classify(self, error: str, evidence: str = "") -> str:
        """Map error text to a failure category.

        Args:
            error: The error message or traceback.
            evidence: Additional evidence text.

        Returns:
            One of: implementation, test, architecture, security, integration,
            configuration, environment, unknown.
        """
        combined = (error + " " + evidence).lower()

        # Check each pattern in priority order
        for pattern, category in self.ERROR_PATTERNS.items():
            if pattern in combined:
                return category

        return "unknown"

    def _get_node_id(self, node: Any) -> str:
        if hasattr(node, 'id'):
            return node.id
        if isinstance(node, dict):
            return node.get('id', 'unknown')
        return 'unknown'

    def _get_capability(self, node: Any) -> str:
        if hasattr(node, 'capability'):
            return node.capability
        if isinstance(node, dict):
            return node.get('capability', '')
        return ''

    def _get_error(self, result: Any) -> str:
        if hasattr(result, 'error') and result.error:
            return str(result.error)
        if hasattr(result, 'result') and isinstance(result.result, dict):
            return result.result.get('error', '') or result.result.get('reason', '')
        if isinstance(result, dict):
            return result.get('error', '') or result.get('result', {}).get('error', '')
        return ''

    def _get_evidence_text(self, evidence: Any) -> str:
        if evidence is None:
            return ""
        if hasattr(evidence, 'to_dict'):
            evidence = evidence.to_dict()
        if isinstance(evidence, dict):
            return str(evidence)
        if isinstance(evidence, str):
            return evidence
        return str(evidence)
