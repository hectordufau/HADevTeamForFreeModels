# Tests for TaskAnalyzer
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.planner.task_analyzer import TaskAnalyzer, TaskAnalyzerError, CapabilityRequirements, MANDATORY_CAPABILITIES
from harness.capabilities.taxonomy import CapabilityTaxonomy


class MockTask:
    def __init__(self, objective="", requirements=None, acceptance_criteria=None, verification=None, task_id="T-001"):
        self.metadata = type('obj', (object,), {"id": task_id})
        self.spec = type('obj', (object,), {
            "objective": objective,
            "requirements": requirements or [],
            "acceptance_criteria": acceptance_criteria or [],
            "allowed_changes": [],
            "verification": verification,
            "autonomy": type('obj', (object,), {"maximum": 3}),
        })


def test_analyze_basic():
    analyzer = TaskAnalyzer()
    task = MockTask(objective="Build an API for user management")
    reqs = analyzer.analyze(task)
    assert "api_design" in reqs.required or "api_design" in reqs.optional
    assert "verification" in reqs.required
    assert "evaluation" in reqs.required
    assert "review" in reqs.required


def test_analyze_mandatory_caps_always_present():
    analyzer = TaskAnalyzer()
    task = MockTask(objective="Fix a typo")
    reqs = analyzer.analyze(task)
    for mc in MANDATORY_CAPABILITIES:
        assert mc in reqs.required, f"Missing mandatory capability: {mc}"


def test_analyze_extracts_from_requirements():
    analyzer = TaskAnalyzer()
    task = MockTask(
        objective="Build a web app",
        requirements=["Must have database schema", "Must have unit tests"],
    )
    reqs = analyzer.analyze(task)
    assert "database_schema" in reqs.required or "database_schema" in reqs.optional
    assert "unit_testing" in reqs.required or "unit_testing" in reqs.optional


def test_analyze_with_taxonomy():
    taxonomy = CapabilityTaxonomy()
    taxonomy._all_capabilities = {"api_design", "backend_development", "testing"}
    taxonomy._children = {}
    taxonomy._parents = {}

    analyzer = TaskAnalyzer(taxonomy=taxonomy)
    task = MockTask(objective="Build API backend")
    reqs = analyzer.analyze(task)
    assert "backend_development" in reqs.required or "backend_development" in reqs.optional
    for mc in MANDATORY_CAPABILITIES:
        assert mc in reqs.required


def test_validate_proposal():
    analyzer = TaskAnalyzer()
    proposal = {
        "required": ["api_design", "backend_development"],
        "optional": ["security_analysis"],
        "constraints": {"minimum_coverage": 1.0},
        "agent_preferences": [],
    }
    reqs = analyzer.validate_proposal(proposal)
    assert "api_design" in reqs.required
    assert "backend_development" in reqs.required
    for mc in MANDATORY_CAPABILITIES:
        assert mc in reqs.required


def test_validate_proposal_removes_invalid():
    taxonomy = CapabilityTaxonomy()
    taxonomy._all_capabilities = {"api_design", "backend_development"}
    taxonomy._children = {}
    taxonomy._parents = {}

    analyzer = TaskAnalyzer(taxonomy=taxonomy)
    proposal = {
        "required": ["api_design", "nonexistent_cap", "backend_development"],
        "optional": [],
        "constraints": {"minimum_coverage": 1.0},
        "agent_preferences": [],
    }
    reqs = analyzer.validate_proposal(proposal)
    assert "api_design" in reqs.required
    assert "backend_development" in reqs.required
    assert "nonexistent_cap" not in reqs.required


def test_empty_task_raises():
    analyzer = TaskAnalyzer()
    try:
        analyzer.analyze("not a task")
        assert False, "Should have raised"
    except TaskAnalyzerError:
        pass


def test_capability_requirements_validation():
    # Empty required should fail validation
    reqs = CapabilityRequirements(required=[])
    assert reqs.is_valid() is False
    assert len(reqs.validate()) > 0

    # Valid requirements
    reqs = CapabilityRequirements(required=["testing"])
    assert reqs.is_valid() is True


def test_analyze_with_acceptance_criteria():
    analyzer = TaskAnalyzer()
    task = MockTask(
        objective="Implement login feature",
        acceptance_criteria=["Tests pass", "Security review completed"],
    )
    reqs = analyzer.analyze(task)
    assert "testing" in reqs.required or "testing" in reqs.optional
    assert "security_analysis" in reqs.required or "security_analysis" in reqs.optional


def test_analyze_with_verification():
    analyzer = TaskAnalyzer()
    verification = type('obj', (object,), {"required": ["unit_tests", "security"]})
    task = MockTask(objective="Build feature", verification=verification)
    reqs = analyzer.analyze(task)
    # Should extract preferences from verification
    assert len(reqs.agent_preferences) > 0


if __name__ == "__main__":
    test_analyze_basic()
    test_analyze_mandatory_caps_always_present()
    test_analyze_extracts_from_requirements()
    test_analyze_with_taxonomy()
    test_validate_proposal()
    test_validate_proposal_removes_invalid()
    test_empty_task_raises()
    test_capability_requirements_validation()
    test_analyze_with_acceptance_criteria()
    test_analyze_with_verification()
    print("All task_analyzer tests passed!")
