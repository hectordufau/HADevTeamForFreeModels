"""Tests for Task Intelligence (Phase H)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.planner.task_intelligence import (
    LLMTaskAnalyzer, TaskValidator, TaskClassifier, TaskClassification,
    TaskIntelligenceError,
)
from harness.capabilities.taxonomy import CapabilityTaxonomy


class MockTask:
    def __init__(self, objective="", requirements=None, acceptances=None):
        self.spec = type('obj', (object,), {
            'objective': objective,
            'requirements': requirements or [],
            'acceptance_criteria': acceptances or [],
        })


def test_task_classification():
    classifier = TaskClassifier()

    # Feature
    result = classifier.classify(MockTask(
        objective="Implement a new user dashboard feature",
        acceptances=["dashboard renders correctly"],
    ))
    assert result.task_type == "feature"
    assert result.complexity_level in ("low", "medium", "high", "critical")

    # Bugfix
    result2 = classifier.classify(MockTask(
        objective="Fix the login bug where sessions expire early",
        acceptances=["login works correctly"],
    ))
    assert result2.task_type == "bugfix"

    # Documentation
    result3 = classifier.classify(MockTask(
        objective="Document the REST API endpoints",
        acceptances=["all endpoints documented"],
    ))
    assert result3.task_type == "documentation"


def test_task_classification_complexity():
    classifier = TaskClassifier()
    low = classifier.classify(MockTask(objective="Fix minor typo in readme"))
    high = classifier.classify(MockTask(objective="Critical security vulnerability in auth system"))

    # Complexity levels differ
    assert low.complexity_level != high.complexity_level


def test_task_classification_mandatory_caps():
    classifier = TaskClassifier()
    result = classifier.classify(MockTask(objective="Implement a feature"))
    assert "verification" in result.required_capabilities
    assert "evaluation" in result.required_capabilities
    assert "review" in result.required_capabilities


def test_task_validator():
    taxo = CapabilityTaxonomy(taxonomy_path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config", "capability_taxonomy.yaml"
    ))
    validator = TaskValidator(taxo)

    proposal = {
        "required": ["backend_development", "nonexistent_cap"],
        "optional": ["testing"],
    }
    validated = validator.validate(proposal)
    assert "backend_development" in validated["required"]
    assert "nonexistent_cap" not in validated["required"]
    assert len(validated["removed"]) == 1


def test_llm_task_analyzer_fallback():
    """Test that LLM task analyzer falls back to keyword analysis."""
    taxo = CapabilityTaxonomy(taxonomy_path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config", "capability_taxonomy.yaml"
    ))
    analyzer = LLMTaskAnalyzer(taxonomy=taxo, llm_client=None)

    result = analyzer.analyze(MockTask(
        objective="Implement a REST API with database integration",
        requirements=["Create endpoints", "Add schema"],
    ))
    assert "required" in result
    assert len(result.get("required", [])) > 0
