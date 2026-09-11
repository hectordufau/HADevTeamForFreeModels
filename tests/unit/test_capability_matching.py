# Tests for CapabilityMatching
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.capabilities.matching import CapabilityMatcher, MatchResult, CapabilityMatchingError
from harness.capabilities.registry import CapabilityRegistry
from harness.capabilities.taxonomy import CapabilityTaxonomy


def _setup_taxonomy():
    taxonomy = CapabilityTaxonomy()
    taxonomy._all_capabilities = {
        "testing", "unit_testing", "integration_testing", "e2e_testing",
        "backend_development", "api_implementation",
        "security_analysis",
    }
    taxonomy._children = {
        "testing": ["unit_testing", "integration_testing", "e2e_testing"],
    }
    taxonomy._parents = {
        "unit_testing": ["testing"],
        "integration_testing": ["testing"],
        "e2e_testing": ["testing"],
    }
    return taxonomy


def test_exact_match():
    taxonomy = _setup_taxonomy()
    registry = CapabilityRegistry()
    registry.register_agent("tester", ["testing", "unit_testing"])

    matcher = CapabilityMatcher(registry, taxonomy)
    result = matcher.best_match(["testing"])
    assert result.agent == "tester"
    assert result.score == 1.0
    assert result.match_type == "exact"


def test_child_match():
    taxonomy = _setup_taxonomy()
    registry = CapabilityRegistry()
    registry.register_agent("tester", ["unit_testing"])  # child of testing

    matcher = CapabilityMatcher(registry, taxonomy)
    result = matcher.best_match(["testing"])
    assert result.agent == "tester"
    assert result.score == 0.9
    assert result.match_type == "child"


def test_parent_match():
    taxonomy = _setup_taxonomy()
    registry = CapabilityRegistry()
    registry.register_agent("tester", ["testing"])  # parent of unit_testing

    matcher = CapabilityMatcher(registry, taxonomy)
    result = matcher.best_match(["unit_testing"])
    assert result.agent == "tester"
    assert result.score == 0.7
    assert result.match_type == "parent"


def test_no_match():
    taxonomy = _setup_taxonomy()
    registry = CapabilityRegistry()
    registry.register_agent("coder", ["backend_development"])

    matcher = CapabilityMatcher(registry, taxonomy)
    result = matcher.best_match(["testing"])
    assert result.score == 0.0
    assert result.match_type == "none"


def test_aggregate_score():
    taxonomy = _setup_taxonomy()
    registry = CapabilityRegistry()
    registry.register_agent("tester", ["testing", "unit_testing"])
    registry.register_agent("coder", ["backend_development"])

    matcher = CapabilityMatcher(registry, taxonomy)
    result = matcher.best_match(["testing", "backend_development"])
    # Both have same aggregate score (0.5), tie-break goes to first iterated
    assert result.score == 0.5


def test_match_returns_list():
    taxonomy = _setup_taxonomy()
    registry = CapabilityRegistry()
    registry.register_agent("tester", ["testing"])
    registry.register_agent("coder", ["backend_development"])

    matcher = CapabilityMatcher(registry, taxonomy)
    results = matcher.match(["testing", "backend_development"])
    assert len(results) == 4  # 2 agents * 2 capabilities
    assert all(isinstance(r, MatchResult) for r in results)


def test_best_match_empty_required_raises():
    matcher = CapabilityMatcher(CapabilityRegistry(), _setup_taxonomy())
    try:
        matcher.best_match([])
        assert False, "Should have raised"
    except CapabilityMatchingError:
        pass


def test_best_match_no_agents_raises():
    matcher = CapabilityMatcher(CapabilityRegistry(), _setup_taxonomy())
    try:
        matcher.best_match(["testing"])
        assert False, "Should have raised"
    except CapabilityMatchingError:
        pass


def test_match_capability_not_in_registry():
    taxonomy = _setup_taxonomy()
    registry = CapabilityRegistry()
    registry.register_agent("coder", ["unknown_cap"])

    matcher = CapabilityMatcher(registry, taxonomy)
    result = matcher.best_match(["testing"])
    assert result.score == 0.0


def test_match_without_taxonomy():
    registry = CapabilityRegistry()
    registry.register_agent("coder", ["backend_development"])

    # Create matcher with None taxonomy
    matcher = CapabilityMatcher(registry, None)
    result = matcher.best_match(["backend_development"])
    assert result.score == 1.0
    assert result.match_type == "exact"


if __name__ == "__main__":
    test_exact_match()
    test_child_match()
    test_parent_match()
    test_no_match()
    test_aggregate_score()
    test_match_returns_list()
    test_best_match_empty_required_raises()
    test_best_match_no_agents_raises()
    test_match_capability_not_in_registry()
    test_match_without_taxonomy()
    print("All capability_matching tests passed!")
