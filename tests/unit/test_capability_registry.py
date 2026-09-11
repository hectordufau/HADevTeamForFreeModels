# Tests for CapabilityRegistry
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.capabilities.registry import CapabilityRegistry, CapabilityRegistryError


def test_register_and_get_capabilities():
    registry = CapabilityRegistry()
    registry.register_agent("coder", ["backend_development", "api_implementation", "testing"])
    caps = registry.get_capabilities("coder")
    assert sorted(caps) == sorted(["backend_development", "api_implementation", "testing"])


def test_find_agents_for_exact_capability():
    registry = CapabilityRegistry()
    registry.register_agent("coder", ["backend_development", "api_implementation"])
    registry.register_agent("tester", ["testing", "unit_testing"])
    agents = registry.find_agents_for("testing")
    assert agents == ["tester"]


def test_find_best_agent_exact_match():
    registry = CapabilityRegistry()
    registry.register_agent("coder", ["backend_development", "api_implementation"])
    registry.register_agent("tester", ["testing", "unit_testing"])
    agent, score = registry.find_best_agent(["testing"])
    assert agent == "tester"
    assert score == 1.0


def test_find_best_agent_no_match():
    registry = CapabilityRegistry()
    registry.register_agent("coder", ["backend_development"])
    agent, score = registry.find_best_agent(["security_analysis"])
    # Returns first agent even with 0 score (best_score starts at -1.0)
    # But 0.0 score means no match
    assert score == 0.0


def test_find_best_agent_parent_match_with_taxonomy():
    from harness.capabilities.taxonomy import CapabilityTaxonomy
    taxonomy = CapabilityTaxonomy()
    # Manually set up the index
    taxonomy._all_capabilities = {"testing", "unit_testing"}
    taxonomy._children = {"testing": ["unit_testing"]}
    taxonomy._parents = {"unit_testing": ["testing"]}

    registry = CapabilityRegistry()
    registry.register_agent("tester", ["unit_testing"])  # child of testing
    agent, score = registry.find_best_agent(["testing"], taxonomy=taxonomy)
    assert agent == "tester"
    assert score == 0.9  # child match


def test_find_best_agent_sibling_match():
    from harness.capabilities.taxonomy import CapabilityTaxonomy
    taxonomy = CapabilityTaxonomy()
    taxonomy._all_capabilities = {"unit_testing", "integration_testing"}
    taxonomy._children = {"testing": ["unit_testing", "integration_testing"]}
    taxonomy._parents = {"unit_testing": ["testing"], "integration_testing": ["testing"]}
    registry = CapabilityRegistry()
    registry.register_agent("tester", ["integration_testing"])
    agent, score = registry.find_best_agent(["unit_testing"], taxonomy=taxonomy)
    # Sibling match via taxonomy
    assert score == 0.4


def test_scoring_custom_weights():
    registry = CapabilityRegistry()
    registry.register_agent("coder", ["backend_development"])
    registry.register_agent("tester", ["testing"])
    agent, score = registry.find_best_agent(
        ["backend_development"],
        scoring={"exact": 2.0, "child": 1.0, "parent": 0.5, "sibling": 0.3}
    )
    assert agent == "coder"
    assert score == 2.0


def test_list_agents():
    registry = CapabilityRegistry()
    registry.register_agent("coder", ["coding"])
    registry.register_agent("reviewer", ["code_review"])
    assert registry.list_agents() == ["coder", "reviewer"]


def test_unregister_agent():
    registry = CapabilityRegistry()
    registry.register_agent("coder", ["coding"])
    registry.unregister_agent("coder")
    assert registry.list_agents() == []
    assert registry.find_agents_for("coding") == []


def test_score_agent_for_capabilities():
    registry = CapabilityRegistry()
    registry.register_agent("coder", ["backend_development", "api_implementation"])
    score = registry.score_agent_for_capabilities("coder", ["backend_development", "testing"])
    assert score == 0.5  # 1.0 for exact match, 0.0 for no match, / 2


def test_empty_registry_returns_none():
    registry = CapabilityRegistry()
    agent, score = registry.find_best_agent(["testing"])
    # Empty registry returns None (no agents to iterate)
    assert agent is None
    assert score == 0.0


def test_register_empty_role_raises():
    registry = CapabilityRegistry()
    try:
        registry.register_agent("", ["coding"])
        assert False, "Should have raised"
    except CapabilityRegistryError:
        pass


if __name__ == "__main__":
    test_register_and_get_capabilities()
    test_find_agents_for_exact_capability()
    test_find_best_agent_exact_match()
    test_find_best_agent_no_match()
    test_find_best_agent_parent_match_with_taxonomy()
    test_find_best_agent_sibling_match()
    test_scoring_custom_weights()
    test_list_agents()
    test_unregister_agent()
    test_score_agent_for_capabilities()
    test_empty_registry_returns_none()
    test_register_empty_role_raises()
    print("All capability_registry tests passed!")
