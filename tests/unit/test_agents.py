# Tests for harness/agents
import sys
import os
import tempfile
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.agents import AgentLoader, AgentError


def test_discover_agents():
    agents_dir = os.path.join(os.path.dirname(__file__), "..", "..", "agents")
    loader = AgentLoader(agents_dir)
    roles = loader.list_roles()
    assert "manager" in roles
    assert "architect" in roles
    assert "coder" in roles


def test_load_manager():
    agents_dir = os.path.join(os.path.dirname(__file__), "..", "..", "agents")
    loader = AgentLoader(agents_dir)
    agent = loader.load("manager")
    assert agent.role == "manager"
    assert agent.profile.name == "FreeRouter"
    assert "capabilities" in agent.profile.__dict__
    assert "manager" in agent.soul_content


def test_load_missing_agent():
    agents_dir = os.path.join(os.path.dirname(__file__), "..", "..", "agents")
    loader = AgentLoader(agents_dir)
    try:
        loader.load("nonexistent")
        assert False, "Should have raised AgentError"
    except AgentError:
        pass


if __name__ == "__main__":
    test_discover_agents()
    test_load_manager()
    test_load_missing_agent()
    print("All agent tests passed!")
