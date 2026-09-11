"""Tests for Policy Engine."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from harness.policy import PolicyEngine, AutonomyPolicy, ToolPolicy, ExecutionPolicy


def test_autonomy_policy_most_restrictive():
    policy = AutonomyPolicy()
    result = policy.check(agent_level=3, task_level=2, tool_level=1)
    assert result.effective_autonomy == 1
    assert result.allowed


def test_autonomy_policy_invalid_level():
    policy = AutonomyPolicy()
    result = policy.check(agent_level=6, task_level=2, tool_level=1)
    assert not result.allowed


def test_tool_policy_blocked():
    policy = ToolPolicy()
    result = policy.check("deploy", current_autonomy=3)
    assert not result.allowed
    assert "requires autonomy 5" in result.violations[0]["reason"]


def test_tool_policy_allowed():
    policy = ToolPolicy()
    result = policy.check("filesystem_read", current_autonomy=1)
    assert result.allowed


def test_execution_policy_max_iterations():
    policy = ExecutionPolicy(max_iterations=3)
    result = policy.check(current_iteration=4)
    assert not result.allowed
    assert "Exceeded max iterations" in result.violations[0]["reason"]


def test_policy_engine_compose():
    engine = PolicyEngine()
    result = engine.evaluate_all(
        agent_level=2, task_level=3, tool_level=5,
        tool_name="deploy", current_autonomy=2,
        current_iteration=1, has_verification=True,
    )
    assert not result.allowed
    # deploy requires 5, autonomy is min(2,3,5)=2
    assert any(v["policy"] == "tool" for v in result.violations)


if __name__ == "__main__":
    test_autonomy_policy_most_restrictive()
    test_autonomy_policy_invalid_level()
    test_tool_policy_blocked()
    test_tool_policy_allowed()
    test_execution_policy_max_iterations()
    test_policy_engine_compose()
    print("All policy tests passed!")
