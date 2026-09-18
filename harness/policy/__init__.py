"""
Policy Engine for V2.1: composable policy evaluation for autonomy, tools, and execution.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


class PolicyViolation(Exception):
    """Raised when a policy check fails."""
    def __init__(self, policy: str, reason: str):
        self.policy = policy
        self.reason = reason
        super().__init__(f"[{policy}] {reason}")


@dataclass
class PolicyResult:
    allowed: bool = True
    violations: List[Dict[str, str]] = field(default_factory=list)
    effective_autonomy: int = 0

    def add_violation(self, policy: str, reason: str):
        self.allowed = False
        self.violations.append({"policy": policy, "reason": reason})


class AutonomyPolicy:
    """Enforces autonomy levels: most restrictive between agent, task, and tool."""

    def __init__(self):
        self.default_level = 2

    def check(self, agent_level: int, task_level: int, tool_level: int) -> PolicyResult:
        result = PolicyResult()
        effective = min(agent_level, task_level, tool_level)
        result.effective_autonomy = effective

        if agent_level < 0 or agent_level > 5:
            result.add_violation("autonomy", f"Invalid agent autonomy level: {agent_level}")
        if task_level < 0 or task_level > 5:
            result.add_violation("autonomy", f"Invalid task autonomy level: {task_level}")
        if tool_level < 0 or tool_level > 5:
            result.add_violation("autonomy", f"Invalid tool autonomy level: {tool_level}")

        return result


class ToolPolicy:
    """Validates tool access against required autonomy levels."""

    def __init__(self):
        self._tool_requirements: Dict[str, int] = {
            "filesystem_read": 1,
            "filesystem_write": 2,
            "terminal": 3,
            "git_commit": 3,
            "deploy": 5,
        }

    def check(self, tool_name: str, current_autonomy: int) -> PolicyResult:
        result = PolicyResult()
        required = self._tool_requirements.get(tool_name, 2)

        if current_autonomy < required:
            result.add_violation(
                "tool",
                f"Tool '{tool_name}' requires autonomy {required}, "
                f"but current autonomy is {current_autonomy}"
            )
        return result


class ExecutionPolicy:
    """Enforces execution constraints like max iterations and required artifacts."""

    def __init__(self, max_iterations: int = 3, require_verification: bool = True):
        self.max_iterations = max_iterations
        self.require_verification = require_verification

    def check(self, current_iteration: int, has_verification: bool = False) -> PolicyResult:
        result = PolicyResult()

        if current_iteration > self.max_iterations:
            result.add_violation(
                "execution",
                f"Exceeded max iterations ({current_iteration} > {self.max_iterations})"
            )

        if self.require_verification and not has_verification:
            result.add_violation(
                "execution",
                "Verification is required but was not performed"
            )

        return result


class PolicyEngine:
    """Composable policy evaluation: runs all policies and collects violations.

    Phase 12: Now includes KnowledgePolicyEngine as a subordinate domain
    component. The canonical hierarchy is:
        PolicyEngine
        ├── Execution Policy
        ├── Tool Policy
        ├── Autonomy Policy
        └── Knowledge Policy (subordinate/domain component)
    """

    def __init__(self):
        self.autonomy = AutonomyPolicy()
        self.tool_policy = ToolPolicy()
        self.execution = ExecutionPolicy()
        # Phase 12: Knowledge policy as subordinate domain component
        from .knowledge_policy import KnowledgePolicyEngine
        from .knowledge_bridge import KnowledgePolicyBridge
        self.knowledge_policy = KnowledgePolicyEngine()
        self.knowledge_bridge = KnowledgePolicyBridge(self, self.knowledge_policy)

    def evaluate_all(self, **kwargs) -> PolicyResult:
        """Evaluate all applicable policies and return combined result."""
        result = PolicyResult()

        # Autonomy check
        if all(k in kwargs for k in ["agent_level", "task_level", "tool_level"]):
            auto_result = self.autonomy.check(
                kwargs["agent_level"], kwargs["task_level"], kwargs["tool_level"]
            )
            result.effective_autonomy = auto_result.effective_autonomy
            if not auto_result.allowed:
                for v in auto_result.violations:
                    result.add_violation(v["policy"], v["reason"])

        # Tool check
        if "tool_name" in kwargs and "current_autonomy" in kwargs:
            tool_result = self.tool_policy.check(
                kwargs["tool_name"], kwargs["current_autonomy"]
            )
            if not tool_result.allowed:
                for v in tool_result.violations:
                    result.add_violation(v["policy"], v["reason"])

        # Execution check
        if "current_iteration" in kwargs:
            exec_result = self.execution.check(
                kwargs["current_iteration"],
                kwargs.get("has_verification", False),
            )
            if not exec_result.allowed:
                for v in exec_result.violations:
                    result.add_violation(v["policy"], v["reason"])

        return result

    def evaluate_knowledge_operation(self, operation: str, subject: str, **kwargs) -> PolicyResult:
        """Evaluate a knowledge operation through KnowledgePolicyEngine.

        Phase 12: Convenience method for knowledge-specific policy evaluation.
        """
        auth = kwargs.pop("authorization", None)
        decision = self.knowledge_policy.evaluate_operation(
            operation=operation, subject=subject, authorization=auth, **kwargs
        )
        result = PolicyResult(allowed=decision.allowed)
        if not decision.allowed:
            result.add_violation(f"knowledge:{decision.policy}", decision.reason)
        return result
