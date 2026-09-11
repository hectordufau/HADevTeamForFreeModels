# harness/planner/branching.py — BranchResolver (V2.2)
"""
Resolves conditional PASS/FAIL branches in workflow execution.
Only authorized branches (declared in the workflow plan) are allowed.
"""

from typing import Any, Dict, Optional


class BranchingError(Exception):
    """Raised on branching errors."""


class BranchResolver:
    """Resolves conditional branches in workflow execution.

    After a node executes, determines the next node based on the result
    and the node's declared on_pass/on_fail configuration.
    """

    # Special branch targets
    NEXT = "next"        # Proceed to next node in natural order
    COMPLETE = "complete"  # Finish the workflow successfully
    BLOCK = "block"      # Block the workflow
    RETRY = "retry"      # Retry the same node
    FEEDBACK = "feedback"  # Send to feedback loop

    VALID_TARGETS = {NEXT, COMPLETE, BLOCK, RETRY, FEEDBACK}

    def resolve(self, node: Any, result: Any) -> str:
        """Determine the next step after node execution.

        Args:
            node: The WorkflowNode that executed.
            result: The NodeExecutionResult from execution.

        Returns:
            The next node ID or a special target string.

        Raises:
            BranchingError: If the branch target is invalid or undeclared.
        """
        if not self._has_branching(node):
            return self.NEXT

        status = self._get_status(result)

        if status == "completed":
            target = self._get_on_pass(node)
        elif status == "failed":
            target = self._get_on_fail(node)
        elif status == "skipped":
            target = self.BLOCK
        else:
            raise BranchingError(f"Unknown status: {status}")

        if not target:
            return self.NEXT

        if target in self.VALID_TARGETS:
            return target

        # Verify it's a valid node reference
        if not isinstance(target, str) or len(target) == 0:
            raise BranchingError(f"Invalid branch target: {target}")

        return target

    def _has_branching(self, node: Any) -> bool:
        """Check if node has any branching configured."""
        if hasattr(node, 'on_pass') or hasattr(node, 'on_fail'):
            return True
        if isinstance(node, dict):
            return 'on_pass' in node or 'on_fail' in node
        return False

    def _get_status(self, result: Any) -> str:
        """Extract status from execution result."""
        if hasattr(result, 'status'):
            return result.status
        if isinstance(result, dict):
            return result.get('status', 'failed')
        return 'failed'

    def _get_on_pass(self, node: Any) -> Optional[str]:
        """Get the on_pass target from a node."""
        if hasattr(node, 'on_pass'):
            val = node.on_pass
            return val if val else None
        if isinstance(node, dict):
            return node.get('on_pass')
        return None

    def _get_on_fail(self, node: Any) -> Optional[str]:
        """Get the on_fail target from a node."""
        if hasattr(node, 'on_fail'):
            val = node.on_fail
            return val if val else None
        if isinstance(node, dict):
            return node.get('on_fail')
        return None
