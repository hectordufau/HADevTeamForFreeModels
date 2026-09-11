# harness/planner/replanner.py — Replanner (V2.2)
"""
Adaptive replanning after execution failure.
Creates new workflow for remaining steps; completed nodes are frozen.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import yaml


class ReplannerError(Exception):
    """Raised on replanning errors."""


@dataclass
class ReplanReport:
    """Documentation of why and how the workflow changed."""
    original_workflow_name: str
    reason: str
    failed_node_id: str
    category: str
    changes: List[Dict[str, Any]] = field(default_factory=list)
    completed_nodes: List[str] = field(default_factory=list)
    modified_nodes: List[str] = field(default_factory=list)
    plan_count: int = 0
    created_at: str = ""


class Replanner:
    """Adaptive replanning after execution failure.

    Rules:
        - Only PENDING nodes can be modified
        - COMPLETED nodes are frozen (cannot be re-executed)
        - COMPLETED evidence is preserved
        - Each replan generates a replan report
        - Maximum replan count is configurable (default 3)
    """

    def __init__(self, max_plans: int = 3):
        self.max_plans = max_plans
        self._plan_count: Dict[str, int] = {}  # task_id -> plan count

    def replan(self, task: Any, current_workflow: Any,
               completed_nodes: List[str], failed_node: str,
               failure_report: Any) -> Any:
        """Create a new workflow for remaining steps.

        Args:
            task: The original TaskContract.
            current_workflow: The current ExecutionWorkflow.
            completed_nodes: List of node IDs that completed successfully.
            failed_node: The node ID that failed.
            failure_report: FailureReport from FailureAnalyzer.

        Returns:
            A new ExecutionWorkflow with modifications.

        Raises:
            ReplannerError: If max replans exceeded or invalid state.
        """
        task_id = self._get_task_id(task)
        count = self._plan_count.get(task_id, 0) + 1

        if count > self.max_plans:
            raise ReplannerError(
                f"Max replans ({self.max_plans}) exceeded for task {task_id}"
            )

        self._plan_count[task_id] = count

        # Clone the workflow
        new_workflow = self._clone_workflow(current_workflow)

        # Validate: completed nodes must exist in the workflow
        completed_set = set(completed_nodes)
        workflow_node_ids = {n.id for n in new_workflow.nodes}
        unknown_completed = completed_set - workflow_node_ids
        if unknown_completed:
            raise ReplannerError(f"Completed nodes not in workflow: {unknown_completed}")

        # Get the failure report data
        category = self._get_category(failure_report)
        missing_caps = self._get_missing_capabilities(failure_report)

        # Find and modify pending nodes
        modified_nodes = []
        for node in new_workflow.nodes:
            if node.id in completed_set:
                # COMPLETED nodes are frozen — verify they're actually marked completed
                if node.id == failed_node:
                    # The failed node is typically NOT completed
                    pass
                continue

            if node.id == failed_node:
                # Failed node: update its configuration based on failure
                self._modify_failed_node(node, category, missing_caps)
                modified_nodes.append(node.id)
            elif node.id not in completed_set:
                # Other pending nodes: may need adjustment
                self._adjust_pending_node(node, category, missing_caps)
                modified_nodes.append(node.id)

        # Build replan report
        report = ReplanReport(
            original_workflow_name=self._get_workflow_name(current_workflow),
            reason=self._get_reason(failure_report),
            failed_node_id=failed_node,
            category=category,
            changes=self._compute_changes(current_workflow, new_workflow),
            completed_nodes=list(completed_set),
            modified_nodes=modified_nodes,
            plan_count=count,
            created_at=datetime.utcnow().isoformat(),
        )

        # Attach the report to the workflow
        new_workflow._replan_report = report

        return new_workflow

    def generate_replan_report(self, original: Any, new: Any,
                               reason: str) -> dict:
        """Generate a replan report as a serializable dict."""
        report = getattr(new, '_replan_report', None)
        if report:
            return {
                "original_workflow": self._get_workflow_name(original),
                "new_workflow": self._get_workflow_name(new),
                "reason": reason,
                "failed_node_id": report.failed_node_id,
                "category": report.category,
                "changes": report.changes,
                "completed_nodes": report.completed_nodes,
                "modified_nodes": report.modified_nodes,
                "plan_count": report.plan_count,
                "created_at": report.created_at,
            }

        # Fallback: generate from scratch
        return {
            "original_workflow": self._get_workflow_name(original),
            "new_workflow": self._get_workflow_name(new),
            "reason": reason,
            "failed_node_id": "",
            "category": "unknown",
            "changes": [],
            "completed_nodes": [],
            "modified_nodes": [],
            "plan_count": 0,
            "created_at": datetime.utcnow().isoformat(),
        }

    def get_plan_count(self, task_id: str) -> int:
        """Get current plan count for a task."""
        return self._plan_count.get(task_id, 0)

    def reset_plan_count(self, task_id: str):
        """Reset plan count for a task (e.g., on fresh execution)."""
        self._plan_count[task_id] = 0

    def _clone_workflow(self, workflow: Any) -> Any:
        """Deep clone a workflow to avoid mutating the original."""
        from harness.planner.workflow_planner import ExecutionWorkflow, WorkflowNode

        if hasattr(workflow, 'to_dict'):
            data = workflow.to_dict()
            return ExecutionWorkflow.from_dict(data)
        return workflow

    def _modify_failed_node(self, node: Any, category: str,
                            missing_caps: List[str]):
        """Modify a failed node based on failure analysis."""
        # Change on_fail behavior based on category
        if category in ("test", "implementation"):
            node.on_fail = "retry"
        elif category in ("configuration", "environment"):
            node.on_fail = "retry"
        elif category in ("security",):
            node.on_fail = "block"
        else:
            node.on_fail = "block"

        # If missing capabilities, add them as optional
        if hasattr(node, 'workspace') and missing_caps:
            pass  # Capabilities will be added as new nodes by the orchestrator

    def _adjust_pending_node(self, node: Any, category: str,
                             missing_caps: List[str]):
        """Adjust a pending node based on failure context."""
        if category in ("test", "implementation"):
            # After a test/implementation failure, subsequent nodes should be more cautious
            if node.on_fail in ("next", "complete"):
                node.on_fail = "block"

    def _compute_changes(self, original: Any, new: Any) -> List[Dict[str, Any]]:
        """Compute differences between original and new workflow."""
        changes = []
        orig_nodes = {n.id: n for n in original.nodes}
        new_nodes = {n.id: n for n in new.nodes}

        for nid, new_node in new_nodes.items():
            if nid not in orig_nodes:
                changes.append({"type": "added", "node": nid})
            else:
                orig = orig_nodes[nid]
                if (orig.on_pass != new_node.on_pass or
                        orig.on_fail != new_node.on_fail):
                    changes.append({
                        "type": "modified",
                        "node": nid,
                        "on_pass": {"from": orig.on_pass, "to": new_node.on_pass},
                        "on_fail": {"from": orig.on_fail, "to": new_node.on_fail},
                    })

        for nid in orig_nodes:
            if nid not in new_nodes:
                changes.append({"type": "removed", "node": nid})

        return changes

    def _get_task_id(self, task: Any) -> str:
        if hasattr(task, 'metadata') and hasattr(task.metadata, 'id'):
            return task.metadata.id
        if isinstance(task, dict):
            return task.get('metadata', {}).get('id', 'unknown')
        return 'unknown'

    def _get_workflow_name(self, workflow: Any) -> str:
        if hasattr(workflow, 'name'):
            return workflow.name
        if isinstance(workflow, dict):
            return workflow.get('name', 'unknown')
        return 'unknown'

    def _get_category(self, report: Any) -> str:
        if hasattr(report, 'category'):
            return report.category
        if isinstance(report, dict):
            return report.get('category', 'unknown')
        return 'unknown'

    def _get_missing_capabilities(self, report: Any) -> List[str]:
        if hasattr(report, 'missing_capabilities'):
            return list(report.missing_capabilities)
        if isinstance(report, dict):
            return list(report.get('missing_capabilities', []))
        return []

    def _get_reason(self, report: Any) -> str:
        if hasattr(report, 'message'):
            return report.message
        if isinstance(report, dict):
            return report.get('message', '')
        return ''
