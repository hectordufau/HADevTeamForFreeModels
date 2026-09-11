"""
Workflow Engine for V2.1: defines and executes agent pipelines with branching.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import yaml
import os


@dataclass
class WorkflowStep:
    role: str
    required_capabilities: List[str] = field(default_factory=list)
    model_policy_ref: str = ""
    allowed_tools: List[str] = field(default_factory=list)
    autonomy_override: Optional[int] = None
    on_pass: str = "next"  # next, complete, or step_name
    on_fail: str = "feedback"  # feedback, retry, block, or step_name


@dataclass
class WorkflowDefinition:
    name: str = "default"
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    max_iterations: int = 3


class WorkflowEngine:
    """Executes multi-step workflows with branching logic."""

    def __init__(self, harness: Any):
        self.harness = harness

    def load_definition(self, path: str) -> WorkflowDefinition:
        """Load a workflow definition from YAML."""
        with open(path) as f:
            data = yaml.safe_load(f)

        steps = []
        for s in data.get("steps", []):
            steps.append(WorkflowStep(
                role=s["role"],
                required_capabilities=s.get("capabilities", []),
                model_policy_ref=s.get("model_policy", ""),
                allowed_tools=s.get("tools", []),
                autonomy_override=s.get("autonomy_override"),
                on_pass=s.get("on_pass", "next"),
                on_fail=s.get("on_fail", "feedback"),
            ))

        return WorkflowDefinition(
            name=data.get("name", "default"),
            description=data.get("description", ""),
            steps=steps,
            max_iterations=data.get("max_iterations", 3),
        )

    def execute(self, workflow: WorkflowDefinition, task: Any) -> Dict[str, Any]:
        """Execute a workflow against a task contract."""
        step_results = []
        iteration = 0
        current_idx = 0
        max_iter = workflow.max_iterations

        while current_idx < len(workflow.steps):
            if iteration > max_iter:
                step_results.append({
                    "step": workflow.steps[current_idx].role,
                    "status": "blocked",
                    "error": f"Exceeded max iterations ({max_iter})",
                })
                break

            step = workflow.steps[current_idx]
            iteration += 1

            # Load agent
            try:
                agent = self.harness.agents.load(step.role)
            except Exception as e:
                step_results.append({"step": step.role, "status": "failed", "error": str(e)})
                if step.on_fail == "block":
                    break
                current_idx += 1
                continue

            # Select model
            try:
                selection = self.harness.router.select(
                    required_capabilities=step.required_capabilities or agent.profile.capabilities,
                    role=step.role,
                )
            except Exception as e:
                step_results.append({"step": step.role, "status": "failed", "error": str(e)})
                if step.on_fail == "block":
                    break
                current_idx += 1
                continue

            # Build context
            context = self.harness.context.build_context(
                task_contract=task,
                agent_role=step.role,
                agent_soul=agent.soul_content,
                relevant_files=task.spec.allowed_changes,
            )

            # Execute
            result = self.harness.executor.execute(
                agent_def=agent,
                model_id=selection.model_id,
                task_contract=task,
                context_pack=context,
            )

            step_result = {
                "step": step.role,
                "model": selection.model_id,
                "status": result.status,
                "changes": result.changes,
                "claims": result.claims,
            }
            step_results.append(step_result)

            # Branching
            if result.status == "completed":
                if step.on_pass == "complete":
                    break
                current_idx += 1
            else:
                if step.on_fail == "block":
                    break
                elif step.on_fail == "feedback":
                    # Stay on same step for retry
                    continue
                else:
                    # Jump to named step
                    named_idx = next(
                        (i for i, s in enumerate(workflow.steps) if s.role == step.on_fail),
                        current_idx + 1,
                    )
                    current_idx = named_idx

        return {
            "workflow": workflow.name,
            "task_id": task.metadata.id,
            "steps": step_results,
            "total_steps": len(step_results),
            "status": "completed" if all(s["status"] == "completed" for s in step_results) else "failed",
        }
