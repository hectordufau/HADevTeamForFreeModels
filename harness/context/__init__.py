# harness/context/__init__.py — Context Engineering
"""
Builds minimal, relevant context packs for agent execution.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ContextPack:
    task_id: str = ""
    task_objective: str = ""
    task_requirements: List[str] = field(default_factory=list)
    task_constraints: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    allowed_changes: List[str] = field(default_factory=list)
    forbidden_changes: List[str] = field(default_factory=list)
    agent_role: str = ""
    agent_soul: str = ""
    relevant_files: List[str] = field(default_factory=list)
    architectural_decisions: List[str] = field(default_factory=list)
    team_rules: str = ""
    memory_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task": {
                "id": self.task_id,
                "objective": self.task_objective,
                "requirements": self.task_requirements,
                "constraints": self.task_constraints,
                "acceptance_criteria": self.acceptance_criteria,
                "allowed_changes": self.allowed_changes,
                "forbidden_changes": self.forbidden_changes,
            },
            "agent": {"role": self.agent_role},
            "relevant_files": self.relevant_files,
            "decisions": self.architectural_decisions,
        }


class ContextManager:
    """Builds context packs for agents, avoiding full-repository injection."""

    def __init__(self, project_root: str = "."):
        self.project_root = project_root

    def build_context(
        self,
        task_contract: Any,
        agent_role: str,
        agent_soul: str = "",
        relevant_files: Optional[List[str]] = None,
        decisions: Optional[List[str]] = None,
        team_rules: str = "",
    ) -> ContextPack:
        """Build a ContextPack for the given task and agent."""
        spec = task_contract.spec

        pack = ContextPack(
            task_id=task_contract.metadata.id,
            task_objective=spec.objective,
            task_requirements=spec.requirements,
            task_constraints=spec.constraints,
            acceptance_criteria=spec.acceptance_criteria,
            allowed_changes=spec.allowed_changes,
            forbidden_changes=spec.forbidden_changes,
            agent_role=agent_role,
            agent_soul=agent_soul,
            relevant_files=relevant_files or [],
            architectural_decisions=decisions or [],
            team_rules=team_rules,
        )
        return pack
