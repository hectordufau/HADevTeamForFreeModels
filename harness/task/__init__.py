# harness/task/__init__.py — Task Contract Engine
"""
Task Contract: the formal definition of an engineering task.
Supports YAML-based task contracts with validation.
"""

import os
import uuid
import yaml
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


class TaskError(Exception):
    """Raised on task contract violations."""


@dataclass
class TaskVerificationSpec:
    required: List[str] = field(default_factory=list)


@dataclass
class TaskSpec:
    objective: str = ""
    requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    allowed_changes: List[str] = field(default_factory=list)
    forbidden_changes: List[str] = field(default_factory=list)
    verification: TaskVerificationSpec = field(default_factory=TaskVerificationSpec)
    autonomy: Dict[str, Any] = field(default_factory=lambda: {"maximum": 3})


@dataclass
class TaskMetadata:
    id: str = ""
    title: str = ""


@dataclass
class TaskContract:
    api_version: str = "harness/v1"
    kind: str = "Task"
    metadata: TaskMetadata = field(default_factory=TaskMetadata)
    spec: TaskSpec = field(default_factory=TaskSpec)

    def validate(self) -> List[str]:
        """Validate the contract, returning a list of errors (empty = valid)."""
        errors = []
        if not self.metadata.id:
            errors.append("metadata.id is required")
        if not self.spec.objective:
            errors.append("spec.objective is required")
        if not self.spec.acceptance_criteria:
            errors.append("spec.acceptance_criteria must have at least one criterion")
        max_auto = self.spec.autonomy.get("maximum", 3)
        if not isinstance(max_auto, int) or max_auto < 0 or max_auto > 5:
            errors.append(f"spec.autonomy.maximum must be integer 0-5, got {max_auto}")
        return errors

    def is_valid(self) -> bool:
        return len(self.validate()) == 0


class TaskManager:
    """Manages task contracts: creation, loading, validation, state persistence."""

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir

    def create(self, definition: dict) -> TaskContract:
        """Create a TaskContract from a dictionary definition."""
        metadata = definition.get("metadata", {})
        spec_dict = definition.get("spec", {})
        ver = spec_dict.get("verification", {})

        contract = TaskContract(
            api_version=definition.get("apiVersion", "harness/v1"),
            kind=definition.get("kind", "Task"),
            metadata=TaskMetadata(
                id=metadata.get("id", self._generate_id()),
                title=metadata.get("title", ""),
            ),
            spec=TaskSpec(
                objective=spec_dict.get("objective", ""),
                requirements=spec_dict.get("requirements", []),
                constraints=spec_dict.get("constraints", []),
                acceptance_criteria=spec_dict.get("acceptance_criteria", []),
                allowed_changes=spec_dict.get("allowed_changes", []),
                forbidden_changes=spec_dict.get("forbidden_changes", []),
                verification=TaskVerificationSpec(
                    required=ver.get("required", []) if isinstance(ver, dict) else []
                ),
                autonomy=spec_dict.get("autonomy", {"maximum": 3}),
            ),
        )
        return contract

    def load(self, path: str) -> TaskContract:
        """Load a TaskContract from a YAML file."""
        if not os.path.exists(path):
            raise TaskError(f"Task file not found: {path}")
        with open(path) as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise TaskError(f"Invalid YAML: {e}")
        if not isinstance(data, dict):
            raise TaskError("Task file must contain a mapping")
        contract = self.create(data)
        errors = contract.validate()
        if errors:
            raise TaskError("Task validation failed: " + "; ".join(errors))
        return contract

    def save(self, contract: TaskContract, path: str):
        """Persist a task contract to a YAML file."""
        data = {
            "apiVersion": contract.api_version,
            "kind": contract.kind,
            "metadata": asdict(contract.metadata),
            "spec": {
                "objective": contract.spec.objective,
                "requirements": contract.spec.requirements,
                "constraints": contract.spec.constraints,
                "acceptance_criteria": contract.spec.acceptance_criteria,
                "allowed_changes": contract.spec.allowed_changes,
                "forbidden_changes": contract.spec.forbidden_changes,
                "verification": {"required": contract.spec.verification.required},
                "autonomy": contract.spec.autonomy,
            },
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def update_state(self, task_id: str, state: str, state_dir: Optional[str] = None):
        """Update and persist task state (delegated to StateManager in production)."""
        # In V2.0 the StateManager handles this. This is a lightweight hook.
        if state_dir is None:
            state_dir = os.path.join(os.getcwd(), "tasks", "active")
        state_file = os.path.join(state_dir, f"{task_id}.state")
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        state_data = {"task_id": task_id, "state": state, "updated_at": datetime.utcnow().isoformat()}
        with open(state_file, "w") as f:
            yaml.dump(state_data, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _generate_id() -> str:
        return f"TASK-{uuid.uuid4().hex[:8].upper()}"
