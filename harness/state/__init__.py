# harness/state/__init__.py — Task State Machine
"""
Persistent task lifecycle management with validated transitions.
"""

from typing import List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime


class StateError(Exception):
    """Raised on invalid state transitions."""


# Valid states per SPEC-V2.md §9
VALID_STATES: Set[str] = {
    "RECEIVED",
    "ANALYZING",
    "PLANNED",
    "IMPLEMENTING",
    "VERIFYING",
    "EVALUATING",
    "REVIEWING",
    "ITERATING",
    "COMPLETED",
    "FAILED",
    "BLOCKED",
    "NEEDS_HUMAN",
    "CANCELLED",
}

# Allowed transitions: {from_state: [to_state, ...]}
ALLOWED_TRANSITIONS: dict = {
    "RECEIVED": ["ANALYZING", "FAILED", "CANCELLED"],
    "ANALYZING": ["PLANNED", "FAILED", "CANCELLED", "BLOCKED"],
    "PLANNED": ["IMPLEMENTING", "FAILED", "CANCELLED", "BLOCKED"],
    "IMPLEMENTING": ["VERIFYING", "FAILED", "CANCELLED", "BLOCKED"],
    "VERIFYING": ["EVALUATING", "FAILED", "CANCELLED", "BLOCKED", "ITERATING"],
    "EVALUATING": ["REVIEWING", "FAILED", "CANCELLED", "BLOCKED", "ITERATING"],
    "REVIEWING": ["COMPLETED", "FAILED", "CANCELLED", "BLOCKED", "ITERATING"],
    "ITERATING": ["IMPLEMENTING", "CANCELLED", "NEEDS_HUMAN"],
    "COMPLETED": [],
    "FAILED": [],
    "BLOCKED": ["NEEDS_HUMAN", "CANCELLED"],
    "NEEDS_HUMAN": ["RECEIVED", "CANCELLED"],
    "CANCELLED": [],
}


@dataclass
class StateTransition:
    from_state: str
    to_state: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class TaskState:
    task_id: str
    current_state: str = "RECEIVED"
    history: List[StateTransition] = field(default_factory=list)

    def transition(self, to_state: str) -> "TaskState":
        """Attempt a state transition. Raises StateError if invalid."""
        if to_state not in VALID_STATES:
            raise StateError(f"Invalid target state: {to_state}")
        allowed = ALLOWED_TRANSITIONS.get(self.current_state, [])
        if to_state not in allowed:
            raise StateError(
                f"Cannot transition from {self.current_state} to {to_state}. "
                f"Allowed: {allowed}"
            )
        transition = StateTransition(from_state=self.current_state, to_state=to_state)
        self.history.append(transition)
        self.current_state = to_state
        return self

    def is_terminal(self) -> bool:
        return self.current_state in {"COMPLETED", "FAILED", "CANCELLED"}


class StateManager:
    """Manages task state persistence and lifecycle."""

    def __init__(self):
        self._states: dict = {}

    def register(self, task_id: str) -> TaskState:
        if task_id in self._states:
            raise StateError(f"Task {task_id} already registered")
        state = TaskState(task_id=task_id)
        self._states[task_id] = state
        return state

    def get(self, task_id: str) -> Optional[TaskState]:
        return self._states.get(task_id)

    def transition(self, task_id: str, to_state: str) -> TaskState:
        state = self._states.get(task_id)
        if not state:
            raise StateError(f"Task {task_id} not found")
        return state.transition(to_state)

    def persist(self, task_id: str, storage_path: str):
        """Persist state to YAML (stub — will integrate with file system in full impl)."""
        import yaml
        state = self._states.get(task_id)
        if not state:
            raise StateError(f"Task {task_id} not found")
        data = {
            "task_id": state.task_id,
            "current_state": state.current_state,
            "history": [{"from": h.from_state, "to": h.to_state, "timestamp": h.timestamp} for h in state.history],
        }
        with open(storage_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def restore(self, task_id: str, storage_path: str):
        """Restore state from YAML."""
        import os, yaml
        if not os.path.exists(storage_path):
            raise StateError(f"State file not found: {storage_path}")
        with open(storage_path) as f:
            data = yaml.safe_load(f)
        state = TaskState(task_id=data["task_id"], current_state=data["current_state"])
        for h in data.get("history", []):
            state.history.append(StateTransition(from_state=h["from"], to_state=h["to"], timestamp=h["timestamp"]))
        self._states[task_id] = state
        return state
