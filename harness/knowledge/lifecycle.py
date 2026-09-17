# harness/knowledge/lifecycle.py — V3.3 Record Lifecycle State Machine
"""
Lifecycle state machine for engineering records.

Each record type has its own lifecycle states and valid transitions.
Invalid transitions FAIL CLOSED — they are never coerced to the nearest
valid state. Terminal states are immutable.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json


class LifecycleError(Exception):
    """Raised on lifecycle validation errors."""


# ──────────────────────────────────────────────────────────────────────
# Lifecycle state definitions per record type
# ──────────────────────────────────────────────────────────────────────

# PRD: draft → review → accepted → deprecated → archived
PRD_LIFECYCLE = {
    "states": ["draft", "review", "accepted", "deprecated", "archived"],
    "initial": "draft",
    "terminal": {"archived"},
    "transitions": {
        "draft": {"review", "deprecated"},
        "review": {"accepted", "draft", "deprecated"},
        "accepted": {"deprecated"},
        "deprecated": {"archived"},
        "archived": set(),  # Terminal
    },
}

# NFR: draft → review → accepted → deprecated → archived
NFR_LIFECYCLE = {
    "states": ["draft", "review", "accepted", "deprecated", "archived"],
    "initial": "draft",
    "terminal": {"archived"},
    "transitions": {
        "draft": {"review", "deprecated"},
        "review": {"accepted", "draft", "deprecated"},
        "accepted": {"deprecated"},
        "deprecated": {"archived"},
        "archived": set(),
    },
}

# DR: proposed → accepted → deprecated → superseded → archived
DR_LIFECYCLE = {
    "states": ["proposed", "accepted", "deprecated", "superseded", "archived"],
    "initial": "proposed",
    "terminal": {"archived"},
    "transitions": {
        "proposed": {"accepted"},
        "accepted": {"deprecated", "superseded"},
        "deprecated": {"archived", "superseded"},
        "superseded": {"archived"},
        "archived": set(),
    },
}

# ADR: proposed → accepted → deprecated → superseded → archived
ADR_LIFECYCLE = {
    "states": ["proposed", "accepted", "deprecated", "superseded", "archived"],
    "initial": "proposed",
    "terminal": {"archived"},
    "transitions": {
        "proposed": {"accepted"},
        "accepted": {"deprecated", "superseded"},
        "deprecated": {"archived", "superseded"},
        "superseded": {"archived"},
        "archived": set(),
    },
}

# TDR: identified → acknowledged → remediation_planned → in_progress → resolved → closed
TDR_LIFECYCLE = {
    "states": ["identified", "acknowledged", "remediation_planned", "in_progress", "resolved", "closed"],
    "initial": "identified",
    "terminal": {"closed"},
    "transitions": {
        "identified": {"acknowledged", "closed"},
        "acknowledged": {"remediation_planned", "closed"},
        "remediation_planned": {"in_progress", "closed"},
        "in_progress": {"resolved", "remediation_planned"},
        "resolved": {"closed"},
        "closed": set(),
    },
}

# RSK: identified → assessed → mitigated → accepted → closed
RSK_LIFECYCLE = {
    "states": ["identified", "assessed", "mitigated", "accepted", "closed"],
    "initial": "identified",
    "terminal": {"closed"},
    "transitions": {
        "identified": {"assessed", "closed"},
        "assessed": {"mitigated", "accepted", "closed"},
        "mitigated": {"accepted", "closed"},
        "accepted": {"closed"},
        "closed": set(),
    },
}

# SEC: active → waived → expired → deprecated
SEC_LIFECYCLE = {
    "states": ["active", "waived", "expired", "deprecated"],
    "initial": "active",
    "terminal": {"deprecated"},
    "transitions": {
        "active": {"waived", "expired"},
        "waived": {"expired", "deprecated"},
        "expired": {"deprecated"},
        "deprecated": set(),
    },
}

# RCA: draft → analysis → corrective_action → preventive_action → closed
RCA_LIFECYCLE = {
    "states": ["draft", "analysis", "corrective_action", "preventive_action", "closed"],
    "initial": "draft",
    "terminal": {"closed"},
    "transitions": {
        "draft": {"analysis", "closed"},
        "analysis": {"corrective_action", "closed"},
        "corrective_action": {"preventive_action", "closed"},
        "preventive_action": {"closed"},
        "closed": set(),
    },
}

# ──────────────────────────────────────────────────────────────────────
# Lifecycle registry
# ──────────────────────────────────────────────────────────────────────

LIFECYCLE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "PRD": PRD_LIFECYCLE,
    "NFR": NFR_LIFECYCLE,
    "DR": DR_LIFECYCLE,
    "ADR": ADR_LIFECYCLE,
    "TDR": TDR_LIFECYCLE,
    "RSK": RSK_LIFECYCLE,
    "SEC": SEC_LIFECYCLE,
    "RCA": RCA_LIFECYCLE,
}

# Valid record types
VALID_RECORD_TYPES = set(LIFECYCLE_DEFINITIONS.keys())


@dataclass
class LifecycleTransition:
    """A single lifecycle transition event."""
    from_state: str
    to_state: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    reason: str = ""
    actor: str = ""

    def to_dict(self) -> dict:
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "actor": self.actor,
        }


def get_lifecycle_definition(record_type: str) -> Dict[str, Any]:
    """
    Get the lifecycle definition for a record type.
    
    Raises LifecycleError if record type is unknown.
    """
    if record_type not in LIFECYCLE_DEFINITIONS:
        raise LifecycleError(
            f"Unknown record type '{record_type}'. "
            f"Must be one of: {', '.join(sorted(VALID_RECORD_TYPES))}"
        )
    return LIFECYCLE_DEFINITIONS[record_type]


def get_initial_state(record_type: str) -> str:
    """Get the initial lifecycle state for a record type."""
    return get_lifecycle_definition(record_type)["initial"]


def get_terminal_states(record_type: str) -> Set[str]:
    """Get the terminal (immutable) states for a record type."""
    return get_lifecycle_definition(record_type)["terminal"]


def get_valid_states(record_type: str) -> List[str]:
    """Get all valid states for a record type."""
    return get_lifecycle_definition(record_type)["states"]


def is_valid_transition(record_type: str, from_state: str, to_state: str) -> bool:
    """
    Check if a lifecycle transition is valid.
    
    Rules:
    - Both states must be valid for the record type
    - The transition must be in the allowed transitions
    - Terminal states cannot transition out
    - Self-transitions (same state) are NOT valid
    """
    definition = get_lifecycle_definition(record_type)

    # Check states are valid
    if from_state not in definition["states"]:
        return False
    if to_state not in definition["states"]:
        return False

    # Self-transition is not valid
    if from_state == to_state:
        return False

    # Terminal states are immutable
    if from_state in definition["terminal"]:
        return False

    # Check allowed transitions
    allowed = definition["transitions"].get(from_state, set())
    return to_state in allowed


def transition(
    record_type: str,
    current_state: str,
    new_state: str,
    reason: str = "",
    actor: str = "",
) -> LifecycleTransition:
    """
    Attempt a lifecycle transition.
    
    Raises LifecycleError if the transition is invalid (fail closed).
    Returns a LifecycleTransition on success.
    """
    if not is_valid_transition(record_type, current_state, new_state):
        definition = get_lifecycle_definition(record_type)
        allowed = definition["transitions"].get(current_state, set())
        raise LifecycleError(
            f"Invalid lifecycle transition for {record_type}: "
            f"'{current_state}' → '{new_state}'. "
            f"Allowed transitions from '{current_state}': "
            f"{', '.join(sorted(allowed)) if allowed else 'none (terminal state)'}"
        )

    return LifecycleTransition(
        from_state=current_state,
        to_state=new_state,
        reason=reason,
        actor=actor,
    )


def is_terminal_state(record_type: str, state: str) -> bool:
    """Check if a state is terminal (immutable) for a record type."""
    return state in get_terminal_states(record_type)


def get_allowed_transitions(record_type: str, state: str) -> Set[str]:
    """Get all allowed next states from a given state."""
    definition = get_lifecycle_definition(record_type)
    if state not in definition["states"]:
        return set()
    return definition["transitions"].get(state, set())


def validate_lifecycle_state(record_type: str, state: str) -> None:
    """
    Validate that a state is valid for a record type.
    
    Raises LifecycleError if invalid.
    """
    definition = get_lifecycle_definition(record_type)
    if state not in definition["states"]:
        raise LifecycleError(
            f"Invalid lifecycle state '{state}' for {record_type}. "
            f"Must be one of: {', '.join(definition['states'])}"
        )
