# tests/unit/test_knowledge_lifecycle.py — Tests for V3.3 Lifecycle State Machine
"""Tests for lifecycle states, transitions, and validation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.knowledge.lifecycle import (
    LifecycleError,
    LifecycleTransition,
    get_lifecycle_definition,
    get_initial_state,
    get_terminal_states,
    get_valid_states,
    is_valid_transition,
    transition,
    is_terminal_state,
    get_allowed_transitions,
    validate_lifecycle_state,
    LIFECYCLE_DEFINITIONS,
    VALID_RECORD_TYPES,
    PRD_LIFECYCLE,
    NFR_LIFECYCLE,
    DR_LIFECYCLE,
    ADR_LIFECYCLE,
    TDR_LIFECYCLE,
    RSK_LIFECYCLE,
    SEC_LIFECYCLE,
    RCA_LIFECYCLE,
)


class TestLifecycleDefinitions:
    """Test lifecycle definitions for all record types."""

    def test_all_record_types_have_lifecycle(self):
        assert VALID_RECORD_TYPES == {"PRD", "NFR", "DR", "ADR", "TDR", "RSK", "SEC", "RCA"}
        assert len(LIFECYCLE_DEFINITIONS) == 8

    def test_prd_lifecycle(self):
        defn = get_lifecycle_definition("PRD")
        assert defn["initial"] == "draft"
        assert "archived" in defn["terminal"]
        assert "draft" in defn["states"]

    def test_nfr_lifecycle(self):
        defn = get_lifecycle_definition("NFR")
        assert defn["initial"] == "draft"
        assert "archived" in defn["terminal"]

    def test_dr_lifecycle(self):
        defn = get_lifecycle_definition("DR")
        assert defn["initial"] == "proposed"
        assert "archived" in defn["terminal"]

    def test_adr_lifecycle(self):
        defn = get_lifecycle_definition("ADR")
        assert defn["initial"] == "proposed"
        assert "archived" in defn["terminal"]

    def test_tdr_lifecycle(self):
        defn = get_lifecycle_definition("TDR")
        assert defn["initial"] == "identified"
        assert "closed" in defn["terminal"]

    def test_rsk_lifecycle(self):
        defn = get_lifecycle_definition("RSK")
        assert defn["initial"] == "identified"
        assert "closed" in defn["terminal"]

    def test_sec_lifecycle(self):
        defn = get_lifecycle_definition("SEC")
        assert defn["initial"] == "active"
        assert "deprecated" in defn["terminal"]

    def test_rca_lifecycle(self):
        defn = get_lifecycle_definition("RCA")
        assert defn["initial"] == "draft"
        assert "closed" in defn["terminal"]

    def test_unknown_record_type(self):
        with pytest.raises(LifecycleError, match="Unknown record type"):
            get_lifecycle_definition("UNKNOWN")


class TestInitialStates:
    """Test initial state for each record type."""

    def test_initial_states(self):
        assert get_initial_state("PRD") == "draft"
        assert get_initial_state("NFR") == "draft"
        assert get_initial_state("DR") == "proposed"
        assert get_initial_state("ADR") == "proposed"
        assert get_initial_state("TDR") == "identified"
        assert get_initial_state("RSK") == "identified"
        assert get_initial_state("SEC") == "active"
        assert get_initial_state("RCA") == "draft"


class TestTerminalStates:
    """Test terminal (immutable) states."""

    def test_terminal_states(self):
        assert is_terminal_state("PRD", "archived")
        assert is_terminal_state("NFR", "archived")
        assert is_terminal_state("DR", "archived")
        assert is_terminal_state("ADR", "archived")
        assert is_terminal_state("TDR", "closed")
        assert is_terminal_state("RSK", "closed")
        assert is_terminal_state("SEC", "deprecated")
        assert is_terminal_state("RCA", "closed")

    def test_non_terminal_states(self):
        assert not is_terminal_state("PRD", "draft")
        assert not is_terminal_state("PRD", "review")
        assert not is_terminal_state("DR", "proposed")
        assert not is_terminal_state("TDR", "identified")
        assert not is_terminal_state("SEC", "active")


class TestValidTransitions:
    """Test valid lifecycle transitions."""

    def test_prd_transitions(self):
        assert is_valid_transition("PRD", "draft", "review")
        assert is_valid_transition("PRD", "review", "accepted")
        assert is_valid_transition("PRD", "accepted", "deprecated")
        assert is_valid_transition("PRD", "deprecated", "archived")

    def test_nfr_transitions(self):
        assert is_valid_transition("NFR", "draft", "review")
        assert is_valid_transition("NFR", "review", "accepted")
        assert is_valid_transition("NFR", "accepted", "deprecated")
        assert is_valid_transition("NFR", "deprecated", "archived")

    def test_dr_transitions(self):
        assert is_valid_transition("DR", "proposed", "accepted")
        assert is_valid_transition("DR", "accepted", "deprecated")
        assert is_valid_transition("DR", "accepted", "superseded")
        assert is_valid_transition("DR", "deprecated", "archived")
        assert is_valid_transition("DR", "superseded", "archived")

    def test_adr_transitions(self):
        assert is_valid_transition("ADR", "proposed", "accepted")
        assert is_valid_transition("ADR", "accepted", "deprecated")
        assert is_valid_transition("ADR", "accepted", "superseded")
        assert is_valid_transition("ADR", "deprecated", "archived")
        assert is_valid_transition("ADR", "superseded", "archived")

    def test_tdr_transitions(self):
        assert is_valid_transition("TDR", "identified", "acknowledged")
        assert is_valid_transition("TDR", "acknowledged", "remediation_planned")
        assert is_valid_transition("TDR", "remediation_planned", "in_progress")
        assert is_valid_transition("TDR", "in_progress", "resolved")
        assert is_valid_transition("TDR", "resolved", "closed")

    def test_rsk_transitions(self):
        assert is_valid_transition("RSK", "identified", "assessed")
        assert is_valid_transition("RSK", "assessed", "mitigated")
        assert is_valid_transition("RSK", "mitigated", "accepted")
        assert is_valid_transition("RSK", "accepted", "closed")

    def test_sec_transitions(self):
        assert is_valid_transition("SEC", "active", "waived")
        assert is_valid_transition("SEC", "waived", "expired")
        assert is_valid_transition("SEC", "expired", "deprecated")

    def test_rca_transitions(self):
        assert is_valid_transition("RCA", "draft", "analysis")
        assert is_valid_transition("RCA", "analysis", "corrective_action")
        assert is_valid_transition("RCA", "corrective_action", "preventive_action")
        assert is_valid_transition("RCA", "preventive_action", "closed")


class TestInvalidTransitions:
    """Test invalid lifecycle transitions (fail closed)."""

    def test_prd_invalid_transitions(self):
        # Cannot skip states
        assert not is_valid_transition("PRD", "draft", "accepted")
        assert not is_valid_transition("PRD", "draft", "archived")
        # Cannot go backward
        assert not is_valid_transition("PRD", "accepted", "draft")
        # Terminal state
        assert not is_valid_transition("PRD", "archived", "draft")

    def test_dr_invalid_transitions(self):
        # Cannot skip
        assert not is_valid_transition("DR", "proposed", "deprecated")
        assert not is_valid_transition("DR", "proposed", "archived")
        # Cannot go backward
        assert not is_valid_transition("DR", "accepted", "proposed")
        # Terminal state
        assert not is_valid_transition("DR", "archived", "proposed")

    def test_tdr_invalid_transitions(self):
        # Cannot skip
        assert not is_valid_transition("TDR", "identified", "in_progress")
        assert not is_valid_transition("TDR", "identified", "resolved")
        # Cannot go backward
        assert not is_valid_transition("TDR", "resolved", "in_progress")
        # Terminal state
        assert not is_valid_transition("TDR", "closed", "identified")

    def test_sec_invalid_transitions(self):
        # Cannot skip
        assert not is_valid_transition("SEC", "active", "deprecated")
        # Cannot go backward
        assert not is_valid_transition("SEC", "waived", "active")
        # Terminal state
        assert not is_valid_transition("SEC", "deprecated", "active")

    def test_self_transition_not_valid(self):
        assert not is_valid_transition("PRD", "draft", "draft")
        assert not is_valid_transition("DR", "proposed", "proposed")
        assert not is_valid_transition("TDR", "identified", "identified")

    def test_invalid_from_state(self):
        assert not is_valid_transition("PRD", "invalid_state", "draft")

    def test_invalid_to_state(self):
        assert not is_valid_transition("PRD", "draft", "invalid_state")


class TestTransitionFunction:
    """Test the transition() function."""

    def test_valid_transition(self):
        t = transition("PRD", "draft", "review")
        assert t.from_state == "draft"
        assert t.to_state == "review"
        assert t.timestamp is not None

    def test_transition_with_reason(self):
        t = transition("PRD", "draft", "review", reason="Ready for review", actor="alice")
        assert t.reason == "Ready for review"
        assert t.actor == "alice"

    def test_invalid_transition_raises(self):
        with pytest.raises(LifecycleError, match="Invalid lifecycle transition"):
            transition("PRD", "draft", "archived")

    def test_terminal_state_raises(self):
        with pytest.raises(LifecycleError, match="terminal state"):
            transition("PRD", "archived", "draft")


class TestGetAllowedTransitions:
    """Test getting allowed transitions from a state."""

    def test_prd_draft_allowed(self):
        allowed = get_allowed_transitions("PRD", "draft")
        assert "review" in allowed
        assert "deprecated" in allowed

    def test_prd_accepted_allowed(self):
        allowed = get_allowed_transitions("PRD", "accepted")
        assert "deprecated" in allowed
        assert len(allowed) == 1

    def test_prd_archived_allowed(self):
        allowed = get_allowed_transitions("PRD", "archived")
        assert len(allowed) == 0

    def test_invalid_state_returns_empty(self):
        allowed = get_allowed_transitions("PRD", "invalid_state")
        assert len(allowed) == 0


class TestValidateLifecycleState:
    """Test lifecycle state validation."""

    def test_valid_states(self):
        for state in ["draft", "review", "accepted", "deprecated", "archived"]:
            validate_lifecycle_state("PRD", state)

    def test_invalid_state(self):
        with pytest.raises(LifecycleError, match="Invalid lifecycle state"):
            validate_lifecycle_state("PRD", "invalid_state")


class TestLifecycleTransitionDataclass:
    """Test LifecycleTransition dataclass."""

    def test_create_transition(self):
        t = LifecycleTransition(
            from_state="draft",
            to_state="review",
            reason="Ready",
            actor="alice",
        )
        assert t.from_state == "draft"
        assert t.to_state == "review"
        assert t.reason == "Ready"
        assert t.actor == "alice"

    def test_to_dict(self):
        t = LifecycleTransition(from_state="draft", to_state="review")
        d = t.to_dict()
        assert d["from_state"] == "draft"
        assert d["to_state"] == "review"
        assert "timestamp" in d
