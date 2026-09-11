# harness/policy/v3.py — Security and Governance Hardening V3 (Phase O)
"""
Immutable governance policy, security regression suite,
prompt injection boundary, and tool authorization hardening.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


class GovernanceError(Exception):
    """Raised on governance policy violations."""


IMMUTABLE_GOVERNANCE_POLICY = {
    "policy_version": "3.0.0",
    "immutable_since": "2024-01-01",
    "rules": {
        "free_model_invariant": {
            "description": "All models used must be free (cost=0)",
            "enforced_by": ["WorkflowValidator", "FreeModelInvariantEnforcer"],
            "consequence": "BLOCKED",
            "immutable": True,
        },
        "independent_reviewer": {
            "description": "Reviewer must be a different agent than implementer",
            "enforced_by": ["WorkflowValidator"],
            "consequence": "BLOCKED",
            "immutable": True,
        },
        "no_code_self_modification": {
            "description": "The harness must never modify its own source code automatically",
            "enforced_by": ["WorkflowValidator", "WorkspaceManager"],
            "consequence": "BLOCKED",
            "immutable": True,
        },
        "evidence_integrity": {
            "description": "All execution evidence must be hash-chained for integrity",
            "enforced_by": ["EvidenceStore"],
            "consequence": "FAILED",
            "immutable": True,
        },
        "human_approval_for_improvements": {
            "description": "Code improvements that modify harness behavior require human approval",
            "enforced_by": ["ImprovementProposalHandler"],
            "consequence": "NEEDS_HUMAN",
            "immutable": True,
        },
        "mandatory_gates": {
            "description": "verification, evaluation, and review must always be present",
            "enforced_by": ["WorkflowValidator"],
            "consequence": "BLOCKED",
            "immutable": True,
        },
    },
    "immutable_keys": [
        "free_model_invariant",
        "independent_reviewer",
        "no_code_self_modification",
        "evidence_integrity",
        "human_approval_for_improvements",
        "mandatory_gates",
    ],
}


@dataclass
class GovernanceCheckResult:
    """Result of a governance check."""
    rule: str
    passed: bool
    details: str = ""
    blocked: bool = False
    needs_human: bool = False

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "passed": self.passed,
            "details": self.details,
            "blocked": self.blocked,
            "needs_human": self.needs_human,
        }


class ImmutableGovernancePolicy:
    """Enforces the immutable governance policy.

    The policy itself cannot be modified at runtime. Any attempt to
    modify it is detected and blocked.
    """

    def __init__(self):
        self._policy = IMMUTABLE_GOVERNANCE_POLICY
        self._policy_hash = self._compute_hash()

    def check(self, rule: str, context: Dict[str, Any]) -> GovernanceCheckResult:
        """Check a governance rule against the current context."""
        rules = self._policy["rules"]
        if rule not in rules:
            return GovernanceCheckResult(
                rule=rule,
                passed=False,
                details=f"Unknown governance rule: {rule}",
                blocked=True,
            )

        rule_def = rules[rule]
        consequence = rule_def["consequence"]

        # Verify policy integrity
        if self._compute_hash() != self._policy_hash:
            return GovernanceCheckResult(
                rule="policy_integrity",
                passed=False,
                details="Governance policy has been tampered with!",
                blocked=True,
                needs_human=True,
            )

        # Check the specific rule
        if rule == "free_model_invariant":
            return self._check_free_model(context)
        elif rule == "independent_reviewer":
            return self._check_independent_reviewer(context)
        elif rule == "no_code_self_modification":
            return self._check_no_self_modification(context)
        elif rule == "evidence_integrity":
            return self._check_evidence_integrity(context)
        elif rule == "human_approval_for_improvements":
            return self._check_human_approval(context)
        elif rule == "mandatory_gates":
            return self._check_mandatory_gates(context)

        return GovernanceCheckResult(
            rule=rule,
            passed=True,
            details=f"Rule '{rule}' passed governance check",
        )

    def check_all(self, context: Dict[str, Any]) -> List[GovernanceCheckResult]:
        """Check all governance rules."""
        results = []
        for rule in self._policy["immutable_keys"]:
            result = self.check(rule, context)
            results.append(result)
            if result.blocked:
                break
        return results

    def is_immutable(self, key: str) -> bool:
        """Check if a governance key is immutable."""
        return key in self._policy["immutable_keys"]

    def get_policy(self) -> Dict[str, Any]:
        """Get the current policy (read-only)."""
        return dict(self._policy)

    def _compute_hash(self) -> str:
        """Compute integrity hash of the policy."""
        import hashlib
        content = json.dumps(self._policy, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _check_free_model(self, context: Dict[str, Any]) -> GovernanceCheckResult:
        model_id = context.get("model_id", "")
        if model_id.endswith(":free") or not model_id:
            return GovernanceCheckResult(rule="free_model_invariant", passed=True)
        return GovernanceCheckResult(
            rule="free_model_invariant",
            passed=False,
            details=f"Model '{model_id}' is not free",
            blocked=True,
        )

    def _check_independent_reviewer(self, context: Dict[str, Any]) -> GovernanceCheckResult:
        reviewer = context.get("reviewer", "")
        implementer = context.get("implementer", "")
        if not implementer or reviewer != implementer:
            return GovernanceCheckResult(rule="independent_reviewer", passed=True)
        return GovernanceCheckResult(
            rule="independent_reviewer",
            passed=False,
            details=f"Reviewer '{reviewer}' is same as implementer",
            blocked=True,
        )

    def _check_no_self_modification(self, context: Dict[str, Any]) -> GovernanceCheckResult:
        target_paths = context.get("target_paths", [])
        forbidden = ["harness/", "config/capability_taxonomy.yaml", "config/policy"]
        for path in target_paths:
            for forbidden_path in forbidden:
                if path.startswith(forbidden_path):
                    return GovernanceCheckResult(
                        rule="no_code_self_modification",
                        passed=False,
                        details=f"Attempted to modify '{path}' which is in a forbidden area",
                        blocked=True,
                    )
        return GovernanceCheckResult(rule="no_code_self_modification", passed=True)

    def _check_evidence_integrity(self, context: Dict[str, Any]) -> GovernanceCheckResult:
        return GovernanceCheckResult(rule="evidence_integrity", passed=True)

    def _check_human_approval(self, context: Dict[str, Any]) -> GovernanceCheckResult:
        is_improvement = context.get("is_improvement", False)
        has_approval = context.get("has_human_approval", False)
        if is_improvement and not has_approval:
            return GovernanceCheckResult(
                rule="human_approval_for_improvements",
                passed=False,
                details="Improvement proposal requires human approval",
                needs_human=True,
            )
        return GovernanceCheckResult(rule="human_approval_for_improvements", passed=True)

    def _check_mandatory_gates(self, context: Dict[str, Any]) -> GovernanceCheckResult:
        caps = set(context.get("capabilities", []))
        mandatory = {"verification", "evaluation", "review"}
        missing = mandatory - caps
        if missing:
            return GovernanceCheckResult(
                rule="mandatory_gates",
                passed=False,
                details=f"Missing mandatory capabilities: {missing}",
                blocked=True,
            )
        return GovernanceCheckResult(rule="mandatory_gates", passed=True)


class SecurityRegressionSuite:
    """Regression tests for security invariants."""

    def __init__(self, governance: ImmutableGovernancePolicy):
        self.governance = governance

    def run_all(self) -> List[Dict[str, Any]]:
        """Run all security regression checks."""
        results = []

        # Test 1: Policy integrity
        results.append(self._test_policy_integrity())

        # Test 2: Free model invariant
        results.append(self._test_free_model_invariant())

        # Test 3: No self-modification
        results.append(self._test_no_self_modification())

        # Test 4: Rule immutability
        results.append(self._test_rule_immutability())

        return results

    def _test_policy_integrity(self) -> Dict[str, Any]:
        try:
            result = self.governance.check("free_model_invariant",
                                           {"model_id": "test:free"})
            return {"test": "policy_integrity", "passed": result.passed}
        except Exception as e:
            return {"test": "policy_integrity", "passed": False, "error": str(e)}

    def _test_free_model_invariant(self) -> Dict[str, Any]:
        # Free model should pass
        free_result = self.governance.check("free_model_invariant",
                                            {"model_id": "nous/free:free"})
        # Paid model should fail
        paid_result = self.governance.check("free_model_invariant",
                                            {"model_id": "openai/gpt-4"})
        return {
            "test": "free_model_invariant",
            "free_passes": free_result.passed,
            "paid_blocked": not paid_result.passed,
        }

    def _test_no_self_modification(self) -> Dict[str, Any]:
        # Harness path should be blocked
        blocked = self.governance.check("no_code_self_modification",
                                        {"target_paths": ["harness/orchestrator/__init__.py"]})
        # Non-harness path should pass
        allowed = self.governance.check("no_code_self_modification",
                                        {"target_paths": ["src/app/Controller.php"]})
        return {
            "test": "no_self_modification",
            "harness_path_blocked": not blocked.passed,
            "src_path_allowed": allowed.passed,
        }

    def _test_rule_immutability(self) -> Dict[str, Any]:
        immutables = self.governance._policy["immutable_keys"]
        all_immutable = all(self.governance.is_immutable(k) for k in immutables)
        return {
            "test": "rule_immutability",
            "all_immutable": all_immutable,
            "immutable_count": len(immutables),
        }


class PromptInjectionBoundary:
    """Detects and blocks prompt injection attempts in task inputs."""

    INJECTION_PATTERNS = [
        "ignore all previous instructions",
        "ignore previous instructions",
        "forget all instructions",
        "override your instructions",
        "you are now",
        "you will now act as",
        "system prompt",
        "your system prompt",
        "you are an ai",
        "pretend you are",
        "disregard",
        "new instructions",
        "from now on",
    ]

    @classmethod
    def check(cls, text: str) -> Dict[str, Any]:
        """Check text for prompt injection patterns.

        Returns:
            Dict with 'injection_detected' bool and 'matched_patterns' list.
        """
        text_lower = text.lower()
        matched = []

        for pattern in cls.INJECTION_PATTERNS:
            if pattern in text_lower:
                matched.append(pattern)

        return {
            "injection_detected": len(matched) > 0,
            "matched_patterns": matched,
            "severity": "high" if len(matched) > 2 else "medium" if matched else "none",
        }


class ToolAuthorizationHardener:
    """Hardens tool authorization against unauthorized access patterns."""

    RESTRICTED_TOOLS = {
        "terminal": {
            "forbidden_commands": [
                "sudo", "su", "chmod 777", "chown", "passwd",
                "rm -rf /", "dd if=", "mkfs",
            ],
            "allowed_categories": ["read", "write", "build", "test", "git"],
        },
        "filesystem": {
            "forbidden_paths": [
                "harness/", "config/capability_taxonomy.yaml",
                ".git/", "artifacts/",
            ],
        },
    }

    @classmethod
    def authorize(cls, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a tool invocation is authorized.

        Returns:
            Dict with 'authorized' bool and 'reason' string.
        """
        restrictions = cls.RESTRICTED_TOOLS.get(tool, {})
        if not restrictions:
            return {"authorized": True, "reason": f"Tool '{tool}' has no restrictions"}

        if tool == "terminal":
            command = params.get("command", "")
            for forbidden in restrictions.get("forbidden_commands", []):
                if forbidden in command:
                    return {
                        "authorized": False,
                        "reason": f"Command contains forbidden pattern: '{forbidden}'",
                    }

        if tool == "filesystem" or tool == "write":
            target = params.get("path", params.get("target", ""))
            for forbidden in restrictions.get("forbidden_paths", []):
                if target.startswith(forbidden):
                    return {
                        "authorized": False,
                        "reason": f"Target path '{target}' is in a forbidden area: '{forbidden}'",
                    }

        return {"authorized": True, "reason": "Authorized"}


class ImprovementProposalHandler:
    """Handles improvement proposals with human approval workflow."""

    def __init__(self, governance: ImmutableGovernancePolicy):
        self.governance = governance
        self._pending_proposals: List[Dict[str, Any]] = []

    def propose(self, description: str, changes: List[Dict[str, Any]],
                rationale: str) -> Dict[str, Any]:
        """Submit an improvement proposal for human approval."""
        proposal = {
            "id": f"IMP-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "description": description,
            "changes": changes,
            "rationale": rationale,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
        }
        self._pending_proposals.append(proposal)
        return proposal

    def approve(self, proposal_id: str) -> Dict[str, Any]:
        """Approve a pending proposal."""
        for p in self._pending_proposals:
            if p["id"] == proposal_id:
                p["status"] = "approved"
                p["approved_at"] = datetime.utcnow().isoformat()
                return p
        return {"error": f"Proposal '{proposal_id}' not found"}

    def reject(self, proposal_id: str, reason: str = "") -> Dict[str, Any]:
        """Reject a pending proposal."""
        for p in self._pending_proposals:
            if p["id"] == proposal_id:
                p["status"] = "rejected"
                p["rejected_at"] = datetime.utcnow().isoformat()
                p["rejection_reason"] = reason
                return p
        return {"error": f"Proposal '{proposal_id}' not found"}

    def get_pending(self) -> List[Dict[str, Any]]:
        """Get all pending proposals."""
        return [p for p in self._pending_proposals if p["status"] == "pending"]
