# harness/verification/__init__.py — Verification Engine
"""
Pluggable verification of implementation results.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


class VerificationError(Exception):
    """Raised on verification failures."""


@dataclass
class CheckResult:
    name: str
    status: str  # passed, failed, skipped, error
    passed: int = 0
    failed: int = 0
    output: str = ""
    error: Optional[str] = None


@dataclass
class VerificationResult:
    status: str = "pending"  # pending, passed, failed
    checks: Dict[str, CheckResult] = field(default_factory=dict)

    def all_passed(self) -> bool:
        return self.status == "passed" and all(
            c.status == "passed" for c in self.checks.values()
        )


class Verifier:
    """Base class for pluggable verifiers."""

    def verify(self, task: Any, workspace: str) -> CheckResult:
        raise NotImplementedError


class UnitTestVerifier(Verifier):
    def verify(self, task: Any, workspace: str) -> CheckResult:
        return CheckResult(name="unit_tests", status="skipped",
                           output="Not yet implemented — requires test runner integration")


class BuildVerifier(Verifier):
    def verify(self, task: Any, workspace: str) -> CheckResult:
        return CheckResult(name="build", status="skipped",
                           output="Not yet implemented — requires build system integration")


class LintVerifier(Verifier):
    def verify(self, task: Any, workspace: str) -> CheckResult:
        return CheckResult(name="lint", status="skipped",
                           output="Not yet implemented — requires linter integration")


class AcceptanceVerifier(Verifier):
    def verify(self, task: Any, workspace: str) -> CheckResult:
        return CheckResult(name="acceptance", status="skipped",
                           output="Not yet implemented — requires acceptance test runner")


class IntegrationTestVerifier(Verifier):
    def verify(self, task: Any, workspace: str) -> CheckResult:
        return CheckResult(name="integration_tests", status="skipped",
                           output="Not yet implemented — requires integration test runner")


class SecurityVerifier(Verifier):
    def verify(self, task: Any, workspace: str) -> CheckResult:
        return CheckResult(name="security", status="skipped",
                           output="Not yet implemented — requires security scanner integration")


class VerificationEngine:
    """Orchestrates verification checks."""

    def __init__(self):
        self._verifiers: Dict[str, Verifier] = {}

    def register(self, name: str, verifier: Verifier):
        self._verifiers[name] = verifier

    def verify(self, task: Any, workspace: str, checks: Optional[List[str]] = None) -> VerificationResult:
        result = VerificationResult()
        to_run = checks or list(self._verifiers.keys())

        for check_name in to_run:
            verifier = self._verifiers.get(check_name)
            if not verifier:
                result.checks[check_name] = CheckResult(
                    name=check_name, status="error",
                    error=f"No verifier registered for {check_name}"
                )
                continue
            try:
                check_result = verifier.verify(task, workspace)
                result.checks[check_name] = check_result
            except Exception as e:
                result.checks[check_name] = CheckResult(
                    name=check_name, status="error", error=str(e)
                )

        # Overall status
        if any(c.status == "failed" for c in result.checks.values()):
            result.status = "failed"
        elif all(c.status == "passed" for c in result.checks.values()):
            result.status = "passed"
        else:
            result.status = "pending"  # some skipped or errored

        return result
