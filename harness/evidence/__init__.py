# harness/evidence/__init__.py — Evidence Collector
"""
Collects and structures evidence for execution results.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import os
import json


@dataclass
class Evidence:
    """Evidence record for a single execution."""

    task_id: str
    changed_files: List[str] = field(default_factory=list)
    commands_executed: List[Dict[str, Any]] = field(default_factory=list)
    test_results: Dict[str, Any] = field(default_factory=dict)
    verification_results: Dict[str, Any] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    collected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "changed_files": self.changed_files,
            "commands_executed": self.commands_executed,
            "test_results": self.test_results,
            "verification_results": self.verification_results,
            "execution_metadata": self.execution_metadata,
            "collected_at": self.collected_at,
        }


class EvidenceCollector:
    """Collects and persists execution evidence."""

    def __init__(self, storage_dir: str = "artifacts/evidence"):
        self.storage_dir = storage_dir

    def collect(self, execution_result: Any, task_id: str) -> Evidence:
        """Build an Evidence record from an execution result."""
        evidence = Evidence(
            task_id=task_id,
            changed_files=execution_result.changes if hasattr(execution_result, "changes") else [],
            execution_metadata={
                "status": execution_result.status if hasattr(execution_result, "status") else "unknown",
                "claims": execution_result.claims if hasattr(execution_result, "claims") else [],
            },
        )
        return evidence

    def persist(self, evidence: Evidence):
        """Save evidence to disk as JSON."""
        task_dir = os.path.join(self.storage_dir, evidence.task_id)
        os.makedirs(task_dir, exist_ok=True)
        path = os.path.join(task_dir, "evidence.json")
        with open(path, "w") as f:
            json.dump(evidence.to_dict(), f, indent=2)
        return path
