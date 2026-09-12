# harness/evidence/unified.py — Canonical Execution Record & Evidence Store (V3.0)
"""
Unified evidence system: canonical execution records, evidence store with
hash-chain integrity, and query API.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
import threading

from ..learning.context import ExperimentContext
from ..learning.isolation import check_test_protection, namespace_path


class EvidenceError(Exception):
    """Raised on evidence store errors."""


@dataclass
class CanonicalExecutionRecord:
    """The canonical record of a single execution, immutable after creation."""
    task_id: str
    execution_id: str
    workflow_name: str
    nodes: List[Dict[str, Any]]
    capability_requirements: Dict[str, Any]
    assignments: Dict[str, Any]
    results: Dict[str, Any]
    evidence_hashes: List[str]
    # V3.2 ExperimentContext fields (optional for backward compatibility)
    validation_run_id: str = ""
    mode: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    previous_hash: str = ""

    def _hash_content(self) -> str:
        """Compute hash based on content fields only (excluding timestamp)."""
        d = {
            "task_id": self.task_id,
            "execution_id": self.execution_id,
            "workflow_name": self.workflow_name,
            "nodes": self.nodes,
            "capability_requirements": self.capability_requirements,
            "assignments": self.assignments,
            "results": self.results,
            "evidence_hashes": self.evidence_hashes,
            "previous_hash": self.previous_hash,
        }
        content = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def compute_hash(self) -> str:
        """Compute the hash chain of this record (excludes timestamp for determinism)."""
        return self._hash_content()

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "execution_id": self.execution_id,
            "workflow_name": self.workflow_name,
            "nodes": self.nodes,
            "capability_requirements": self.capability_requirements,
            "assignments": self.assignments,
            "results": self.results,
            "evidence_hashes": self.evidence_hashes,
            "validation_run_id": self.validation_run_id,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
        }


@dataclass
class EvidencePackage:
    """A package of evidence collected during a node execution."""
    node_id: str
    capability: str
    agent: str
    task_id: str
    execution_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    # V3.2 ExperimentContext fields (optional for backward compatibility)
    validation_run_id: str = ""
    benchmark_id: str = ""
    mode: str = ""
    content_hash: str = ""
    previous_hash: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def _hash_content(self) -> str:
        """Compute hash based on content fields only (excluding timestamp and content_hash)."""
        d = {
            "node_id": self.node_id,
            "capability": self.capability,
            "agent": self.agent,
            "task_id": self.task_id,
            "execution_id": self.execution_id,
            "data": self.data,
            "previous_hash": self.previous_hash,
        }
        content = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def compute_hash(self) -> str:
        """Compute content hash for integrity verification (excludes timestamp for determinism)."""
        return self._hash_content()

    def store_dict(self) -> dict:
        """Dictionary representation for persistent storage (includes all fields)."""
        return {
            "node_id": self.node_id,
            "capability": self.capability,
            "agent": self.agent,
            "task_id": self.task_id,
            "execution_id": self.execution_id,
            "data": self.data,
            "validation_run_id": self.validation_run_id,
            "benchmark_id": self.benchmark_id,
            "mode": self.mode,
            "content_hash": self.content_hash,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
        }


@dataclass
class EvidenceQuery:
    """Query parameters for retrieving evidence."""
    task_id: Optional[str] = None
    execution_id: Optional[str] = None
    node_id: Optional[str] = None
    capability: Optional[str] = None
    limit: int = 100
    offset: int = 0


class EvidenceStore:
    """Persistent evidence store with hash-chain integrity.

    Every evidence package and execution record is chained to the previous
    one via SHA-256 hashes, creating an immutable audit trail.
    """

    def __init__(self, storage_dir: str = "",
                 experiment_context: Optional[ExperimentContext] = None):
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "artifacts", "evidence_v3"
        )
        self._lock = threading.Lock()
        os.makedirs(self.storage_dir, exist_ok=True)
        self.experiment_context = experiment_context

    def store_evidence(self, package: EvidencePackage,
                       experiment_context: Optional[ExperimentContext] = None) -> str:
        """Store an evidence package in namespaced directory and return its hash chain position."""
        if experiment_context is None:
            experiment_context = self.experiment_context
        # Apply experiment context fields if provided
        if experiment_context:
            package.validation_run_id = experiment_context.validation_run_id
            package.benchmark_id = experiment_context.benchmark_id
            package.mode = experiment_context.mode
        # Compute hash chain first
        previous_hash = self._get_latest_hash()
        package.previous_hash = previous_hash

        # Compute content hash (after setting previous_hash but before storing)
        content_hash = package._hash_content()
        package.content_hash = content_hash

        with self._lock:
            store_dir = namespace_path(
                    "evidence",
                    experiment_context=experiment_context
                ) if experiment_context else self.storage_dir
            path = self._evidence_path(package.task_id, package.execution_id, package.node_id, store_dir)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(package.store_dict(), f, indent=2, default=str)

            # Update chain index
            chain_entry = {
                "hash": package.content_hash,
                "previous_hash": previous_hash,
                "task_id": package.task_id,
                "execution_id": package.execution_id,
                "node_id": package.node_id,
                "timestamp": package.timestamp,
            }
            self._append_chain(chain_entry, store_dir)

        return package.content_hash

    def store_record(self, record: CanonicalExecutionRecord,
                     experiment_context: Optional[ExperimentContext] = None) -> str:
        """Store a canonical execution record in namespaced directory."""
        if experiment_context is None:
            experiment_context = self.experiment_context
        # Apply experiment context fields if provided
        if experiment_context:
            record.validation_run_id = experiment_context.validation_run_id
            record.mode = experiment_context.mode
        # Compute hash
        record_hash = record._hash_content()

        # Chain to previous record
        prev_hash = self._get_latest_record_hash()
        record.previous_hash = prev_hash

        store_dir = namespace_path(
            "evidence",
            experiment_context=experiment_context
        ) if experiment_context else self.storage_dir
        path = self._record_path(record.task_id, record.execution_id, store_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(record.to_dict(), f, indent=2, default=str)

        return record_hash

    def verify_integrity(self, task_id: str, execution_id: str) -> Dict[str, Any]:
        """Verify the hash-chain integrity of an execution's evidence."""
        record = self.get_record(task_id, execution_id)
        if not record:
            return {"valid": False, "error": "Record not found"}

        # Verify record hash
        verification = record._hash_content()

        # Verify chain of evidence packages
        evidence_list = self.get_evidence(task_id, execution_id)
        previous = ""
        for pkg in evidence_list:
            computed = pkg._hash_content()
            if pkg.previous_hash != previous:
                return {
                    "valid": False,
                    "error": f"Hash chain broken at {pkg.node_id}: expected previous {previous}, got {pkg.previous_hash}",
                }
            previous = computed

        return {
            "valid": True,
            "record_hash": verification,
            "evidence_count": len(evidence_list),
        }

    def get_record(self, task_id: str, execution_id: str,
                   experiment_context: Optional[ExperimentContext] = None) -> Optional[CanonicalExecutionRecord]:
        """Retrieve a canonical execution record, filtered by context."""
        if experiment_context is None:
            experiment_context = self.experiment_context
        store_dir = namespace_path(
            "evidence",
            experiment_context=experiment_context
        ) if experiment_context else self.storage_dir
        path = self._record_path(task_id, execution_id, store_dir)
        if not os.path.exists(path):
            # Fallback: try non-namespaced if no context
            if experiment_context is None:
                path = self._record_path(task_id, execution_id, self.storage_dir)
                if not os.path.exists(path):
                    return None
            else:
                return None
        with open(path) as f:
            data = json.load(f)
        return CanonicalExecutionRecord(**data)

    def get_evidence(self, task_id: str, execution_id: str,
                     experiment_context: Optional[ExperimentContext] = None) -> List[EvidencePackage]:
        """Retrieve all evidence packages for an execution, filtered by context."""
        if experiment_context is None:
            experiment_context = self.experiment_context
        store_dir = namespace_path(
            "evidence",
            experiment_context=experiment_context
        ) if experiment_context else self.storage_dir
        dir_path = os.path.join(store_dir, "packages", task_id, execution_id)
        if not os.path.exists(dir_path):
            return []
        packages = []
        for fname in sorted(os.listdir(dir_path)):
            if fname.endswith(".json"):
                with open(os.path.join(dir_path, fname)) as f:
                    data = json.load(f)
                # CanonicalExecutionRecord uses store_dict format (from disk)
                packages.append(EvidencePackage(**data))
        return packages

    def query(self, query: EvidenceQuery,
              experiment_context: Optional[ExperimentContext] = None) -> List[EvidencePackage]:
        """Query evidence by criteria, filtered by context."""
        if experiment_context is None:
            experiment_context = self.experiment_context
        results = []

        store_dir = namespace_path(
            "evidence",
            experiment_context=experiment_context
        ) if experiment_context else self.storage_dir
        base_dir = os.path.join(store_dir, "packages")

        if query.task_id:
            task_dir = os.path.join(base_dir, query.task_id)
            if not os.path.exists(task_dir):
                return results
            for exec_id in os.listdir(task_dir):
                if query.execution_id and exec_id != query.execution_id:
                    continue
                exec_dir = os.path.join(task_dir, exec_id)
                if not os.path.isdir(exec_dir):
                    continue
                for fname in sorted(os.listdir(exec_dir)):
                    if not fname.endswith(".json"):
                        continue
                    with open(os.path.join(exec_dir, fname)) as f:
                        data = json.load(f)
                    pkg = EvidencePackage(**data)
                    if query.node_id and pkg.node_id != query.node_id:
                        continue
                    if query.capability and pkg.capability != query.capability:
                        continue
                    results.append(pkg)

        return results[query.offset:query.offset + query.limit]

    def query_global(self, query: EvidenceQuery) -> List[EvidencePackage]:
        """Query evidence globally across all namespaces (no filtering)."""
        results = []
        global_base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "artifacts", "v3.2"
        )

        # Check all v3.2 namespaced directories
        if os.path.exists(global_base):
            for run_id in os.listdir(global_base):
                run_dir = os.path.join(global_base, run_id)
                if not os.path.isdir(run_dir):
                    continue
                for mode in os.listdir(run_dir):
                    if mode not in ("COLD", "LEARNED", "TEST"):
                        continue
                    packages_dir = os.path.join(run_dir, mode, "evidence", "packages")
                    self._query_packages_dir(packages_dir, query, results)

        # Also check V3.0/3.1 flat directory
        packages_dir = os.path.join(self.storage_dir, "packages")
        self._query_packages_dir(packages_dir, query, results)

        return results[query.offset:query.offset + query.limit]

    def _query_packages_dir(self, packages_dir: str, query: EvidenceQuery,
                            results: List[EvidencePackage]):
        """Helper to query a single packages directory."""
        if not os.path.exists(packages_dir):
            return
        if query.task_id:
            task_dir = os.path.join(packages_dir, query.task_id)
            if not os.path.exists(task_dir):
                return
            for exec_id in os.listdir(task_dir):
                if query.execution_id and exec_id != query.execution_id:
                    continue
                exec_dir = os.path.join(task_dir, exec_id)
                if not os.path.isdir(exec_dir):
                    continue
                for fname in sorted(os.listdir(exec_dir)):
                    if not fname.endswith(".json"):
                        continue
                    with open(os.path.join(exec_dir, fname)) as f:
                        data = json.load(f)
                    pkg = EvidencePackage(**data)
                    if query.node_id and pkg.node_id != query.node_id:
                        continue
                    if query.capability and pkg.capability != query.capability:
                        continue
                    results.append(pkg)

    def _evidence_path(self, task_id: str, execution_id: str, node_id: str,
                       store_dir: Optional[str] = None) -> str:
        base = store_dir or self.storage_dir
        return os.path.join(base, "packages", task_id, execution_id, f"{node_id}.json")

    def _record_path(self, task_id: str, execution_id: str,
                     store_dir: Optional[str] = None) -> str:
        base = store_dir or self.storage_dir
        return os.path.join(base, "records", f"{task_id}_{execution_id}.json")

    def _get_latest_hash(self, store_dir: Optional[str] = None) -> str:
        """Get the latest hash from the chain index."""
        base = store_dir or self.storage_dir
        chain_path = os.path.join(base, "chain_index.json")
        if not os.path.exists(chain_path):
            return ""
        with open(chain_path) as f:
            chain = json.load(f)
        return chain[-1]["hash"] if chain else ""

    def _get_latest_record_hash(self, store_dir: Optional[str] = None) -> str:
        """Get the latest record hash."""
        base = store_dir or self.storage_dir
        records_dir = os.path.join(base, "records")
        if not os.path.exists(records_dir):
            return ""
        record_files = sorted(os.listdir(records_dir))
        if not record_files:
            return ""
        with open(os.path.join(records_dir, record_files[-1])) as f:
            data = json.load(f)
        # Recompute hash deterministically
        record = CanonicalExecutionRecord(**data)
        return record._hash_content()

    def _append_chain(self, entry: dict, store_dir: Optional[str] = None):
        """Append to the chain index."""
        base = store_dir or self.storage_dir
        chain_path = os.path.join(base, "chain_index.json")
        chain = []
        if os.path.exists(chain_path):
            with open(chain_path) as f:
                chain = json.load(f)
        chain.append(entry)
        with open(chain_path, "w") as f:
            json.dump(chain, f, indent=2, default=str)
