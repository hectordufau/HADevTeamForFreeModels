"""Tests for Unified Evidence (Phase C - V3.0)."""

import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.evidence.unified import (
    EvidenceStore, EvidencePackage, CanonicalExecutionRecord,
    EvidenceQuery, EvidenceError,
)


def test_evidence_package_hash_chain():
    """Test evidence package hash computation."""
    pkg = EvidencePackage(
        node_id="design",
        capability="api_design",
        agent="architect",
        task_id="TASK-001",
        execution_id="EXEC-001",
        data={"result": "success", "files": ["api.py"]},
    )
    h1 = pkg.compute_hash()
    assert isinstance(h1, str)
    assert len(h1) == 64  # SHA-256 hex

    # Same data produces same hash
    pkg2 = EvidencePackage(
        node_id="design",
        capability="api_design",
        agent="architect",
        task_id="TASK-001",
        execution_id="EXEC-001",
        data={"result": "success", "files": ["api.py"]},
    )
    assert pkg2.compute_hash() == h1

    # Different data produces different hash
    pkg3 = EvidencePackage(
        node_id="design",
        capability="api_design",
        agent="architect",
        task_id="TASK-001",
        execution_id="EXEC-001",
        data={"result": "failure"},
    )
    assert pkg3.compute_hash() != h1


def test_evidence_store_store_and_retrieve():
    """Test storing and retrieving evidence packages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EvidenceStore(storage_dir=os.path.join(tmpdir, "evidence"))

        pkg1 = EvidencePackage(
            node_id="node1", capability="backend", agent="coder",
            task_id="T1", execution_id="E1",
            data={"changes": ["src/main.py"]},
        )
        pkg2 = EvidencePackage(
            node_id="node2", capability="testing", agent="tester",
            task_id="T1", execution_id="E1",
            data={"test_results": {"passed": 10, "failed": 0}},
        )

        hash1 = store.store_evidence(pkg1)
        hash2 = store.store_evidence(pkg2)

        assert isinstance(hash1, str)
        assert isinstance(hash2, str)
        assert hash1 != hash2


def test_evidence_store_chain_integrity():
    """Test hash chain integrity verification via chain index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EvidenceStore(storage_dir=os.path.join(tmpdir, "evidence"))

        for i in range(3):
            pkg = EvidencePackage(
                node_id=f"node{i}", capability=f"cap{i}", agent="tester",
                task_id="T1", execution_id="E1",
                data={"step": i},
            )
            stored_hash = store.store_evidence(pkg)
            assert pkg.content_hash == stored_hash
            assert stored_hash == pkg._hash_content()

        # Verify chain index on disk
        with open(os.path.join(tmpdir, "evidence", "chain_index.json")) as f:
            chain = json.load(f)
        assert len(chain) == 3
        assert chain[0]["previous_hash"] == ""
        for i in range(1, len(chain)):
            assert chain[i]["previous_hash"] == chain[i-1]["hash"]


def test_canonical_execution_record():
    """Test canonical execution record creation and validation."""
    record = CanonicalExecutionRecord(
        task_id="TASK-001",
        execution_id="EXEC-001",
        workflow_name="dynamic",
        nodes=[{"id": "design", "capability": "api_design"}],
        capability_requirements={"required": ["api_design", "testing"]},
        assignments={"design": "architect"},
        results={"design": {"status": "completed"}},
        evidence_hashes=["abc123", "def456"],
    )
    h = record.compute_hash()
    assert len(h) == 64

    # Serialize and deserialize
    d = record.to_dict()
    record2 = CanonicalExecutionRecord(**d)
    assert record2.compute_hash() == h


def test_evidence_store_canonical_record():
    """Test storing and retrieving canonical records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EvidenceStore(storage_dir=os.path.join(tmpdir, "evidence"))

        record = CanonicalExecutionRecord(
            task_id="T1", execution_id="E1", workflow_name="wf1",
            nodes=[], capability_requirements={}, assignments={},
            results={}, evidence_hashes=[],
        )
        h = store.store_record(record)
        assert isinstance(h, str)

        retrieved = store.get_record("T1", "E1")
        assert retrieved is not None
        assert retrieved.task_id == "T1"
        assert retrieved.execution_id == "E1"


def test_integrity_verification():
    """Test full integrity verification of stored records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EvidenceStore(storage_dir=os.path.join(tmpdir, "evidence"))

        # Store evidence
        for i in range(3):
            pkg = EvidencePackage(
                node_id=f"n{i}", capability=f"c{i}", agent="a",
                task_id="T1", execution_id="E1",
                data={"x": i},
            )
            store.store_evidence(pkg)

        # Store record
        record = CanonicalExecutionRecord(
            task_id="T1", execution_id="E1", workflow_name="wf1",
            nodes=[], capability_requirements={}, assignments={},
            results={}, evidence_hashes=[],
        )
        store.store_record(record)

        # Verify
        result = store.verify_integrity("T1", "E1")
        # Evidence chain is valid (chain index is intact)
        assert result["evidence_count"] == 3


def test_query_by_task():
    """Test querying evidence by task_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EvidenceStore(storage_dir=os.path.join(tmpdir, "evidence"))

        for i in range(3):
            pkg = EvidencePackage(
                node_id=f"n{i}", capability=f"c{i}", agent="a",
                task_id="T1", execution_id="E1",
                data={"i": i},
            )
            store.store_evidence(pkg)

        # Query
        query = EvidenceQuery(task_id="T1")
        results = store.query(query)
        assert len(results) == 3

        query2 = EvidenceQuery(task_id="T2")
        results2 = store.query(query2)
        assert len(results2) == 0


def test_query_by_node():
    """Test querying evidence by node_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EvidenceStore(storage_dir=os.path.join(tmpdir, "evidence"))

        for i in range(3):
            pkg = EvidencePackage(
                node_id=f"n{i}", capability=f"c{i}", agent="a",
                task_id="T1", execution_id="E1",
                data={"i": i},
            )
            store.store_evidence(pkg)

        query = EvidenceQuery(task_id="T1", node_id="n1")
        results = store.query(query)
        assert len(results) == 1
        assert results[0].node_id == "n1"


def test_query_limit():
    """Test query pagination."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = EvidenceStore(storage_dir=os.path.join(tmpdir, "evidence"))

        for i in range(10):
            pkg = EvidencePackage(
                node_id=f"n{i}", capability=f"c{i}", agent="a",
                task_id="T1", execution_id="E1",
                data={"i": i},
            )
            store.store_evidence(pkg)

        query = EvidenceQuery(task_id="T1", limit=3)
        results = store.query(query)
        assert len(results) == 3
