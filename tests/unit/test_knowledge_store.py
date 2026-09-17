# tests/unit/test_knowledge_store.py — Phase 2: KnowledgeStore Persistence, Reload, Indexes, Integrity
"""Tests for KnowledgeStore: SQLite persistence, reload, indexes, and integrity verification."""

import json
import os
import sys
import tempfile
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.knowledge.records import EngineeringRecord, RecordError, create_record
from harness.knowledge.provenance import Provenance, AUTHORITY_PROPOSED, AUTHORITY_ACCEPTED
from harness.knowledge.lifecycle import get_initial_state
from harness.knowledge.store import (
    KnowledgeStore,
    StoreError,
    IntegrityError,
    RecordNotFoundError,
    DuplicateRecordError,
)


def make_provenance(**kwargs):
    defaults = {"author": "test_user", "source": "human"}
    defaults.update(kwargs)
    return Provenance(**defaults)


def make_record(**kwargs):
    defaults = {
        "record_id": "PRD-001",
        "record_type": "PRD",
        "title": "Test Record",
        "description": "A test record",
        "status": "draft",
        "authority": "proposed",
        "provenance": make_provenance(),
    }
    defaults.update(kwargs)
    return EngineeringRecord(**defaults)


@pytest.fixture
def tmp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def store(tmp_db):
    """Create a KnowledgeStore with a temporary database."""
    return KnowledgeStore(tmp_db)


# ──────────────────────────────────────────────────────────────────────
# Persistence: save and retrieve records
# ──────────────────────────────────────────────────────────────────────

class TestPersistence:
    """Test basic save and retrieve operations."""

    def test_save_and_get_record(self, store):
        record = make_record()
        store.save(record)
        retrieved = store.get("PRD-001")
        assert retrieved is not None
        assert retrieved.record_id == "PRD-001"
        assert retrieved.title == "Test Record"

    def test_get_nonexistent_record(self, store):
        with pytest.raises(RecordNotFoundError):
            store.get("NONEXISTENT")

    def test_save_multiple_records(self, store):
        r1 = make_record(record_id="PRD-001", record_type="PRD")
        r2 = make_record(record_id="NFR-001", record_type="NFR")
        r3 = make_record(record_id="ADR-001", record_type="ADR", status="proposed")
        store.save(r1)
        store.save(r2)
        store.save(r3)
        assert store.count() == 3
        assert store.get("PRD-001").record_type == "PRD"
        assert store.get("NFR-001").record_type == "NFR"
        assert store.get("ADR-001").record_type == "ADR"

    def test_save_duplicate_raises(self, store):
        record = make_record()
        store.save(record)
        with pytest.raises(DuplicateRecordError):
            store.save(record, update=False)

    def test_update_existing_record(self, store):
        record = make_record()
        store.save(record)
        record.title = "Updated Title"
        store.save(record)
        retrieved = store.get("PRD-001")
        assert retrieved.title == "Updated Title"

    def test_delete_record(self, store):
        record = make_record()
        store.save(record)
        assert store.count() == 1
        store.delete("PRD-001")
        assert store.count() == 0
        with pytest.raises(RecordNotFoundError):
            store.get("PRD-001")

    def test_delete_nonexistent_raises(self, store):
        with pytest.raises(RecordNotFoundError):
            store.delete("NONEXISTENT")


# ──────────────────────────────────────────────────────────────────────
# Reload: persist to disk and reload
# ──────────────────────────────────────────────────────────────────────

class TestReload:
    """Test persistence across store instances."""

    def test_reload_from_disk(self, tmp_db):
        store1 = KnowledgeStore(tmp_db)
        record = make_record()
        store1.save(record)
        store1.close()

        store2 = KnowledgeStore(tmp_db)
        retrieved = store2.get("PRD-001")
        assert retrieved is not None
        assert retrieved.record_id == "PRD-001"
        assert retrieved.title == "Test Record"
        store2.close()

    def test_reload_preserves_all_fields(self, tmp_db):
        store1 = KnowledgeStore(tmp_db)
        record = make_record(
            tags=["test", "v3.3"],
            version=3,
            superseded_by="PRD-002",
        )
        record.add_relationship("NFR-001", "REQUIRES")
        store1.save(record)
        store1.close()

        store2 = KnowledgeStore(tmp_db)
        retrieved = store2.get("PRD-001")
        assert retrieved.tags == ["test", "v3.3"]
        assert retrieved.version == 3
        assert retrieved.superseded_by == "PRD-002"
        assert len(retrieved.get_relationships()) == 1
        store2.close()

    def test_reload_empty_store(self, tmp_db):
        store1 = KnowledgeStore(tmp_db)
        store1.close()

        store2 = KnowledgeStore(tmp_db)
        assert store2.count() == 0
        store2.close()

    def test_reload_multiple_records(self, tmp_db):
        store1 = KnowledgeStore(tmp_db)
        for i in range(5):
            r = make_record(
                record_id=f"PRD-{i:03d}",
                record_type="PRD",
                title=f"Record {i}",
            )
            store1.save(r)
        store1.close()

        store2 = KnowledgeStore(tmp_db)
        assert store2.count() == 5
        for i in range(5):
            r = store2.get(f"PRD-{i:03d}")
            assert r.title == f"Record {i}"
        store2.close()


# ──────────────────────────────────────────────────────────────────────
# Indexes: efficient lookups by type, status, authority, tags
# ──────────────────────────────────────────────────────────────────────

class TestIndexes:
    """Test indexed lookups."""

    def test_find_by_type(self, store):
        store.save(make_record(record_id="PRD-001", record_type="PRD"))
        store.save(make_record(record_id="PRD-002", record_type="PRD"))
        store.save(make_record(record_id="NFR-001", record_type="NFR"))
        store.save(make_record(record_id="ADR-001", record_type="ADR", status="proposed"))

        prds = store.find_by_type("PRD")
        assert len(prds) == 2
        assert all(r.record_type == "PRD" for r in prds)

        nfrs = store.find_by_type("NFR")
        assert len(nfrs) == 1
        assert nfrs[0].record_id == "NFR-001"

    def test_find_by_status(self, store):
        store.save(make_record(record_id="PRD-001", record_type="PRD", status="draft"))
        store.save(make_record(record_id="PRD-002", record_type="PRD", status="review"))
        store.save(make_record(record_id="PRD-003", record_type="PRD", status="draft"))

        drafts = store.find_by_status("draft")
        assert len(drafts) == 2
        assert all(r.status == "draft" for r in drafts)

    def test_find_by_authority(self, store):
        store.save(make_record(record_id="PRD-001", record_type="PRD", authority="proposed"))
        store.save(make_record(record_id="PRD-002", record_type="PRD", authority="accepted"))
        store.save(make_record(record_id="NFR-001", record_type="NFR", authority="proposed"))

        proposed = store.find_by_authority("proposed")
        assert len(proposed) == 2
        assert all(r.authority == "proposed" for r in proposed)

    def test_find_by_tag(self, store):
        store.save(make_record(record_id="PRD-001", tags=["security", "auth"]))
        store.save(make_record(record_id="PRD-002", tags=["performance"]))
        store.save(make_record(record_id="NFR-001", record_type="NFR", tags=["security"]))

        security = store.find_by_tag("security")
        assert len(security) == 2
        assert all("security" in r.tags for r in security)

    def test_find_by_type_and_status(self, store):
        store.save(make_record(record_id="PRD-001", record_type="PRD", status="draft"))
        store.save(make_record(record_id="PRD-002", record_type="PRD", status="review"))
        store.save(make_record(record_id="NFR-001", record_type="NFR", status="draft"))

        results = store.find_by_type_and_status("PRD", "draft")
        assert len(results) == 1
        assert results[0].record_id == "PRD-001"

    def test_find_returns_empty_for_no_matches(self, store):
        store.save(make_record(record_id="PRD-001", record_type="PRD"))
        assert store.find_by_type("NONEXISTENT") == []
        assert store.find_by_status("nonexistent") == []
        assert store.find_by_tag("nonexistent") == []


# ──────────────────────────────────────────────────────────────────────
# Integrity: hash verification and tamper detection
# ──────────────────────────────────────────────────────────────────────

class TestIntegrity:
    """Test integrity verification."""

    def test_record_hash_stored(self, store):
        record = make_record()
        store.save(record)
        stored_hash = store.get_hash("PRD-001")
        assert stored_hash is not None
        assert len(stored_hash) == 64  # SHA-256 hex

    def test_integrity_check_passes(self, store):
        record = make_record()
        store.save(record)
        assert store.verify_integrity("PRD-001") is True

    def test_integrity_check_fails_on_tampering(self, store):
        record = make_record()
        store.save(record)
        # Tamper with the database directly — modify the JSON blob
        conn = sqlite3.connect(store.db_path)
        row = conn.execute(
            "SELECT record_json FROM records WHERE record_id = 'PRD-001'"
        ).fetchone()
        data = json.loads(row[0])
        data["title"] = "Tampered"
        conn.execute(
            "UPDATE records SET record_json = ? WHERE record_id = 'PRD-001'",
            (json.dumps(data, sort_keys=True),),
        )
        conn.commit()
        conn.close()
        assert store.verify_integrity("PRD-001") is False

    def test_verify_all_integrity(self, store):
        store.save(make_record(record_id="PRD-001"))
        store.save(make_record(record_id="NFR-001", record_type="NFR"))
        results = store.verify_all_integrity()
        assert results == {"PRD-001": True, "NFR-001": True}

    def test_verify_all_integrity_with_tampering(self, store):
        store.save(make_record(record_id="PRD-001"))
        store.save(make_record(record_id="NFR-001", record_type="NFR"))
        # Tamper with one record's JSON
        conn = sqlite3.connect(store.db_path)
        row = conn.execute(
            "SELECT record_json FROM records WHERE record_id = 'PRD-001'"
        ).fetchone()
        data = json.loads(row[0])
        data["title"] = "Tampered"
        conn.execute(
            "UPDATE records SET record_json = ? WHERE record_id = 'PRD-001'",
            (json.dumps(data, sort_keys=True),),
        )
        conn.commit()
        conn.close()
        results = store.verify_all_integrity()
        assert results["PRD-001"] is False
        assert results["NFR-001"] is True

    def test_reload_detects_tampering(self, tmp_db):
        store1 = KnowledgeStore(tmp_db)
        store1.save(make_record())
        store1.close()

        # Tamper with the file's JSON content
        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT record_json FROM records WHERE record_id = 'PRD-001'"
        ).fetchone()
        data = json.loads(row[0])
        data["title"] = "Tampered"
        conn.execute(
            "UPDATE records SET record_json = ? WHERE record_id = 'PRD-001'",
            (json.dumps(data, sort_keys=True),),
        )
        conn.commit()
        conn.close()

        store2 = KnowledgeStore(tmp_db)
        with pytest.raises(IntegrityError):
            store2.get("PRD-001")
        store2.close()


# ──────────────────────────────────────────────────────────────────────
# Store lifecycle and metadata
# ──────────────────────────────────────────────────────────────────────

class TestStoreLifecycle:
    """Test store open/close and metadata."""

    def test_store_creates_tables(self, tmp_db):
        store = KnowledgeStore(tmp_db)
        conn = sqlite3.connect(tmp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        table_names = [t[0] for t in tables]
        assert "records" in table_names
        assert "record_hashes" in table_names
        assert "relationships" in table_names
        store.close()

    def test_store_isolation(self, tmp_db):
        store1 = KnowledgeStore(tmp_db)
        store1.save(make_record())
        store1.close()

        store2 = KnowledgeStore(tmp_db)
        assert store2.count() == 1
        store2.close()

    def test_context_manager(self, tmp_db):
        with KnowledgeStore(tmp_db) as store:
            store.save(make_record())
            assert store.count() == 1
        # After close, data should persist
        store2 = KnowledgeStore(tmp_db)
        assert store2.count() == 1
        store2.close()


# ──────────────────────────────────────────────────────────────────────
# Search and listing
# ──────────────────────────────────────────────────────────────────────

class TestSearch:
    """Test search and listing operations."""

    def test_list_all(self, store):
        store.save(make_record(record_id="PRD-001", record_type="PRD"))
        store.save(make_record(record_id="NFR-001", record_type="NFR"))
        all_records = store.list_all()
        assert len(all_records) == 2

    def test_list_by_type(self, store):
        store.save(make_record(record_id="PRD-001", record_type="PRD"))
        store.save(make_record(record_id="PRD-002", record_type="PRD"))
        store.save(make_record(record_id="NFR-001", record_type="NFR"))
        prds = store.list_by_type("PRD")
        assert len(prds) == 2

    def test_search_by_title(self, store):
        store.save(make_record(record_id="PRD-001", title="Security Requirements"))
        store.save(make_record(record_id="PRD-002", title="Performance Goals"))
        store.save(make_record(record_id="NFR-001", title="Security Constraints", record_type="NFR"))
        results = store.search_by_title("Security")
        assert len(results) == 2

    def test_search_by_description(self, store):
        store.save(make_record(record_id="PRD-001", description="Authentication flow"))
        store.save(make_record(record_id="PRD-002", description="Authorization model"))
        results = store.search_by_description("auth")
        assert len(results) == 2


# ──────────────────────────────────────────────────────────────────────
# Error handling
# ──────────────────────────────────────────────────────────────────────

class TestErrorHandling:
    """Test error handling."""

    def test_invalid_db_path(self):
        with pytest.raises(StoreError):
            KnowledgeStore("/nonexistent/path/db.sqlite")

    def test_corrupted_database(self, tmp_db):
        # Write garbage to the db file
        with open(tmp_db, "w") as f:
            f.write("not a sqlite database")
        with pytest.raises(StoreError):
            KnowledgeStore(tmp_db)

    def test_save_invalid_record(self, store):
        with pytest.raises(RecordError):
            store.save(make_record(title=""))
