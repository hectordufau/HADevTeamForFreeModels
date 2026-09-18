# harness/knowledge/store.py — V3.3 KnowledgeStore: Persistence, Reload, Indexes, Integrity
"""
SQLite-backed persistence for EngineeringRecord instances.

Provides:
- Save/update/delete records with ACID transactions
- Reload from disk (durability across restarts)
- Indexed lookups by type, status, authority, tags
- SHA-256 integrity verification (tamper detection)
- Search by title and description
- Context manager support (with statement)

Architecture:
- Records stored in normalized SQLite schema
- Record hashes stored separately for integrity verification
- Relationships stored in separate table
- All lookups use indexes for O(log n) performance

Three-domain separation:
- KnowledgeStore manages EngineeringRecord persistence only
- Does NOT touch Learning domain (experience, strategy, failure)
- Does NOT touch Evidence domain (evidence packages, hash chains)
"""

import hashlib
import json
import os
import sqlite3
from typing import Dict, List, Optional, Set
from contextlib import contextmanager

from .records import EngineeringRecord, RecordError, RecordRelationship
from .provenance import Provenance
from .lifecycle import VALID_RECORD_TYPES


class StoreError(Exception):
    """Raised on store operation errors."""


class IntegrityError(Exception):
    """Raised when record integrity verification fails."""


class RecordNotFoundError(Exception):
    """Raised when a requested record does not exist."""


class DuplicateRecordError(Exception):
    """Raised when saving a record that already exists (without update)."""


SCHEMA_VERSION = 1

# Typed field markers — presence of these in record JSON indicates typed origin
_TYPED_FIELD_MARKERS = {
    "PRD": ("objective", "scope"),
    "REQ": ("req_type", "priority", "parent_prd"),
    "NFR": ("category", "scope", "qualitative"),
    "DR": ("decision_type", "context", "rationale"),
    "ADR": ("architecture_domain", "pattern_selected", "trade_offs"),
    "TDR": ("debt_type", "severity", "remediation"),
    "RSK": ("risk_category", "likelihood", "mitigation"),
    "SEC": ("category", "enforcement", "authority"),
}


def _has_typed_fields(record_data: dict, record_type: str) -> bool:
    """Check if record data contains typed-specific fields."""
    markers = _TYPED_FIELD_MARKERS.get(record_type, ())
    return any(marker in record_data for marker in markers)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    record_id TEXT PRIMARY KEY,
    record_type TEXT NOT NULL,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    version INTEGER NOT NULL,
    superseded_by TEXT,
    status TEXT NOT NULL,
    authority TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_records_type ON records(record_type);
CREATE INDEX IF NOT EXISTS idx_records_status ON records(status);
CREATE INDEX IF NOT EXISTS idx_records_authority ON records(authority);
CREATE INDEX IF NOT EXISTS idx_records_updated ON records(updated_at);

CREATE TABLE IF NOT EXISTS record_hashes (
    record_id TEXT PRIMARY KEY,
    record_hash TEXT NOT NULL,
    computed_at TEXT NOT NULL,
    FOREIGN KEY (record_id) REFERENCES records(record_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS relationships (
    source_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    provenance TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (source_id, relation_type, target_id),
    FOREIGN KEY (source_id) REFERENCES records(record_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relation_type);

CREATE TABLE IF NOT EXISTS store_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class KnowledgeStore:
    """
    SQLite-backed knowledge store for engineering records.

    Usage:
        store = KnowledgeStore("knowledge.db")
        store.save(record)
        record = store.get("PRD-001")
        store.close()

    Or as context manager:
        with KnowledgeStore("knowledge.db") as store:
            store.save(record)
    """

    def __init__(self, db_path: str):
        """
        Initialize a KnowledgeStore.

        Args:
            db_path: Path to SQLite database file. Created if it doesn't exist.

        Raises:
            StoreError: If the database cannot be opened or the path is invalid.
        """
        if not isinstance(db_path, str) or not db_path.strip():
            raise StoreError("db_path must be a non-empty string")

        # Ensure parent directory exists
        parent = os.path.dirname(db_path)
        if parent and not os.path.exists(parent):
            raise StoreError(f"Parent directory does not exist: {parent}")

        # Check if file exists and is not a valid SQLite DB
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                conn.execute("PRAGMA schema_version")
                conn.close()
            except sqlite3.DatabaseError:
                raise StoreError(f"File exists but is not a valid SQLite database: {db_path}")

        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._open()
        self._init_schema()

    def _open(self) -> None:
        """Open the SQLite connection."""
        try:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        except sqlite3.Error as e:
            raise StoreError(f"Failed to open database: {e}")

    def _init_schema(self) -> None:
        """Initialize database schema."""
        if self._conn is None:
            raise StoreError("Database connection is not open")
        try:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR REPLACE INTO store_metadata (key, value) VALUES (?, ?)",
                ("schema_version", str(SCHEMA_VERSION)),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            raise StoreError(f"Failed to initialize schema: {e}")

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "KnowledgeStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @contextmanager
    def _cursor(self):
        """Provide a transactional cursor context."""
        if self._conn is None:
            raise StoreError("Database connection is not open")
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    # ──────────────────────────────────────────────────────────────────────
    # Core CRUD
    # ──────────────────────────────────────────────────────────────────────

    def save(self, record: EngineeringRecord, *, update: bool = True) -> None:
        """
        Save an EngineeringRecord to the store.

        Args:
            record: The record to save.
            update: If True, overwrite existing record with same ID.
                   If False, raise DuplicateRecordError.

        Raises:
            RecordError: If record validation fails.
            DuplicateRecordError: If record exists and update=False.
        """
        record.validate()

        # Phase 4: If record is a typed class instance (PRD, REQ, NFR), ensure
        # it uses the canonical typed class for consistent serialization.
        # Base EngineeringRecord instances are stored as-is (no conversion).
        from .registry import TYPED_RECORD_CLASSES
        target_cls = TYPED_RECORD_CLASSES.get(record.record_type)
        if target_cls is not None and isinstance(record, target_cls) and type(record) is not target_cls:
            record = target_cls.from_dict(record.to_dict())

        with self._cursor() as cursor:
            # Check for existing
            existing = cursor.execute(
                "SELECT 1 FROM records WHERE record_id = ?",
                (record.record_id,),
            ).fetchone()

            if existing and not update:
                raise DuplicateRecordError(
                    f"Record '{record.record_id}' already exists"
                )

            # Serialize
            record_json = record.to_json()
            tags_json = json.dumps(sorted(record.tags))

            if existing:
                # Update
                cursor.execute(
                    """
                    UPDATE records SET
                        record_type = ?,
                        record_json = ?,
                        updated_at = ?,
                        version = ?,
                        superseded_by = ?,
                        status = ?,
                        authority = ?,
                        tags = ?
                    WHERE record_id = ?
                    """,
                    (
                        record.record_type,
                        record_json,
                        record.updated_at,
                        record.version,
                        record.superseded_by,
                        record.status,
                        record.authority,
                        tags_json,
                        record.record_id,
                    ),
                )
            else:
                # Insert
                cursor.execute(
                    """
                    INSERT INTO records (
                        record_id, record_type, record_json,
                        created_at, updated_at, version,
                        superseded_by, status, authority, tags
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.record_id,
                        record.record_type,
                        record_json,
                        record.created_at,
                        record.updated_at,
                        record.version,
                        record.superseded_by,
                        record.status,
                        record.authority,
                        tags_json,
                    ),
                )

            # Store hash
            record_hash = record.compute_hash()
            cursor.execute(
                """
                INSERT OR REPLACE INTO record_hashes (record_id, record_hash, computed_at)
                VALUES (?, ?, ?)
                """,
                (record.record_id, record_hash, record.updated_at),
            )

            # Store relationships
            # Delete existing relationships for this record
            cursor.execute(
                "DELETE FROM relationships WHERE source_id = ?",
                (record.record_id,),
            )
            # Insert current relationships
            for rel in record.get_relationships():
                prov_json = rel.provenance.to_json() if rel.provenance else None
                meta_json = json.dumps(rel.metadata, sort_keys=True)
                cursor.execute(
                    """
                    INSERT INTO relationships (source_id, relation_type, target_id, provenance, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (rel.source_id, rel.relation_type, rel.target_id, prov_json, meta_json),
                )

    def get(self, record_id: str, *, verify: bool = True) -> EngineeringRecord:
        """
        Retrieve a record by ID.

        Args:
            record_id: The record identifier.
            verify: If True, verify integrity before returning.

        Returns:
            The EngineeringRecord.

        Raises:
            RecordNotFoundError: If the record does not exist.
            IntegrityError: If integrity verification fails.
        """
        # Check existence first
        with self._cursor() as cursor:
            row = cursor.execute(
                "SELECT 1 FROM records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"Record '{record_id}' not found")

        if verify:
            if not self.verify_integrity(record_id):
                raise IntegrityError(
                    f"Integrity check failed for record '{record_id}'"
                )

        with self._cursor() as cursor:
            row = cursor.execute(
                "SELECT record_json FROM records WHERE record_id = ?",
                (record_id,),
            ).fetchone()

            record_data = json.loads(row["record_json"])
            # Phase 4: Dispatch to typed class only if the record has typed-specific fields
            from .registry import TYPED_RECORD_CLASSES
            record_type = record_data.get("record_type", "")
            target_cls = TYPED_RECORD_CLASSES.get(record_type)
            if target_cls is not None:
                markers = _TYPED_FIELD_MARKERS.get(record_type, ())
                if any(m in record_data for m in markers):
                    try:
                        return target_cls.from_dict(record_data)
                    except (RecordError, TypeError, ValueError, KeyError):
                        pass
            return EngineeringRecord.from_dict(record_data)

    def delete(self, record_id: str) -> None:
        """
        Delete a record and its relationships.

        Args:
            record_id: The record to delete.

        Raises:
            RecordNotFoundError: If the record does not exist.
        """
        with self._cursor() as cursor:
            row = cursor.execute(
                "SELECT 1 FROM records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                raise RecordNotFoundError(f"Record '{record_id}' not found")

            # Relationships are cascade-deleted via FK
            cursor.execute(
                "DELETE FROM records WHERE record_id = ?",
                (record_id,),
            )

    def count(self) -> int:
        """Return the total number of records."""
        with self._cursor() as cursor:
            row = cursor.execute("SELECT COUNT(*) FROM records").fetchone()
            return row[0] if row else 0

    def exists(self, record_id: str) -> bool:
        """Check if a record exists."""
        with self._cursor() as cursor:
            row = cursor.execute(
                "SELECT 1 FROM records WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            return row is not None

    # ──────────────────────────────────────────────────────────────────────
    # Reload / durability
    # ──────────────────────────────────────────────────────────────────────

    def reload(self) -> None:
        """
        Close and reopen the database connection.
        Forces WAL checkpoint and re-reads all data from disk.
        """
        self.close()
        self._open()
        self._init_schema()

    # ──────────────────────────────────────────────────────────────────────
    # Indexed lookups
    # ──────────────────────────────────────────────────────────────────────

    def find_by_type(self, record_type: str) -> List[EngineeringRecord]:
        """Find all records of a given type."""
        return self._find_by_column("record_type", record_type)

    def find_by_status(self, status: str) -> List[EngineeringRecord]:
        """Find all records with a given status."""
        return self._find_by_column("status", status)

    def find_by_authority(self, authority: str) -> List[EngineeringRecord]:
        """Find all records with a given authority level."""
        return self._find_by_column("authority", authority)

    def find_by_type_and_status(self, record_type: str, status: str) -> List[EngineeringRecord]:
        """Find records matching both type and status."""
        with self._cursor() as cursor:
            rows = cursor.execute(
                "SELECT record_json FROM records WHERE record_type = ? AND status = ?",
                (record_type, status),
            ).fetchall()
            return [self._record_from_row(row) for row in rows]

    def find_by_tag(self, tag: str) -> List[EngineeringRecord]:
        """Find all records that have a specific tag."""
        with self._cursor() as cursor:
            rows = cursor.execute(
                "SELECT record_json, tags FROM records"
            ).fetchall()
            results = []
            for row in rows:
                tags = json.loads(row["tags"])
                if tag in tags:
                    results.append(self._record_from_row(row))
            return results

    def _find_by_column(self, column: str, value: str) -> List[EngineeringRecord]:
        """Helper to find records by a specific column value."""
        with self._cursor() as cursor:
            rows = cursor.execute(
                f"SELECT record_json FROM records WHERE {column} = ?",
                (value,),
            ).fetchall()
            return [self._record_from_row(row) for row in rows]

    def _record_from_row(self, row: sqlite3.Row) -> EngineeringRecord:
        """Reconstruct a record from a database row with typed dispatch."""
        record_data = json.loads(row["record_json"])
        record_type = record_data.get("record_type", "")
        # Dispatch to typed class if markers match
        from .registry import TYPED_RECORD_CLASSES
        target_cls = TYPED_RECORD_CLASSES.get(record_type)
        if target_cls is not None:
            markers = _TYPED_FIELD_MARKERS.get(record_type, ())
            if any(m in record_data for m in markers):
                try:
                    return target_cls.from_dict(record_data)
                except (RecordError, TypeError, ValueError, KeyError):
                    pass
        return EngineeringRecord.from_dict(record_data)

    # ──────────────────────────────────────────────────────────────────────
    # Listing and search
    # ──────────────────────────────────────────────────────────────────────

    def list_all(self) -> List[EngineeringRecord]:
        """Return all records."""
        with self._cursor() as cursor:
            rows = cursor.execute(
                "SELECT record_json FROM records ORDER BY record_id"
            ).fetchall()
            return [self._record_from_row(row) for row in rows]

    def list_by_type(self, record_type: str) -> List[EngineeringRecord]:
        """Alias for find_by_type (returns typed records ordered by ID)."""
        return self.find_by_type(record_type)

    def search_by_title(self, query: str) -> List[EngineeringRecord]:
        """Search records by title substring (case-insensitive)."""
        with self._cursor() as cursor:
            rows = cursor.execute(
                "SELECT record_json FROM records"
            ).fetchall()
            results = []
            query_lower = query.lower()
            for row in rows:
                record_data = json.loads(row["record_json"])
                title = record_data.get("title", "")
                if query_lower in title.lower():
                    results.append(EngineeringRecord.from_dict(record_data))
            return results

    def search_by_description(self, query: str) -> List[EngineeringRecord]:
        """Search records by description substring (case-insensitive)."""
        with self._cursor() as cursor:
            rows = cursor.execute(
                "SELECT record_json FROM records"
            ).fetchall()
            results = []
            query_lower = query.lower()
            for row in rows:
                record_data = json.loads(row["record_json"])
                description = record_data.get("description", "")
                if query_lower in description.lower():
                    results.append(EngineeringRecord.from_dict(record_data))
            return results

    # ──────────────────────────────────────────────────────────────────────
    # Integrity
    # ──────────────────────────────────────────────────────────────────────

    def get_hash(self, record_id: str) -> Optional[str]:
        """Get the stored SHA-256 hash for a record."""
        with self._cursor() as cursor:
            row = cursor.execute(
                "SELECT record_hash FROM record_hashes WHERE record_id = ?",
                (record_id,),
            ).fetchone()
            return row["record_hash"] if row else None

    def verify_integrity(self, record_id: str) -> bool:
        """
        Verify the integrity of a stored record.

        Computes the hash of the stored record JSON directly and compares
        it with the hash stored at save time.

        Returns:
            True if integrity is verified, False otherwise.
        """
        stored_hash = self.get_hash(record_id)
        if stored_hash is None:
            return False

        try:
            with self._cursor() as cursor:
                row = cursor.execute(
                    "SELECT record_json FROM records WHERE record_id = ?",
                    (record_id,),
                ).fetchone()
                if row is None:
                    return False
                record_data = json.loads(row["record_json"])
        except (RecordNotFoundError, json.JSONDecodeError):
            return False

        # Compute hash from the stored JSON directly (preserves typed fields)
        canonical = json.dumps(record_data, sort_keys=True, separators=(",", ":"))
        computed_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return stored_hash == computed_hash

    def verify_all_integrity(self) -> Dict[str, bool]:
        """
        Verify integrity of all stored records.

        Returns:
            Dict mapping record_id to integrity result (True/False).
        """
        results = {}
        with self._cursor() as cursor:
            rows = cursor.execute("SELECT record_id FROM records").fetchall()
            for row in rows:
                rid = row["record_id"]
                results[rid] = self.verify_integrity(rid)
        return results

    def rebuild_indexes(self) -> None:
        """Rebuild all indexes (for maintenance/recovery)."""
        with self._cursor() as cursor:
            cursor.execute("REINDEX")

    # ──────────────────────────────────────────────────────────────────────
    # Metadata
    # ──────────────────────────────────────────────────────────────────────

    def get_metadata(self, key: str) -> Optional[str]:
        """Get a metadata value."""
        with self._cursor() as cursor:
            row = cursor.execute(
                "SELECT value FROM store_metadata WHERE key = ?",
                (key,),
            ).fetchone()
            return row["value"] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        """Set a metadata value."""
        with self._cursor() as cursor:
            cursor.execute(
                "INSERT OR REPLACE INTO store_metadata (key, value) VALUES (?, ?)",
                (key, value),
            )

    def get_schema_version(self) -> Optional[int]:
        """Get the schema version of the database."""
        val = self.get_metadata("schema_version")
        return int(val) if val is not None else None
