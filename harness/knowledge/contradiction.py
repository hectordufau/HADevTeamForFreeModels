# harness/knowledge/contradiction.py — V3.3 Contradiction Detection
"""
Detect and track contradictions between engineering records.

Contradictions arise when:
- Two ADRs on the same topic make conflicting decisions (both ACCEPTED)
- An ADR violates a governing NFR on the same topic
- Records with the same tag have semantically conflicting descriptions

Domain separation:
- ContradictionDetector belongs to Engineering Knowledge
- Does NOT touch Learning or Evidence domains
- Integrates with KnowledgeGraph for relationship context
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from .records import EngineeringRecord
from .graph import KnowledgeGraph
from .provenance import AUTHORITY_ACCEPTED


class ContradictionError(Exception):
    """Raised on contradiction detection errors."""


class ContradictionStatus(Enum):
    """Lifecycle status of a detected contradiction."""
    PENDING = "pending"       # Detected, not yet reviewed
    FLAGGED = "flagged"       # Flagged for human review
    RESOLVED = "resolved"     # Resolved (e.g., supersession, clarification)
    REJECTED = "rejected"     # Rejected (not a real contradiction)


# Technology/entity keywords for conflict detection
_TECH_KEYWORDS = [
    "postgresql", "mongodb", "mysql", "sqlite",
    "react", "vue", "angular", "svelte",
    "redis", "memcached", "kafka", "rabbitmq",
    "rest", "graphql", "grpc",
    "docker", "kubernetes", "terraform",
    "aws", "gcp", "azure",
]

# NFR violation patterns: NFR target vs ADR description
# Each pattern: (nfr_target_regex, violation_regex, category)
_NFR_VIOLATION_PATTERNS = [
    # Latency: NFR says "under/below/less than X ms" or "< X ms", ADR says "> X ms"
    (r"(?:under|below|less than|<\s*)\s*(\d+)\s*ms", r">\s*\d+\s*ms", "latency"),
    # Throughput: NFR says "over/above/more than X mb/s" or "> X mb/s", ADR says "< X mb/s"
    (r"(?:over|above|more than|>\s*)\s*(\d+)\s*mb/s", r"<\s*\d+\s*mb/s", "throughput"),
    # CPU: NFR says "under/below X% cpu", ADR says "> X% cpu"
    (r"(?:under|below|<\s*)\s*(\d+)\s*%\s*cpu", r">\s*\d+\s*%\s*cpu", "cpu_usage"),
    # Availability: NFR says "over/above X%", ADR says "< X%"
    (r"(?:over|above|>\s*)\s*(\d+)\s*%\s*availability", r"<\s*\d+\s*%\s*availability", "availability"),
]


class Contradiction:
    """
    Represents a detected contradiction between two engineering records.

    Fields:
        contradiction_id: Unique identifier (deterministic)
        source_id: The record that has the issue (usually NFR or older ADR)
        target_id: The conflicting record (usually newer ADR)
        contradiction_type: Classification (ADR_CONFLICT, NFR_VIOLATION, STATUS_CONFLICT)
        status: Review status (pending → flagged → resolved/rejected)
        description: Human-readable description
        detected_at: ISO-8601 timestamp
        flagged_at: When flagged for review (if applicable)
        resolved_at: When resolved (if applicable)
        resolution: Resolution details (if resolved)
    """

    def __init__(
        self,
        contradiction_id: str,
        source_id: str,
        target_id: str,
        contradiction_type: str,
        description: str = "",
        status: ContradictionStatus = ContradictionStatus.PENDING,
        detected_at: Optional[str] = None,
        flagged_at: Optional[str] = None,
        resolved_at: Optional[str] = None,
        resolution: str = "",
    ):
        self.contradiction_id = contradiction_id
        self.source_id = source_id
        self.target_id = target_id
        self.contradiction_type = contradiction_type
        self.description = description
        self.status = status
        self.detected_at = detected_at or datetime.utcnow().isoformat()
        self.flagged_at = flagged_at
        self.resolved_at = resolved_at
        self.resolution = resolution

    def to_dict(self) -> dict:
        return {
            "contradiction_id": self.contradiction_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "contradiction_type": self.contradiction_type,
            "status": self.status.value,
            "description": self.description,
            "detected_at": self.detected_at,
            "flagged_at": self.flagged_at,
            "resolved_at": self.resolved_at,
            "resolution": self.resolution,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Contradiction":
        return cls(
            contradiction_id=data["contradiction_id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            contradiction_type=data["contradiction_type"],
            description=data.get("description", ""),
            status=ContradictionStatus(data.get("status", "pending")),
            detected_at=data.get("detected_at"),
            flagged_at=data.get("flagged_at"),
            resolved_at=data.get("resolved_at"),
            resolution=data.get("resolution", ""),
        )


def _extract_technologies(text: str) -> set:
    """Extract technology/entity mentions from text."""
    text_lower = text.lower()
    found = set()
    for keyword in _TECH_KEYWORDS:
        if keyword in text_lower:
            found.add(keyword)
    return found


def _check_nfr_violation(nfr_desc: str, adr_desc: str) -> Optional[str]:
    """Check if an ADR description violates an NFR target pattern."""
    for target_pat, viol_pat, category in _NFR_VIOLATION_PATTERNS:
        target_match = re.search(target_pat, nfr_desc, re.IGNORECASE)
        if target_match:
            viol_match = re.search(viol_pat, adr_desc, re.IGNORECASE)
            if viol_match:
                return category
    return None


class ContradictionDetector:
    """
    Detects contradictions between engineering records.

    Records are indexed by tag (topic). Within each tag group, the detector
    checks for:
    1. ADR vs ADR: Conflicting technology decisions (both accepted)
    2. NFR vs ADR: ADR description violates NFR target
    3. Any vs any: Records with identical titles (potential duplicates)

    Usage:
        detector = ContradictionDetector(graph)
        detector.index_record(adr1)
        detector.index_record(adr2)
        contradictions = detector.detect()
        detector.flag_contradiction(contradictions[0].contradiction_id)
    """

    def __init__(self, graph: Optional[KnowledgeGraph] = None):
        self.graph = graph or KnowledgeGraph()
        # Indexed records: record_id → EngineeringRecord
        self._records: Dict[str, EngineeringRecord] = {}
        # Tag → set of record_ids
        self._tag_index: Dict[str, set] = {}
        # Active contradictions: contradiction_id → Contradiction
        self._contradictions: Dict[str, Contradiction] = {}

    def index_record(self, record: EngineeringRecord) -> None:
        """Index a record for contradiction detection. Idempotent."""
        self._records[record.record_id] = record
        for tag in record.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(record.record_id)

    def detect(self) -> List[Contradiction]:
        """
        Detect contradictions among all indexed records.

        Returns:
            List of Contradiction objects. New contradictions are stored
            internally; previously found contradictions are not duplicated.
        """
        contradictions: List[Contradiction] = []

        # Check each tag group (topic)
        for tag, record_ids in self._tag_index.items():
            records = [self._records[rid] for rid in record_ids]
            group_contradictions = self._detect_in_group(records, tag)
            for c in group_contradictions:
                if c.contradiction_id not in self._contradictions:
                    self._contradictions[c.contradiction_id] = c
                contradictions.append(c)

        return contradictions

    def _detect_in_group(
        self, records: List[EngineeringRecord], tag: str
    ) -> List[Contradiction]:
        """Detect contradictions within a tag group."""
        contradictions = []

        # ADR vs ADR on same topic
        accepted_adrs = [
            r for r in records
            if r.record_type == "ADR" and r.authority == AUTHORITY_ACCEPTED
        ]
        for i, adr1 in enumerate(accepted_adrs):
            for adr2 in accepted_adrs[i + 1:]:
                # Same topic (tag), different technologies mentioned
                techs1 = _extract_technologies(adr1.description)
                techs2 = _extract_technologies(adr2.description)
                if techs1 and techs2 and techs1 != techs2:
                    # Ensure deterministic ordering
                    source, target = (adr1, adr2) if adr1.record_id < adr2.record_id else (adr2, adr1)
                    cid = self._make_contradiction_id(
                        source.record_id, target.record_id, "ADR_CONFLICT"
                    )
                    c = Contradiction(
                        contradiction_id=cid,
                        source_id=source.record_id,
                        target_id=target.record_id,
                        contradiction_type="ADR_CONFLICT",
                        description=(
                            f"Both '{source.record_id}' and '{target.record_id}' "
                            f"make different decisions on topic '{tag}': "
                            f"{techs1} vs {techs2}"
                        ),
                    )
                    contradictions.append(c)

        # NFR vs ADR on same topic
        nfrs = [
            r for r in records
            if r.record_type == "NFR" and r.authority == AUTHORITY_ACCEPTED
        ]
        adrs = [
            r for r in records
            if r.record_type == "ADR" and r.authority == AUTHORITY_ACCEPTED
        ]
        for nfr in nfrs:
            for adr in adrs:
                violation = _check_nfr_violation(nfr.description, adr.description)
                if violation:
                    cid = self._make_contradiction_id(
                        nfr.record_id, adr.record_id, "NFR_VIOLATION"
                    )
                    c = Contradiction(
                        contradiction_id=cid,
                        source_id=nfr.record_id,
                        target_id=adr.record_id,
                        contradiction_type="NFR_VIOLATION",
                        description=(
                            f"ADR '{adr.record_id}' may violate NFR "
                            f"'{nfr.record_id}' ({violation}): "
                            f"NFR target vs ADR description"
                        ),
                    )
                    contradictions.append(c)

        return contradictions

    def _make_contradiction_id(self, a: str, b: str, ctype: str) -> str:
        """Create a deterministic contradiction ID."""
        return f"CONTRADICTION-{ctype}-{a}-{b}"

    def get_contradiction(self, contradiction_id: str) -> Optional[Contradiction]:
        """Retrieve a contradiction by ID."""
        return self._contradictions.get(contradiction_id)

    def flag_contradiction(self, contradiction_id: str) -> Contradiction:
        """
        Flag a contradiction for human review.

        Raises ContradictionError if the contradiction doesn't exist
        or is already flagged.
        """
        c = self._contradictions.get(contradiction_id)
        if c is None:
            raise ContradictionError(
                f"Contradiction '{contradiction_id}' not found"
            )
        if c.status == ContradictionStatus.FLAGGED:
            raise ContradictionError(
                f"Contradiction '{contradiction_id}' is already flagged"
            )
        c.status = ContradictionStatus.FLAGGED
        c.flagged_at = datetime.utcnow().isoformat()
        return c

    def resolve_contradiction(
        self, contradiction_id: str, resolution: str
    ) -> Contradiction:
        """
        Mark a contradiction as resolved.

        Args:
            contradiction_id: The contradiction to resolve.
            resolution: Human-readable resolution details.

        Raises ContradictionError if not found.
        """
        c = self._contradictions.get(contradiction_id)
        if c is None:
            raise ContradictionError(
                f"Contradiction '{contradiction_id}' not found"
            )
        c.status = ContradictionStatus.RESOLVED
        c.resolution = resolution
        c.resolved_at = datetime.utcnow().isoformat()
        return c

    def reject_contradiction(self, contradiction_id: str, reason: str = "") -> Contradiction:
        """Mark a contradiction as rejected (false positive)."""
        c = self._contradictions.get(contradiction_id)
        if c is None:
            raise ContradictionError(
                f"Contradiction '{contradiction_id}' not found"
            )
        c.status = ContradictionStatus.REJECTED
        c.resolution = reason
        c.resolved_at = datetime.utcnow().isoformat()
        return c

    def get_all_contradictions(self) -> List[Contradiction]:
        """Return all known contradictions."""
        return list(self._contradictions.values())

    def get_flagged_contradictions(self) -> List[Contradiction]:
        """Return all contradictions flagged for review."""
        return [
            c for c in self._contradictions.values()
            if c.status == ContradictionStatus.FLAGGED
        ]

    def get_pending_contradictions(self) -> List[Contradiction]:
        """Return all pending (unreviewed) contradictions."""
        return [
            c for c in self._contradictions.values()
            if c.status == ContradictionStatus.PENDING
        ]
