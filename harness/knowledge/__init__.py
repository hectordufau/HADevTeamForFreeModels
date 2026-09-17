# harness/knowledge/__init__.py — V3.3 Engineering Knowledge Package
"""
Engineering Knowledge package for V3.3.

Provides structured engineering records (PRD, NFR, DR, ADR, TDR, RSK, SEC, RCA)
with provenance, authority, lifecycle, relationships, versioning, and validation.

Architecture boundaries:
- Engineering Knowledge is separate from Learning Knowledge and Evidence
- Does NOT replace V3.2 learning modules (StructuredExperience, FailureLesson, Strategy, etc.)
- Does NOT replace Evidence modules (EvidencePackage, EvidenceStore)
- Maintains three-domain separation: Engineering Knowledge / Learning Knowledge / Evidence
"""

from .records import (
    EngineeringRecord,
    RecordRelationship,
    RecordError,
    RECORD_TYPE_CLASSES,
    VALID_RECORD_TYPES,
    VALID_RELATIONSHIP_TYPES,
    create_record,
)
from .provenance import (
    Provenance,
    ProvenanceError,
    validate_authority_level,
    is_authority_at_least,
    can_transition_authority,
    AUTHORITY_PROPOSED,
    AUTHORITY_ACCEPTED,
    AUTHORITY_DEPRECATED,
    AUTHORITY_ARCHIVED,
    VALID_AUTHORITY_LEVELS,
    VALID_SOURCES,
)
from .lifecycle import (
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
)
from .store import (
    KnowledgeStore,
    StoreError,
    IntegrityError,
    RecordNotFoundError,
    DuplicateRecordError,
)
from .graph import (
    KnowledgeGraph,
    GraphError,
    EdgeType,
    EDGE_TYPES,
    VALID_EDGE_TYPES,
)
from .contradiction import (
    ContradictionDetector,
    ContradictionError,
    Contradiction,
    ContradictionStatus,
)

__all__ = [
    # Records
    "EngineeringRecord",
    "RecordRelationship",
    "RecordError",
    "PRD",
    "NFR",
    "DR",
    "ADR",
    "TDR",
    "RSK",
    "SEC",
    "RCA",
    "RECORD_TYPE_CLASSES",
    "VALID_RECORD_TYPES",
    "VALID_RELATIONSHIP_TYPES",
    "create_record",
    # Provenance
    "Provenance",
    "ProvenanceError",
    "validate_authority_level",
    "is_authority_at_least",
    "can_transition_authority",
    "AUTHORITY_PROPOSED",
    "AUTHORITY_ACCEPTED",
    "AUTHORITY_DEPRECATED",
    "AUTHORITY_ARCHIVED",
    "VALID_AUTHORITY_LEVELS",
    "VALID_SOURCES",
    # Lifecycle
    "LifecycleError",
    "LifecycleTransition",
    "get_lifecycle_definition",
    "get_initial_state",
    "get_terminal_states",
    "get_valid_states",
    "is_valid_transition",
    "transition",
    "is_terminal_state",
    "get_allowed_transitions",
    "validate_lifecycle_state",
    "LIFECYCLE_DEFINITIONS",
    # Store
    "KnowledgeStore",
    "StoreError",
    "IntegrityError",
    "RecordNotFoundError",
    "DuplicateRecordError",
    # Graph
    "KnowledgeGraph",
    "GraphError",
    "EdgeType",
    "EDGE_TYPES",
    "VALID_EDGE_TYPES",
    # Contradiction
    "ContradictionDetector",
    "ContradictionError",
    "Contradiction",
    "ContradictionStatus",
]
