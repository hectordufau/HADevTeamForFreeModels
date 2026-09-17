# harness/knowledge/registry.py — V3.3 Record Type Registry
"""
Central registry for record type → class mapping.

Breaks circular imports between records.py and requirements.py.
Import this module from anywhere that needs the registry.
"""

from typing import Dict, Type
from .records import EngineeringRecord
from .requirements import PRD, Requirement, NFR

# Typed record classes (Phase 4: PRD, REQ, NFR)
TYPED_RECORD_CLASSES: Dict[str, Type] = {
    "PRD": PRD,
    "REQ": Requirement,
    "NFR": NFR,
    # DR, ADR, TDR, RSK, SEC, RCA deferred to Phase 5-7
}

# Full registry including untyped (base class) entries
RECORD_TYPE_CLASSES: Dict[str, Type] = {
    **TYPED_RECORD_CLASSES,
    "DR": EngineeringRecord,
    "ADR": EngineeringRecord,
    "TDR": EngineeringRecord,
    "RSK": EngineeringRecord,
    "SEC": EngineeringRecord,
    "RCA": EngineeringRecord,
}
