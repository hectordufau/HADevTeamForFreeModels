# harness/knowledge/registry.py — V3.3 Record Type Registry
"""
Central registry for record type → class mapping.

Breaks circular imports between records.py and requirements.py.
Import this module from anywhere that needs the registry.
"""

from typing import Dict, Type
from .records import EngineeringRecord
from .requirements import PRD, Requirement, NFR
from .decisions import DecisionRecord, ArchitectureDecisionRecord
from .risk_security import TechnicalDebtRecord, RiskRecord, SecurityRecord

# Typed record classes (Phase 4: PRD, REQ, NFR; Phase 5: DR, ADR; Phase 6: TDR, RSK, SEC)
TYPED_RECORD_CLASSES: Dict[str, Type] = {
    "PRD": PRD,
    "REQ": Requirement,
    "NFR": NFR,
    "DR": DecisionRecord,
    "ADR": ArchitectureDecisionRecord,
    "TDR": TechnicalDebtRecord,
    "RSK": RiskRecord,
    "SEC": SecurityRecord,
    # RCA deferred to Phase 7
}

# Full registry including untyped (base class) entries
RECORD_TYPE_CLASSES: Dict[str, Type] = {
    **TYPED_RECORD_CLASSES,
    "RCA": EngineeringRecord,
}
