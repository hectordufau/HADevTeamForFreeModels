"""
ADR automation for V2.1: automatically generate ADRs from architecture decisions.

Phase 5 Update: ADRAutomation now creates canonical ArchitectureDecisionRecord
instances stored through KnowledgeStore. The old memory-based path is preserved
for backward compatibility.

Architecture:
- Legacy ADRAutomation API → adapter → Engineering Knowledge ADR
- Canonical ADR lives only in KnowledgeStore (no duplicate canonical store)
- Memory may reference ADRs but never contains independent ADR canonical truth
"""

import os
import tempfile
from typing import List, Optional

from harness.memory import MemoryManager


class ADRAutomation:
    """
    Automatically creates ADRs for architecture decisions.

    Compatibility adapter: creates canonical ArchitectureDecisionRecord instances
    while preserving the existing memory-based storage path for backward compat.

    Agent-generated ADRs always start as PROPOSED (never auto-ACCEPTED).
    """

    def __init__(self, memory: MemoryManager, knowledge_store_path: Optional[str] = None):
        """
        Initialize ADRAutomation.

        Args:
            memory: MemoryManager instance (for backward compat storage)
            knowledge_store_path: Path to KnowledgeStore SQLite database.
                                  If None, uses a temp file (records are still
                                  canonical, but don't persist across restarts).
        """
        self.memory = memory
        self._knowledge_store_path = knowledge_store_path
        self._knowledge_store = None

    def _get_store(self):
        """Lazy-initialize KnowledgeStore."""
        if self._knowledge_store is None:
            from harness.knowledge import KnowledgeStore
            path = self._knowledge_store_path
            if path is None:
                # Temp file for non-persistent usage
                fd, path = tempfile.mkstemp(suffix=".db", prefix="adr_knowledge_")
                os.close(fd)
            self._knowledge_store = KnowledgeStore(path)
        return self._knowledge_store

    def record_decision(self, adr_id: str, context: str, decision: str,
                        alternatives: str, consequences: str) -> str:
        """
        Record an architecture decision as a canonical ADR.

        Creates an ArchitectureDecisionRecord stored in KnowledgeStore.
        Also stores via legacy MemoryManager for backward compatibility.

        Args:
            adr_id: ADR identifier (e.g., 'ADR-001')
            context: Decision context
            decision: The decision made
            alternatives: Alternatives considered (stored as text)
            consequences: Expected consequences

        Returns:
            Path to the markdown ADR file (for backward compatibility)
        """
        store = self._get_store()

        from harness.knowledge import (
            ArchitectureDecisionRecord,
            Provenance,
            Alternative,
            Consequences,
        )

        # Parse alternatives text into structured alternatives
        alt_list = []
        if alternatives:
            for i, alt_text in enumerate(
                [a.strip() for a in alternatives.split(";") if a.strip()],
                start=1,
            ):
                alt_list.append(Alternative(
                    alt_id=f"ALT-{i:03d}",
                    description=alt_text,
                    disposition="REJECTED",
                ))

        # Parse consequences text
        cons = Consequences(
            positive=[consequences] if consequences else [],
            negative=[],
            risks=[],
        )

        # Create canonical ADR record
        adr = ArchitectureDecisionRecord(
            record_id=adr_id,
            title=f"ADR: {decision[:80]}",
            description=f"Context: {context}\nDecision: {decision}",
            status="proposed",  # Agent-generated = PROPOSED, never auto-ACCEPTED
            authority="proposed",
            decision_type="ARCHITECTURE",
            context=context,
            decision=decision,
            alternatives=alt_list,
            rationale="Agent-generated ADR (auto-recorded)",
            consequences=cons,
            architecture_domain="infrastructure",  # Default domain
            patterns_considered=[],
            pattern_selected="",
            trade_offs="",
            impact="",
            provenance=Provenance(
                author="agent",
                source="generated",
            ),
            tags=["adr", "agent-generated", adr_id],
        )

        store.save(adr)

        # Also store via legacy path for backward compat
        md_path = self.memory.store_adr(adr_id, context, decision, alternatives, consequences)

        return md_path

    def get_relevant_adrs(self, task_context: str) -> list:
        """
        Get ADRs relevant to the current task context.

        Queries the KnowledgeStore for ADR records matching the task context.
        Falls back to legacy memory search if KnowledgeStore is unavailable.

        Args:
            task_context: Task description to match against ADRs

        Returns:
            List of ADR records (or MemoryEntry legacy results on fallback)
        """
        store = self._get_store()

        try:
            from harness.knowledge.records import EngineeringRecord
            # Query KnowledgeStore for ADRs
            all_records = store.list_all()
            # Use getattr to detect typed ADR instances (works with base EngineeringRecord too)
            adr_records = [
                r for r in all_records
                if getattr(r, 'record_type', None) == "ADR"
                or getattr(r, 'architecture_domain', None) is not None
            ]

            # Simple relevance: context match
            results = []
            task_lower = task_context.lower()
            for adr in adr_records:
                if (task_lower in adr.context.lower() or
                    task_lower in adr.decision.lower() or
                    task_lower in adr.title.lower()):
                    results.append(adr)

            return results
        except Exception:
            # Fallback: legacy memory search
            results = self.memory.search(task_context, category="lessons")
            return [r for r in results if "adr" in r.tags]
