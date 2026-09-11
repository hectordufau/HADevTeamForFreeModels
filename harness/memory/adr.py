"""
ADR automation for V2.1: automatically generate ADRs from architecture decisions.
"""

from harness.memory import MemoryManager


class ADRAutomation:
    """Automatically creates ADRs for architecture decisions."""

    def __init__(self, memory: MemoryManager):
        self.memory = memory

    def record_decision(self, adr_id: str, context: str, decision: str,
                        alternatives: str, consequences: str) -> str:
        """Record an architecture decision as ADR."""
        return self.memory.store_adr(adr_id, context, decision, alternatives, consequences)

    def get_relevant_adrs(self, task_context: str) -> list:
        """Get ADRs relevant to the current task context."""
        results = self.memory.search(task_context, category="lessons")
        return [r for r in results if "adr" in r.tags]
