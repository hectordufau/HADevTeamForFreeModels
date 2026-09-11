# harness/context/v3.py — Context Manager V3 (Phase K)
"""
Context Manager V3 with relevance scoring and budget management.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json


class ContextV3Error(Exception):
    """Raised on Context V3 errors."""


@dataclass
class ContextItem:
    """A single item in the context window."""
    content: str
    source: str  # task, memory, evidence, system
    relevance: float = 0.5  # 0.0 to 1.0
    priority: int = 5  # 1 (highest) to 10 (lowest)
    category: str = ""
    estimated_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "content": self.content[:100],
            "source": self.source,
            "relevance": self.relevance,
            "priority": self.priority,
            "category": self.category,
            "estimated_tokens": self.estimated_tokens,
        }


@dataclass
class ContextBudget:
    """Budget for context window management."""
    def __init__(self, max_tokens: int = 128000, reserved_tokens: int = None):
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_tokens if reserved_tokens is not None else min(max_tokens // 4, 32000)
        self.available_tokens = self.max_tokens - self.reserved_tokens
        self.used_tokens = 0

    def can_add(self, item: ContextItem) -> bool:
        """Check if an item can be added within budget."""
        return self.used_tokens + item.estimated_tokens <= self.available_tokens

    def add(self, item: ContextItem):
        """Add item to budget tracking."""
        self.used_tokens += item.estimated_tokens

    def remove(self, item: ContextItem):
        """Remove item from budget tracking."""
        self.used_tokens = max(0, self.used_tokens - item.estimated_tokens)

    def remaining(self) -> int:
        """Get remaining available tokens."""
        return self.available_tokens - self.used_tokens

    def to_dict(self) -> dict:
        return {
            "max_tokens": self.max_tokens,
            "reserved_tokens": self.reserved_tokens,
            "available_tokens": self.available_tokens,
            "used_tokens": self.used_tokens,
            "remaining": self.remaining(),
        }


class ContextRelevanceScorer:
    """Scores context items for relevance to the current task."""

    @staticmethod
    def score(item: Any, query: str) -> float:
        """Score relevance of a context item to a query.
        
        Returns 0.0 to 1.0 based on keyword overlap and source authority.
        """
        if not query or not item:
            return 0.0

        query_lower = query.lower()
        query_words = set(query_lower.split())

        content = ""
        if hasattr(item, 'content'):
            content = item.content
        elif hasattr(item, 'to_dict'):
            content = json.dumps(item.to_dict())
        content_lower = content.lower()

        # Keyword overlap score
        matched = sum(1 for w in query_words if w in content_lower)
        keyword_score = matched / max(len(query_words), 1) if query_words else 0.0

        # Source authority bonus
        source = ""
        if hasattr(item, 'source'):
            source = item.source
        source_bonus = {
            "task": 0.2,
            "system": 0.15,
            "evidence": 0.1,
            "memory": 0.05,
        }.get(source, 0.0)

        return min(1.0, keyword_score * 0.7 + source_bonus)


class ContextBuilderV3:
    """Builds optimized context packs within budget constraints.

    Features:
    - Relevance scoring for each context item
    - Budget management to prevent context overflow
    - Priority-based inclusion
    - Source-aware ordering
    """

    def __init__(self, project_root: str = "",
                 max_tokens: int = 128000):
        self.project_root = project_root
        self.budget = ContextBudget(max_tokens=max_tokens)

    def build_context(self, task: Any, agent_role: str,
                     agent_soul: str = "",
                     relevant_files: Optional[List[str]] = None,
                     memory_context: Optional[List[Any]] = None,
                     evidence_context: Optional[List[Any]] = None) -> 'ContextPackV3':
        """Build an optimized context pack within budget."""
        pack = ContextPackV3(
            agent_role=agent_role,
            budget=self.budget,
        )

        # Always include system context (reserved tokens)
        system_item = ContextItem(
            content=f"HADevTeamForFreeModels: Capability-Driven Engineering Harness\n"
                    f"Role: {agent_role}\n"
                    f"Mode: software-engineering\n"
                    f"Model Policy: Free models only (cost=0)",
            source="system",
            relevance=1.0,
            priority=1,
            category="system",
            estimated_tokens=100,
        )
        if self.budget.can_add(system_item):
            pack.add(system_item)

        # Include agent soul
        if agent_soul:
            soul_item = ContextItem(
                content=agent_soul[:2000],
                source="system",
                relevance=0.9,
                priority=2,
                category="agent",
                estimated_tokens=min(len(agent_soul) // 4, 500),
            )
            if self.budget.can_add(soul_item):
                pack.add(soul_item)

        # Include task context
        task_obj = ""
        if hasattr(task, 'spec') and hasattr(task.spec, 'objective'):
            task_obj = task.spec.objective
        if task_obj:
            task_item = ContextItem(
                content=f"Task: {task_obj}",
                source="task",
                relevance=0.9,
                priority=1,
                category="task",
                estimated_tokens=len(task_obj) // 4,
            )
            if self.budget.can_add(task_item):
                pack.add(task_item)

        # Include relevant file information
        if relevant_files:
            file_item = ContextItem(
                content=f"Allowed changes: {', '.join(relevant_files)}",
                source="task",
                relevance=0.7,
                priority=3,
                category="files",
                estimated_tokens=len(relevant_files) * 5,
            )
            if self.budget.can_add(file_item):
                pack.add(file_item)

        # Include relevant memory (scored and filtered)
        memory_score = ContextRelevanceScorer()
        if memory_context:
            for mem in memory_context[:20]:  # limit to 20 memories
                relevance = memory_score.score(mem, task_obj)
                if relevance > 0.3:  # only include if relevant enough
                    mem_item = ContextItem(
                        content=str(getattr(mem, 'content', str(mem)))[:500],
                        source="memory",
                        relevance=relevance,
                        priority=6,
                        category="lessons",
                        estimated_tokens=min(len(str(getattr(mem, 'content', ''))) // 4, 125),
                    )
                    if self.budget.can_add(mem_item):
                        pack.add(mem_item)

        # Include evidence context
        if evidence_context:
            for ev in evidence_context[:10]:
                ev_item = ContextItem(
                    content=str(getattr(ev, 'content', str(ev)))[:300],
                    source="evidence",
                    relevance=0.5,
                    priority=7,
                    category="evidence",
                    estimated_tokens=75,
                )
                if self.budget.can_add(ev_item):
                    pack.add(ev_item)

        # Sort by priority, then relevance
        pack.items.sort(key=lambda x: (x.priority, -x.relevance))

        return pack


class ContextPackV3:
    """An optimized context pack for an agent execution."""

    def __init__(self, agent_role: str, budget: ContextBudget):
        self.agent_role = agent_role
        self.budget = budget
        self.items: List[ContextItem] = []

    def add(self, item: ContextItem):
        """Add an item to the context pack."""
        self.items.append(item)
        self.budget.add(item)

    def to_dict(self) -> dict:
        return {
            "agent_role": self.agent_role,
            "budget": self.budget.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "total_items": len(self.items),
        }
