"""Tests for Context V3 (Phase K)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.context.v3 import (
    ContextBuilderV3, ContextRelevanceScorer, ContextBudget,
    ContextItem, ContextPackV3, ContextV3Error,
)
from harness.memory import MemoryManager


class MockTask:
    def __init__(self, objective="Build a feature"):
        self.spec = type('obj', (object,), {
            'objective': objective,
        })


def test_context_budget():
    budget = ContextBudget(max_tokens=1000)
    assert budget.available_tokens == 750  # 1000 - 250 reserved
    assert budget.remaining() == 750

    item = ContextItem(content="test", source="task", estimated_tokens=100)
    assert budget.can_add(item)
    budget.add(item)
    assert budget.remaining() == 650

    budget.remove(item)
    assert budget.remaining() == 750


def test_context_relevance_scorer():
    item = ContextItem(content="Implement user authentication API", source="task")
    score = ContextRelevanceScorer.score(item, "authentication API")
    assert score > 0.0

    # Unrelated
    score2 = ContextRelevanceScorer.score(item, "database schema")
    assert score2 < score


def test_context_builder():
    builder = ContextBuilderV3(project_root="/tmp")
    task = MockTask(objective="Implement a new REST API endpoint")

    pack = builder.build_context(
        task=task,
        agent_role="coder",
        agent_soul="You are a senior software engineer...",
        relevant_files=["src/api/**", "tests/**"],
    )

    assert isinstance(pack, ContextPackV3)
    assert pack.agent_role == "coder"
    assert len(pack.items) > 0

    # System context should be present
    sources = [item.source for item in pack.items]
    assert "system" in sources


def test_context_pack():
    budget = ContextBudget(max_tokens=10000)
    pack = ContextPackV3(agent_role="tester", budget=budget)

    item1 = ContextItem(content="Test item", source="task", relevance=0.9, priority=1)
    item2 = ContextItem(content="Memory item", source="memory", relevance=0.5, priority=5)

    pack.add(item1)
    pack.add(item2)

    assert len(pack.items) == 2
    assert pack.to_dict()["total_items"] == 2
