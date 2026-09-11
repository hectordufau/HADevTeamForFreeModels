"""
Persistent Memory for V2.1: project, task, agent, and lessons memory.
"""

import os
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MemoryEntry:
    key: str
    content: str
    category: str  # project, task, agent, lesson
    tags: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class MemoryManager:
    """Manages persistent memory across executions."""

    def __init__(self, storage_dir: str = ""):
        if not storage_dir:
            storage_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "artifacts", "memory"
            )
        self.storage_dir = storage_dir
        self._entry_type = MemoryEntry
        os.makedirs(storage_dir, exist_ok=True)
        for sub in ["project", "task", "agent", "lessons"]:
            os.makedirs(os.path.join(storage_dir, sub), exist_ok=True)

    def store(self, entry: MemoryEntry):
        """Store a memory entry as JSON."""
        category_dir = os.path.join(self.storage_dir, entry.category)
        path = os.path.join(category_dir, f"{entry.key}.json")
        with open(path, "w") as f:
            json.dump({
                "key": entry.key,
                "content": entry.content,
                "category": entry.category,
                "tags": entry.tags,
                "timestamp": entry.timestamp,
            }, f, indent=2)

    def retrieve(self, key: str, category: str = "") -> Optional[MemoryEntry]:
        """Retrieve a specific memory entry."""
        if category:
            search_dirs = [os.path.join(self.storage_dir, category)]
        else:
            search_dirs = [
                os.path.join(self.storage_dir, d)
                for d in ["project", "task", "agent", "lessons"]
            ]

        for d in search_dirs:
            path = os.path.join(d, f"{key}.json")
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                return MemoryEntry(**data)
        return None

    def search(self, query: str, category: str = "") -> List[MemoryEntry]:
        """Search memory entries by content (simple substring match)."""
        results = []
        if category:
            dirs = [os.path.join(self.storage_dir, category)]
        else:
            dirs = [
                os.path.join(self.storage_dir, d)
                for d in ["project", "task", "agent", "lessons"]
            ]

        for d in dirs:
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if not fname.endswith(".json"):
                    continue
                with open(os.path.join(d, fname)) as f:
                    try:
                        data = json.load(f)
                        if query.lower() in json.dumps(data).lower():
                            results.append(MemoryEntry(**data))
                    except (json.JSONDecodeError, KeyError):
                        continue
        return results

    def store_adr(self, adr_id: str, context: str, decision: str,
                  alternatives: str, consequences: str) -> str:
        """Store an ADR as both memory entry and markdown file."""
        entry = MemoryEntry(
            key=adr_id,
            content=f"Context: {context}\nDecision: {decision}\nAlternatives: {alternatives}\nConsequences: {consequences}",
            category="lessons",
            tags=["adr", adr_id],
        )
        self.store(entry)

        # Markdown file
        adr_dir = os.path.join(
            os.path.dirname(os.path.dirname(self.storage_dir)),
            "docs", "adr"
        )
        os.makedirs(adr_dir, exist_ok=True)
        md_path = os.path.join(adr_dir, f"{adr_id}.md")
        with open(md_path, "w") as f:
            f.write(f"# {adr_id}\n\n")
            f.write(f"## Context\n\n{context}\n\n")
            f.write(f"## Decision\n\n{decision}\n\n")
            f.write(f"## Alternatives\n\n{alternatives}\n\n")
            f.write(f"## Consequences\n\n{consequences}\n")
        return md_path
