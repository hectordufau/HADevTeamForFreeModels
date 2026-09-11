"""Unit tests for Memory and ADR modules."""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from harness.memory import MemoryManager, MemoryEntry
from harness.memory.adr import ADRAutomation


def test_memory_store_and_retrieve():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = MemoryManager(storage_dir=os.path.join(tmpdir, "memory"))
        entry = MemoryEntry(key="test-key", content="test content", category="project")
        memory.store(entry)
        retrieved = memory.retrieve("test-key", category="project")
        assert retrieved is not None
        assert retrieved.key == "test-key"
        assert retrieved.content == "test content"
        print("  ✓ store and retrieve works")


def test_memory_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = MemoryManager(storage_dir=os.path.join(tmpdir, "memory"))
        memory.store(MemoryEntry(key="k1", content="hello world", category="lessons"))
        memory.store(MemoryEntry(key="k2", content="goodbye world", category="lessons"))
        results = memory.search("hello")
        assert len(results) == 1
        assert results[0].key == "k1"
        print("  ✓ search works")


def test_adr_creation():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = MemoryManager(storage_dir=os.path.join(tmpdir, "memory"))
        adr = ADRAutomation(memory)
        md_path = adr.record_decision(
            adr_id="ADR-001",
            context="Need to choose auth method",
            decision="JWT",
            alternatives="OAuth, Session",
            consequences="More secure, more complex",
        )
        assert os.path.exists(md_path)
        retrieved = memory.retrieve("ADR-001", category="lessons")
        assert retrieved is not None
        assert "adr" in retrieved.tags
        print("  ✓ ADR creation works")


def test_memory_search_cross_category():
    with tempfile.TemporaryDirectory() as tmpdir:
        memory = MemoryManager(storage_dir=os.path.join(tmpdir, "memory"))
        memory.store(MemoryEntry(key="p1", content="project note", category="project"))
        memory.store(MemoryEntry(key="t1", content="task note", category="task"))
        memory.store(MemoryEntry(key="a1", content="agent note", category="agent"))
        results = memory.search("note")  # cross-category
        assert len(results) == 3
        print("  ✓ cross-category search works")


if __name__ == "__main__":
    print("\n=== Memory & ADR Tests ===\n")
    test_memory_store_and_retrieve()
    test_memory_search()
    test_adr_creation()
    test_memory_search_cross_category()
    print("\n✓ All memory tests passed!")
