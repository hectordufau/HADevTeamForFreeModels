# Tests for V2.1 features
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.routing.performance import ModelPerformanceRegistry, ModelTaskRecord, AdaptiveModelRouter
from harness.routing import ModelCatalog, ModelCandidate


def test_registry_record_and_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        registry = ModelPerformanceRegistry(db_path)

        # Record some tasks
        for i in range(5):
            registry.record(ModelTaskRecord(
                task_id=f"TASK-{i}",
                model_id="model-a:free",
                role="coder",
                success=(i % 2 == 0),
                score=0.8 + (i * 0.05),
                iterations=1 + (i % 3),
                latency_ms=1200 + (i * 100),
                capabilities_used=["coding", "debugging", "reasoning"],
            ))

        stats = registry.get_model_stats(model_id="model-a:free", role="coder")
        assert len(stats) == 1
        assert stats[0]["total"] == 5
        assert stats[0]["successful"] == 3  # even indices: 0, 2, 4


def test_registry_best_for_capability():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        registry = ModelPerformanceRegistry(db_path)

        registry.record(ModelTaskRecord(
            task_id="TASK-1", model_id="model-a:free", role="coder",
            success=True, score=0.9, iterations=1, latency_ms=1000,
            capabilities_used=["coding", "debugging", "reasoning"],
        ))
        registry.record(ModelTaskRecord(
            task_id="TASK-2", model_id="model-b:free", role="coder",
            success=False, score=0.5, iterations=3, latency_ms=2000,
            capabilities_used=["coding"],
        ))

        best = registry.get_best_model_for_capability("coding", min_records=1)
        assert best is not None
        assert best["model_id"] == "model-a:free"


def test_adaptive_router_selects_best():
    catalog = ModelCatalog()
    catalog.register(ModelCandidate(
        model_id="model-a:free", provider="nous", cost=0,
        capabilities={"coding": 0.75, "debugging": 0.7, "reasoning": 0.6},
    ))
    catalog.register(ModelCandidate(
        model_id="model-b:free", provider="nous", cost=0,
        capabilities={"coding": 0.7, "debugging": 0.65, "reasoning": 0.65},
    ))

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        registry = ModelPerformanceRegistry(db_path)

        # Record that model-b was better historically (and model-a is unreliable)
        registry.record(ModelTaskRecord(
            task_id="TASK-1", model_id="model-b:free", role="coder",
            success=True, score=0.95, iterations=1, latency_ms=800,
            capabilities_used=["coding", "debugging", "reasoning"],
        ))
        registry.record(ModelTaskRecord(
            task_id="TASK-2", model_id="model-b:free", role="coder",
            success=True, score=0.88, iterations=2, latency_ms=900,
            capabilities_used=["coding", "debugging", "reasoning"],
        ))
        registry.record(ModelTaskRecord(
            task_id="TASK-3", model_id="model-b:free", role="coder",
            success=True, score=0.92, iterations=1, latency_ms=850,
            capabilities_used=["coding", "debugging", "reasoning"],
        ))
        registry.record(ModelTaskRecord(
            task_id="TASK-4", model_id="model-a:free", role="coder",
            success=False, score=0.4, iterations=3, latency_ms=3000,
            capabilities_used=["coding", "debugging", "reasoning"],
        ))
        registry.record(ModelTaskRecord(
            task_id="TASK-5", model_id="model-a:free", role="coder",
            success=False, score=0.35, iterations=3, latency_ms=3500,
            capabilities_used=["coding", "debugging", "reasoning"],
        ))

        from harness.routing import ModelRouter
        fallback = ModelRouter(catalog)
        router = AdaptiveModelRouter(catalog, registry, fallback)

        selection = router.select(["coding", "debugging"], role="coder")
        # model-b has lower cap score but better history
        assert selection.model_id == "model-b:free"


if __name__ == "__main__":
    test_registry_record_and_stats()
    test_registry_best_for_capability()
    test_adaptive_router_selects_best()
    print("All V2.1 tests passed!")
