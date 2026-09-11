"""Tests for Performance Registry V3 (Phase E)."""

import sys, os, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.routing.performance_v3 import (
    PerformanceRegistryV3, PerformanceSample, AggregatedStats,
    MIN_SAMPLES_REQUIRED,
)


def make_sample(model_id="m1", capability="test_cap", success=True, score=0.9, latency=100.0):
    return PerformanceSample(
        task_id="T1", model_id=model_id, capability=capability,
        success=success, score=score, latency_ms=latency, iterations=1,
    )


def test_performance_sample():
    s = make_sample()
    assert s.model_id == "m1"
    assert s.success is True
    assert s.score == 0.9


def test_record_and_stats():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        fname = f.name

    try:
        reg = PerformanceRegistryV3(storage_path=fname)
        for i in range(5):
            reg.record(make_sample(capability="testing", score=0.8 + i * 0.05))

        stats = reg.get_stats("m1", "testing")
        assert stats is not None
        assert stats.sample_count == 5
        assert stats.min_samples_met is True
        assert 0.8 <= stats.avg_score <= 1.0
    finally:
        os.unlink(fname)


def test_min_samples_policy():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        fname = f.name

    try:
        reg = PerformanceRegistryV3(storage_path=fname)
        # Record only 3 samples (below MIN_SAMPLES_REQUIRED=5)
        for i in range(3):
            reg.record(make_sample())

        stats = reg.get_stats("m1", "test_cap")
        assert stats is not None
        assert stats.sample_count == 3
        assert stats.min_samples_met is False

        # Should not appear in ranked results
        ranked = reg.rank_models("test_cap", min_samples=5)
        assert len(ranked) == 0
    finally:
        os.unlink(fname)


def test_rank_models():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        fname = f.name

    try:
        reg = PerformanceRegistryV3(storage_path=fname)

        # Model A: high score but few samples
        for i in range(2):
            reg.record(make_sample(model_id="model_a", score=0.95))

        # Model B: sufficient samples, moderate score
        for i in range(5):
            reg.record(make_sample(model_id="model_b", score=0.85))

        # Model C: fewer samples but higher score
        for i in range(3):
            reg.record(make_sample(model_id="model_c", score=0.90))

        ranked = reg.rank_models("test_cap", min_samples=5)
        assert len(ranked) >= 1
        # Only model_b has sufficient samples
        assert ranked[0].model_id == "model_b"

        # Lower minimum samples to include more
        ranked2 = reg.rank_models("test_cap", min_samples=2)
        assert len(ranked2) >= 2
    finally:
        os.unlink(fname)


def test_confidence_score():
    stats = AggregatedStats(
        model_id="m1", capability="c1", sample_count=5,
        success_rate=0.8, avg_score=0.9, avg_latency_ms=100.0,
        avg_iterations=1, std_score=0.1, std_latency=20.0,
        min_samples_met=True,
    )
    confidence = stats.confidence_score()
    assert confidence == 1.0  # Exactly at threshold

    stats2 = AggregatedStats(
        model_id="m1", capability="c1", sample_count=2,
        success_rate=0.8, avg_score=0.9, avg_latency_ms=100.0,
        avg_iterations=1, std_score=0.1, std_latency=20.0,
        min_samples_met=False,
    )
    assert stats2.confidence_score() < 1.0


def test_best_model():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        fname = f.name

    try:
        reg = PerformanceRegistryV3(storage_path=fname)
        for i in range(5):
            reg.record(make_sample(model_id="best_model", capability="api", score=0.9))
            reg.record(make_sample(model_id="worst_model", capability="api", score=0.5))

        best = reg.best_model("api")
        assert best == "best_model"
    finally:
        os.unlink(fname)
