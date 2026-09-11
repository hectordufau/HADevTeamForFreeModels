"""Tests for Model Intelligence (Phase F)."""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.planner.model_intelligence import (
    AdaptiveModelPolicyV3, ModelProfileStore, ModelSpecializationProfile,
    ExperimentTracker, ModelExperiment, FreeModelInvariantEnforcer,
)
from harness.routing import ModelCatalog, ModelCandidate, ModelRouter


def make_catalog():
    catalog = ModelCatalog()
    catalog.register(ModelCandidate(
        model_id="model_a:free", provider="nous", cost=0,
        capabilities={"coding": 0.9, "testing": 0.7},
    ))
    catalog.register(ModelCandidate(
        model_id="model_b:free", provider="nous", cost=0,
        capabilities={"coding": 0.8, "architecture": 0.9},
    ))
    catalog.register(ModelCandidate(
        model_id="paid_model", provider="openai", cost=0.05,
        capabilities={"coding": 1.0},
    ))
    return catalog


def test_model_profile_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ModelProfileStore(storage_dir=os.path.join(tmpdir, "profiles"))
        profile = ModelSpecializationProfile(
            model_id="nous/model_x",
            capabilities={"coding": 0.9, "testing": 0.7},
            specialization_score=0.6,
            top_capabilities=["coding", "testing"],
        )
        store.save_profile(profile)

        loaded = store.load_profile("nous/model_x")
        assert loaded is not None
        assert loaded.model_id == "nous/model_x"
        assert loaded.specialization_score == 0.6


def test_adaptive_policy_exploration():
    catalog = make_catalog()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        fname = f.name

    try:
        from harness.routing.performance_v3 import PerformanceRegistryV3
        perf_reg = PerformanceRegistryV3(storage_path=fname)
        fallback = ModelRouter(catalog)

        policy = AdaptiveModelPolicyV3(catalog, perf_reg, fallback)

        # Force exploitation
        sel = policy.select(["coding"], role="coder", force_exploit=True)
        assert sel is not None
        assert hasattr(sel, 'model_id')
    finally:
        os.unlink(fname)


def test_experiment_tracker():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(storage_dir=os.path.join(tmpdir, "experiments"))
        exp = ModelExperiment(
            experiment_id="exp1", model_id="m1", capability="coding",
            task_id="T1", result_score=0.9, is_exploration=True,
        )
        tracker.record(exp)

        results = tracker.get_results("m1", "coding")
        assert len(results) == 1
        assert results[0].result_score == 0.9


def test_free_model_invariant():
    catalog = make_catalog()
    enforcer = FreeModelInvariantEnforcer(catalog)

    assert enforcer.enforce("model_a:free") is True
    assert enforcer.enforce("paid_model") is False
    assert enforcer.enforce("unknown:free") is True  # by naming convention
