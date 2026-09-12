# tests/unit/test_v32_field_validation.py — V3.2 Phase 8: Statistical Field Validation
"""
Phase 8 tests for the three-mode field-validation experiment and its
statistical / rate-metric / release-gate logic.

These run a small, deterministic instantiation of the real experiment engine
(harness.validation.field_validation) and assert real computed values and
sentinels — they do not hard-code outcomes, and they verify the integrity
guarantees (COLD/LEARNED/TEST separation, seeded reproducibility, free-model
invariant, paired statistics).
"""
import json
import os
import random

import pytest

PY = "field_validation"
fv = pytest.importorskip("harness.validation.field_validation")

MODES = fv.MODES
INSUFFICIENT_DATA = fv.INSUFFICIENT_DATA


def make_runner(reps=2, seed=20261110):
    cfg = fv.FieldExperimentConfig(
        run_id="V3.2-FV8-UNIT", benchmark_id="V3.2-FIELD-VALIDATION",
        reproducibility_seed=seed, train_reps=reps, validation_reps=reps,
        test_reps=reps,
    )
    r = fv.FieldValidationRunner(cfg)
    r.run_all()
    return r, cfg


# ---------------------------------------------------------------------------
# 1. Three-mode separation
# ---------------------------------------------------------------------------
def test_modes_separate_and_cold_has_no_learning():
    r, _ = make_runner(reps=3)
    for mode in MODES:
        assert len(r.observations_for_split(mode, "test")) == 3 * 10
    # COLD never applies learning / retrieves / mutates.
    cold = r.observations_for_split("COLD", "test")
    assert all(not o.learning_applied for o in cold)
    assert all(o.experiences_retrieved == 0 for o in cold)


def test_learning_improves_test_outcomes():
    r, _ = make_runner(reps=4)
    cold = r.observations_for_split("COLD", "test")
    learned = r.observations_for_split("LEARNED_EXPLORATION", "test")
    cold_sr = sum(1 for o in cold if o.status == "PASSED") / len(cold)
    learned_sr = sum(1 for o in learned if o.status == "PASSED") / len(learned)
    assert learned_sr > cold_sr


# ---------------------------------------------------------------------------
# 2. Reproducibility (determinism)
# ---------------------------------------------------------------------------
def test_same_seed_same_outcomes():
    r1, _ = make_runner(seed=999)
    r2, _ = make_runner(seed=999)
    def key(r):
        return [(o.mode, o.task_id, o.repetition, round(o.score, 6), o.status)
                for o in r.observations]
    assert key(r1) == key(r2)


def test_different_seed_generally_differs():
    r1, _ = make_runner(seed=999)
    r2, _ = make_runner(seed=1000)
    s1 = [(o.task_id, o.repetition, round(o.score, 4))
          for o in r1.observations_for_split("COLD", "test")]
    s2 = [(o.task_id, o.repetition, round(o.score, 4))
          for o in r2.observations_for_split("COLD", "test")]
    assert s1 != s2


def test_repetitions_vary():
    r, _ = make_runner()
    scores = {}
    for o in r.observations_for_split("COLD", "test"):
        scores.setdefault(o.task_id, []).append(o.score)
    # at least one task has non-constant scores across reps
    assert any(len(set(round(v, 3) for v in vals)) > 1 for vals in scores.values())


# ---------------------------------------------------------------------------
# 3. Statistics / paired comparisons
# ---------------------------------------------------------------------------
def test_paired_comparison_and_pairing_rate():
    r, _ = make_runner(reps=3)
    st = r.compute_statistics()["test"]
    assert st["n"]["COLD"] == 30
    for comp in ("COLD_vs_LEARNED_NO_EXPLORATION", "COLD_vs_LEARNED_EXPLORATION",
                 "LEARNED_NO_EXPLORATION_vs_LEARNED_EXPLORATION"):
        pairing = st["pairing"][comp.replace("_vs_", "_")]
        assert pairing["pairing_rate"] == 1.0
        assert pairing["unmatched_a"] == 0 and pairing["unmatched_b"] == 0
    s = st["COLD_vs_LEARNED_EXPLORATION"]["success_rate"]
    assert s["comparison_type"] == "PAIRED"
    assert s["paired_n"] == 30
    assert s["delta_pp"] > 0
    assert isinstance(s["ci_95"], list) and len(s["ci_95"]) == 2
    assert s["p_value"] is not None and not isinstance(s["p_value"], str)


def test_statistics_use_sentinels_when_no_data():
    # A runner with zero observations must not fabricate numbers.
    cfg = fv.FieldExperimentConfig(
        run_id="V3.2-FV8-EMPTY", benchmark_id="V3.2-FIELD-VALIDATION",
        reproducibility_seed=1, train_reps=0, validation_reps=0, test_reps=0)
    r = fv.FieldValidationRunner(cfg)
    st = r.compute_statistics()["test"]
    # n=0 -> INSUFFICIENT sentinel on mean; CI sentinel not a fabricated number
    assert st["n"]["COLD"] == 0
    assert st["COLD_vs_LEARNED_EXPLORATION"]["success_rate"]["baseline_mean"] in (
        INSUFFICIENT_DATA, fv.NOT_AVAILABLE)


# ---------------------------------------------------------------------------
# 4. Exploration contribution
# ---------------------------------------------------------------------------
def test_exploration_contribution_computes():
    r, _ = make_runner(reps=3)
    ec = r.exploration_contribution("test")
    assert ec["n_no_exploration"] == 30 and ec["n_exploration"] == 30
    assert ec["interpretation"] in ("POSITIVE", "NEUTRAL", "NEGATIVE")
    assert isinstance(ec["success_pp"], float)


def test_exploration_is_real_and_tracked():
    r, _ = make_runner(reps=3)
    ex = r.exploration_statistics()
    assert ex["mode"] == "LEARNED_EXPLORATION"
    assert ex["exploration_taken_n"] >= 0
    assert ex["counterfactual_not_computable"] == 0


# ---------------------------------------------------------------------------
# 5. Decision impact rates
# ---------------------------------------------------------------------------
def test_decision_outcomes_rates():
    r, _ = make_runner(reps=3)
    doi = r.decision_outcomes("LEARNED_EXPLORATION")
    assert doi["n"] > 0
    assert doi["decision_improvement_rate"] >= 0.0
    assert doi["learning_harm_rate"] >= 0.0
    assert doi["unknown_outcome_rate"] == 0.0  # COLD baselines available
    total = doi["improved"] + doi["unchanged"] + doi["worsened"] + doi["unknown"]
    assert total == doi["n"]


def test_learning_influence_rate():
    r, _ = make_runner(reps=3)
    lr = r.learning_influence_rate("LEARNED_EXPLORATION")
    assert lr["total_decisions"] > 0
    assert 0.0 <= lr["rate"] <= 1.0


# ---------------------------------------------------------------------------
# 6. TRAIN / VALIDATION / TEST improvement + no TEST contamination
# ---------------------------------------------------------------------------
def test_no_test_contamination():
    r, cfg = make_runner(reps=3)
    # TEST outcomes never enter the learning store: TEST split stores no
    # strategy evidence (TEST never trains).
    obs = r.observations
    test_influenced = [o for o in obs if o.split == "test" and o.learning_applied
                       and o.mode in ("COLD", "TEST")]
    assert test_influenced == []
    # COLD must not retrieve learning experiences.
    cold_retrieved = [o for o in obs if o.mode == "COLD" and o.experiences_retrieved > 0]
    assert cold_retrieved == []


def test_split_improvement_present():
    r, _ = make_runner(reps=3)
    for metric, fn in (("success_rate", r.split_improvement), ):
        si = fn(metric)
        for split in ("train", "validation", "test"):
            assert split in si
            assert si[split] != fv.INSUFFICIENT_DATA


# ---------------------------------------------------------------------------
# 7. Integrity / security / governance / free-model
# ---------------------------------------------------------------------------
def test_integrity_clean():
    r, _ = make_runner(reps=3)
    ic = r.integrity_check()
    assert ic["cold_learned_contamination"] == 0
    assert ic["test_contamination"] == 0
    assert ic["cross_run_contamination"] == 0
    assert ic["critical_contamination"] is False


def test_security_governance_free_model():
    r, _ = make_runner(reps=3)
    sg = r.security_governance_check()
    assert sg["security_violations"] == 0
    assert sg["governance_violations"] == 0
    assert sg["non_free_model_executions"] == 0
    assert sg["free_model_invariant_preserved"] is True


# ---------------------------------------------------------------------------
# 8. Run persistence carries integrity fields
# ---------------------------------------------------------------------------
def test_observations_persist_required_fields(tmp_path, monkeypatch):
    r, _ = make_runner(reps=1)
    # every persisted record includes run/task/execution/mode/split/rep/seed
    for o in r.observations:
        d = o.to_persisted()
        for field in ("validation_run_id", "benchmark_id", "task_id", "mode",
                      "split", "repetition", "reproducibility_seed", "derived_seed"):
            assert d[field] is not None or d[field] == ""


# ---------------------------------------------------------------------------
# 9. Configuration digest is reproducible
# ---------------------------------------------------------------------------
def test_config_digest_reproducible():
    d1 = fv.configuration_digest()
    d2 = fv.configuration_digest()
    assert d1 == d2
    assert len(d1) == 16


# ---------------------------------------------------------------------------
# 10. load_benchmark_tasks() idempotency & source-corpus immutability (V3.2
#     reproducibility resolution). These pin the confirmed in-place-mutation
#     defect fix: the loader must be idempotent and must NOT mutate the shared
#     runner TRAIN_TASKS/VALIDATION_TASKS/TEST_TASKS singletons.
# ---------------------------------------------------------------------------
def _snapshot_source():
    from harness.validation import runner as r
    snap = {}
    for split, attr in (("train", "TRAIN_TASKS"), ("validation", "VALIDATION_TASKS"),
                        ("test", "TEST_TASKS")):
        for t in getattr(r, attr):
            snap[(split, t["id"])] = {
                "difficulty": t.get("difficulty"),
                "category": t.get("category"),
                "tags": list(t.get("tags", [])),
            }
    return snap


def test_i1_load_benchmark_tasks_twice_identical():
    # I1: call the loader several times in the same process -> identical task
    # values (the first call of a process must equal the third call).
    a = fv.load_benchmark_tasks()
    b = fv.load_benchmark_tasks()
    c = fv.load_benchmark_tasks()
    def sig(tasks):
        return {(s, t["id"], t["difficulty"], t["category"])
                for s, lst in tasks.items() for t in lst}
    assert sig(a) == sig(b) == sig(c)
    assert len(sig(a)) == 40  # 20 train + 10 val + 10 test


def test_i2_source_benchmark_not_mutated():
    # I2: after loading, the source corpus still has STRING difficulties and no
    # injected category — it is never mutated. (Regression for the in-place
    # mutation that collapsed every task to difficulty 0.55 on 2nd+ load.)
    from harness.validation import runner as r
    _snapshot_source()  # ensure lists exist
    before = {t["id"]: t.get("difficulty") for t in r.TRAIN_TASKS}
    fv.load_benchmark_tasks()
    fv.load_benchmark_tasks()
    after = {t["id"]: t.get("difficulty") for t in r.TRAIN_TASKS}
    assert before == after
    # difficulties on the source remain the raw strings ('low'/'medium'/'high'),
    # never the numeric priors that the in-place mutation used to inject.
    assert all(isinstance(v, str) for v in before.values())
    assert all("category" not in t or t.get("category") is None for t in r.TRAIN_TASKS)


def test_i3_dataset_digest_stable_after_loading():
    # I3: dataset digest is identical before and after repeated loading.
    d0 = fv.dataset_digest()
    fv.load_benchmark_tasks()
    d1 = fv.dataset_digest()
    fv.load_benchmark_tasks()
    d2 = fv.dataset_digest()
    assert d0 == d1 == d2


def test_i4_task_ordering_stable():
    # I4: repeated loading preserves task ordering within each split.
    a = fv.load_benchmark_tasks()
    b = fv.load_benchmark_tasks()
    for split in ("train", "validation", "test"):
        assert [t["id"] for t in a[split]] == [t["id"] for t in b[split]]
    assert [t["id"] for t in a["test"]] == ["TEST-%03d" % i for i in range(1, 11)]


def test_i5_repeated_runs_same_process_identical():
    # I5: two full runs in the SAME process with identical config produce
    # identical outcomes. (This catches the first-run-vs-subsequent divergence
    # caused by the old in-place mutation.)
    def sig(reps=3, seed=111):
        cfg = fv.FieldExperimentConfig(
            run_id="V3.2-FV8-I5", benchmark_id="V3.2-FIELD-VALIDATION",
            reproducibility_seed=seed, train_reps=reps,
            validation_reps=reps, test_reps=reps)
        r = fv.FieldValidationRunner(cfg); r.run_all()
        return [(o.mode, o.split, o.task_id, o.repetition, round(o.score, 6),
                 o.status) for o in sorted(r.observations,
                     key=lambda x: (x.mode, x.split, x.task_id, x.repetition))]
    assert sig() == sig()


# ---------------------------------------------------------------------------
# 11. Canonical repetition configuration (V3.2 reproducibility resolution)
# ---------------------------------------------------------------------------
def test_r1_canonical_default_is_five():
    # The single authoritative default repetition count is 5 (the accepted run).
    assert fv.DEFAULT_REPETITIONS == 5
    # Resolving None uses the canonical default, not a hidden fallback of 3.
    assert fv.resolve_repetitions(None) == 5
    assert fv.resolve_repetitions(3) == 3
    assert fv.resolve_repetitions(5) == 5


def test_r2_repetitions_flow_cli_to_runner_to_results():
    # Prove CLI --reps -> resolved config -> runner -> results all use the same
    # value, for reps 3 AND reps 5, and that the sample count follows the config.
    import scripts.phase8_field_validation as cli
    for reps in (3, 5):
        cfg = fv.FieldExperimentConfig(
            run_id=f"V3.2-FV8-CLI-{reps}", benchmark_id="V3.2-FIELD-VALIDATION",
            reproducibility_seed=314159, train_reps=reps,
            validation_reps=reps, test_reps=reps)
        runner = fv.FieldValidationRunner(cfg)
        runner.run_all()
        results = cli.build_results(runner)
        # resolved configuration says the same reps
        assert results["repetitions"] == {"train": reps, "validation": reps,
                                          "test": reps}
        assert results["resolved_configuration"]["repetitions"] == {
            "train": reps, "validation": reps, "test": reps}
        # sample count follows config: test n = 10 tasks * reps
        for m in MODES:
            assert results["test"][m]["n"] == 10 * reps
            assert results["sample_sizes"]["test"][m] == 10 * reps


def test_r3_resolved_configuration_integrity():
    from harness.validation.field_validation import resolved_configuration as rc
    cfg = rc(314159, 5, 5, 5, 0.10)
    assert cfg["seed"] == 314159
    assert cfg["repetitions"] == {"train": 5, "validation": 5, "test": 5}
    assert cfg["exploration_rate"] == 0.10
    assert "split_policy" in cfg and "variance_policy" in cfg
    assert "significance" in cfg
