import json
from pathlib import Path

import pytest

from harness.validation.phase16_preexperiment import (
    CATEGORIES, KNOWLEDGE_LEVELS, MODES, Phase16Config, IsolatedRunner,
    Metrics, analyze_effect, load_benchmark, load_knowledge, short_digest,
    synthetic_golden_cases,
)


def test_benchmark_is_balanced_and_has_no_answer_leakage():
    tasks = load_benchmark()
    assert len(tasks) == 20
    assert {t["category"] for t in tasks} == set(CATEGORIES)
    assert {t["knowledge_level"] for t in tasks} == set(KNOWLEDGE_LEVELS)
    assert all(sum(t["knowledge_level"] == level for t in tasks) == 5 for level in KNOWLEDGE_LEVELS)
    assert all(not ({"answer", "solution", "expected_answer"} & set(t)) for t in tasks)


def test_knowledge_dataset_has_no_answer_leakage():
    items = load_knowledge()
    assert len(items) == 20
    assert all(not ({"answer", "solution", "expected_answer"} & set(item)) for item in items)


def test_loaders_return_immutable_input_copies():
    first = load_benchmark()
    first[0]["prompt"] = "mutated"
    assert load_benchmark()[0]["prompt"] != "mutated"


def test_digests_are_sha256_and_seed_is_deterministic():
    config = Phase16Config()
    a = Metrics.execution_seed(config, "P16-001", "CONTROL", 1)
    b = Metrics.execution_seed(config, "P16-001", "CONTROL", 1)
    assert a == b
    assert len(short_digest(load_benchmark())) == 16


def test_runner_has_three_isolated_modes_and_control_is_cold():
    runner = IsolatedRunner(Phase16Config())
    observations = runner.run()
    assert {o.mode for o in observations} == set(MODES)
    isolation = runner.isolation_check()
    assert isolation["control_knowledge"] == 0
    assert isolation["control_learning"] == 0
    assert isolation["knowledge_only_learning"] == 0
    assert isolation["distinct_mode_state"] is True


def test_runner_cannot_execute_accepted_run():
    with pytest.raises(RuntimeError, match="accepted stages"):
        IsolatedRunner(Phase16Config(accepted_run=True)).run()


def test_observations_are_attributed_and_aggregatable():
    runner = IsolatedRunner(Phase16Config())
    observations = runner.run()
    assert all(o.run_id == "V3.3-P16-PILOT" and o.split == "pilot" for o in observations)
    aggregate = Metrics.aggregate(observations)
    assert aggregate["n"] == 60
    assert Metrics.observation_digest(observations) == Metrics.observation_digest(observations)


def test_synthetic_statistics_golden_positive_neutral_negative():
    cases = synthetic_golden_cases()
    assert cases["positive"]["classification"] == "POSITIVE"
    assert cases["neutral"]["classification"] == "NEUTRAL"
    assert cases["negative"]["classification"] == "NEGATIVE"


def test_statistics_are_predeclared_not_tuned_by_sample_size():
    assert analyze_effect([0.5], [0.56])["classification"] == "POSITIVE"
    assert analyze_effect([0.5, 0.5], [0.54, 0.54])["classification"] == "NEUTRAL"


def test_pilot_artifact_is_machine_readable(tmp_path, monkeypatch):
    import harness.validation.phase16_preexperiment as p16
    monkeypatch.setattr(p16, "PHASE16_DIR", tmp_path)
    path = p16.write_pilot(Phase16Config())
    data = json.loads(path.read_text())
    assert data["schema_version"] == "v3.3-phase16-pilot"
    assert data["status"] == "PILOT_ONLY"
    assert data["accepted_run_executed"] is False
    assert len(data["observations"]) == 60
