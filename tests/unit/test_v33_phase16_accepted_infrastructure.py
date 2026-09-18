import json

from harness.validation.phase16_accepted import (
    AcceptedRunConfig,
    AcceptedRunRunner,
    cluster_aware_analysis,
)


def test_accepted_config_is_frozen_without_executing():
    config = AcceptedRunConfig.frozen()
    assert config.repetitions == 10
    assert config.expected_executions == 600
    assert config.bootstrap_iterations == 10000
    assert config.permutation_iterations == 10000
    assert config.alpha == 0.05
    assert config.practical_threshold == 0.05
    assert config.statistical_unit == "task_level_paired"


def test_accepted_runner_preserves_cluster_and_repeated_observation_identity():
    config = AcceptedRunConfig.frozen()
    runner = AcceptedRunRunner(config)
    rows = [
        {"task_cluster_id": "P16-001", "task_id": "P16-001", "mode": mode,
         "repetition": repetition, "score": score}
        for mode, score in (("CONTROL", 0.40), ("KNOWLEDGE_ONLY", 0.60),
                            ("KNOWLEDGE_PLUS_LEARNING", 0.70))
        for repetition in (1, 2)
    ]
    analysis = cluster_aware_analysis(rows, config)
    assert analysis["raw_observation_count"] == 6
    assert analysis["task_cluster_count"] == 1
    assert analysis["repetitions_are_repeated_observations"] is True
    assert analysis["repetitions_counted_as_independent_n"] is False
    assert analysis["primary"]["cluster_count"] == 1
    assert analysis["primary"]["cluster_ids"] == ["P16-001"]
    assert analysis["primary"]["mean_paired_delta"] == 0.2
    assert runner.config.expected_executions == 600


def test_accepted_runner_has_artifact_writer_but_no_artifact_is_created():
    runner = AcceptedRunRunner()
    assert callable(runner.run)
    assert runner.config.run_id == "V3.3-P16-ACCEPTED"
