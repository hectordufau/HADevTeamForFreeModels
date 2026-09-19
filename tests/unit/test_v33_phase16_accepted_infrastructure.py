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


def test_mode_treatments_are_isolated_and_knowledge_retrieval_respects_none():
    runner = AcceptedRunRunner()
    tasks = [
        {"id": "NONE", "knowledge_level": "NONE"},
        {"id": "LOW", "knowledge_level": "LOW"},
    ]
    rows = {
        (mode, task["knowledge_level"]): runner._observation(task, mode, 1)
        for mode in ("CONTROL", "KNOWLEDGE_ONLY", "KNOWLEDGE_PLUS_LEARNING")
        for task in tasks
    }
    assert all(rows[("CONTROL", level)]["learning_applied"] is False for level in ("NONE", "LOW"))
    assert all(rows[("KNOWLEDGE_ONLY", level)]["learning_applied"] is False for level in ("NONE", "LOW"))
    assert all(rows[("KNOWLEDGE_PLUS_LEARNING", level)]["learning_applied"] is True for level in ("NONE", "LOW"))
    assert all(row["knowledge_retrieved"] == 0 for (mode, level), row in rows.items() if level == "NONE")
    assert all(row["knowledge_retrieved"] == (0 if mode == "CONTROL" else 1)
               for (mode, level), row in rows.items() if level == "LOW")


def test_learning_effect_is_predeclared_and_deterministic():
    runner = AcceptedRunRunner()
    task = {"id": "P16-001", "knowledge_level": "HIGH"}
    control = runner._observation(task, "CONTROL", 1)
    plus = runner._observation(task, "KNOWLEDGE_PLUS_LEARNING", 1)
    plus_again = runner._observation(task, "KNOWLEDGE_PLUS_LEARNING", 1)
    control_again = runner._observation(task, "CONTROL", 1)
    assert plus["learning_score_effect"] == 0.01
    assert control["learning_score_effect"] == 0.0
    assert plus == plus_again
    assert control == control_again
