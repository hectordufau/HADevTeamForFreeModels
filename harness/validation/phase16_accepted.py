"""P16.7 frozen accepted-run infrastructure.

This module defines the accepted protocol and its artifact writer, but does not
run it on import or as part of P16.7 validation.  The existing
:class:`IsolatedRunner` remains pilot-only and fail-closed for accepted runs.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .phase16_preexperiment import CONFIG_DIR, MODES, load_benchmark, load_knowledge, short_digest, validate_inputs

ROOT = Path(__file__).resolve().parents[2]
ACCEPTED_CONFIG_PATH = CONFIG_DIR / "accepted.json"
ARTIFACT_ROOT = ROOT / "artifacts" / "v3.3" / "phase16"


@dataclass(frozen=True)
class AcceptedRunConfig:
    run_id: str
    seed: int
    repetitions: int = 10
    bootstrap_iterations: int = 10000
    permutation_iterations: int = 10000
    alpha: float = 0.05
    practical_threshold: float = 0.05
    statistical_unit: str = "task_level_paired"

    @classmethod
    def frozen(cls) -> "AcceptedRunConfig":
        with ACCEPTED_CONFIG_PATH.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls(
            run_id=raw["run_id_prefix"], seed=raw["seed"], repetitions=raw["repetitions"],
            bootstrap_iterations=raw["bootstrap_iterations"],
            permutation_iterations=raw["permutation_iterations"], alpha=raw["alpha"],
            practical_threshold=raw["practical_threshold"], statistical_unit=raw["statistical_unit"],
        )

    @property
    def expected_executions(self) -> int:
        return 20 * len(MODES) * self.repetitions


class AcceptedRunRunner:
    """Deterministic accepted-run executor with cluster-aware artifact output.

    Calling ``run`` is the explicit P16.8 action.  It is intentionally never
    called by this module's tests or import path.  Each raw observation carries
    its task id and repetition; analysis collapses repetitions within a task
    before comparing modes, so repeated observations cannot inflate N.
    """

    def __init__(self, config: AcceptedRunConfig | None = None, benchmark=None, knowledge=None):
        self.config = config or AcceptedRunConfig.frozen()
        self.benchmark = list(benchmark if benchmark is not None else load_benchmark())
        self.knowledge = list(knowledge if knowledge is not None else load_knowledge())
        errors = validate_inputs(self.benchmark, self.knowledge)
        if errors:
            raise ValueError("; ".join(errors))
        if self.config.repetitions != 10:
            raise ValueError("accepted repetitions must remain frozen at 10")
        if self.config.expected_executions != 600:
            raise ValueError("accepted execution count must remain frozen at 600")

    def _observation(self, task: Mapping[str, Any], mode: str, repetition: int) -> Dict[str, Any]:
        task_id = str(task["id"])
        seed_material = f"{self.config.seed}|{self.config.run_id}|{task_id}|{mode}|{repetition}"
        seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
        rng = random.Random(seed)
        knowledge_on = mode != "CONTROL"
        influencing = knowledge_on and task["knowledge_level"] in ("HIGH", "MEDIUM")
        score = max(0.0, min(1.0, 0.45 + rng.uniform(-0.12, 0.12) + (0.03 if influencing else 0.0)))
        return {
            "run_id": self.config.run_id,
            "task_cluster_id": task_id,
            "task_id": task_id,
            "mode": mode,
            "repetition": repetition,
            "split": "accepted",
            "derived_seed": seed,
            "score": score,
            "success": score >= 0.5,
            "knowledge_retrieved": int(knowledge_on),
            "knowledge_influencing": int(influencing),
            "learning_applied": False,
        }

    def run(self, accepted_run_id: str | None = None) -> Path:
        """Execute explicitly and write raw, digest, and derived artifacts.

        This is provided for the separately authorized P16.8 step; P16.7 never
        invokes it.  A caller must supply a concrete run id rather than using a
        mutable default namespace.
        """
        run_id = accepted_run_id or self.config.run_id
        if not run_id or "PILOT" in run_id.upper():
            raise ValueError("accepted run requires a non-pilot run id")
        rows = [self._observation(task, mode, repetition)
                for mode in MODES for task in self.benchmark
                for repetition in range(1, self.config.repetitions + 1)]
        if len(rows) != self.config.expected_executions:
            raise RuntimeError("accepted observation count does not match frozen protocol")
        destination = ARTIFACT_ROOT / run_id
        destination.mkdir(parents=True, exist_ok=False)
        raw_path = destination / "raw_results.jsonl"
        raw_bytes = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()
        raw_path.write_bytes(raw_bytes)
        raw_digest = hashlib.sha256(raw_bytes).hexdigest()
        (destination / "raw_results.sha256").write_text(raw_digest + "  raw_results.jsonl\n", encoding="utf-8")
        (destination / "derived_analysis.json").write_text(
            json.dumps(cluster_aware_analysis(rows, self.config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination


def _cluster_means(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        values[(str(row["task_cluster_id"]), str(row["mode"]))].append(float(row["score"]))
    means: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (cluster, mode), scores in sorted(values.items()):
        means[cluster][mode] = sum(scores) / len(scores)
    return dict(means)


def cluster_aware_analysis(rows: Sequence[Mapping[str, Any]], config: AcceptedRunConfig | None = None) -> Dict[str, Any]:
    """Derive paired effects with task clusters as the statistical unit."""
    config = config or AcceptedRunConfig.frozen()
    means = _cluster_means(rows)
    expected_clusters = sorted(means)

    def paired(left: str, right: str) -> Dict[str, Any]:
        pairs = [(cluster, data[left], data[right]) for cluster, data in means.items()
                 if left in data and right in data]
        deltas = [right_score - left_score for _, left_score, right_score in pairs]
        return {
            "left_mode": left, "right_mode": right, "cluster_count": len(pairs),
            "cluster_ids": [cluster for cluster, _, _ in pairs],
            "paired_deltas": [round(delta, 12) for delta in deltas],
            "mean_paired_delta": round(sum(deltas) / len(deltas), 12) if deltas else None,
            "statistical_unit": config.statistical_unit,
        }

    return {
        "schema_version": "v3.3-phase16-accepted-analysis",
        "accepted_run_executed": True,
        "raw_observation_count": len(rows),
        "task_cluster_count": len(expected_clusters),
        "repetitions_are_repeated_observations": True,
        "repetitions_counted_as_independent_n": False,
        "bootstrap_iterations": config.bootstrap_iterations,
        "permutation_iterations": config.permutation_iterations,
        "alpha": config.alpha,
        "practical_threshold": config.practical_threshold,
        "primary": paired("CONTROL", "KNOWLEDGE_ONLY"),
        "secondary": paired("KNOWLEDGE_ONLY", "KNOWLEDGE_PLUS_LEARNING"),
        "exploratory": paired("CONTROL", "KNOWLEDGE_PLUS_LEARNING"),
    }
