"""Phase 16 P16.0-P16.6 pre-experiment readiness infrastructure.

This module is deliberately an experiment harness, not a replacement for the
V3.3 knowledge or learning architecture.  It owns only immutable inputs,
measurement, mode isolation, predeclared statistics, and pilot evidence.
Accepted field validation is intentionally not implemented here.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
PHASE16_DIR = ROOT / "artifacts" / "v3.3" / "phase16"
CONFIG_DIR = ROOT / "config" / "phase16"
MODES = ("CONTROL", "KNOWLEDGE_ONLY", "KNOWLEDGE_PLUS_LEARNING")
KNOWLEDGE_LEVELS = ("HIGH", "MEDIUM", "LOW", "NONE")
CATEGORIES = ("functional", "architecture", "security", "requirements", "NFR", "TDR", "risk", "RCA", "maintenance", "irrelevant")


def digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def short_digest(value: Any) -> str:
    return digest(value)[:16]


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_benchmark() -> List[Dict[str, Any]]:
    """Return a deep copy; callers cannot mutate the frozen source."""
    return copy.deepcopy(load_json(CONFIG_DIR / "benchmark.json"))


def load_knowledge() -> List[Dict[str, Any]]:
    return copy.deepcopy(load_json(CONFIG_DIR / "knowledge.json"))


def validate_inputs(benchmark: Sequence[Mapping[str, Any]], knowledge: Sequence[Mapping[str, Any]]) -> List[str]:
    errors: List[str] = []
    ids = [str(x.get("id")) for x in benchmark]
    if len(ids) != len(set(ids)):
        errors.append("duplicate benchmark ids")
    for task in benchmark:
        if task.get("category") not in CATEGORIES:
            errors.append(f"invalid category: {task.get('id')}")
        if task.get("knowledge_level") not in KNOWLEDGE_LEVELS:
            errors.append(f"invalid knowledge level: {task.get('id')}")
        if "answer" in task or "expected_answer" in task:
            errors.append(f"answer leakage: {task.get('id')}")
    if not benchmark:
        errors.append("empty benchmark")
    if not knowledge:
        errors.append("empty knowledge dataset")
    for item in knowledge:
        if item.get("category") not in CATEGORIES:
            errors.append(f"invalid knowledge category: {item.get('id')}")
        if any(k in item for k in ("answer", "solution", "expected_answer")):
            errors.append(f"knowledge answer leakage: {item.get('id')}")
    return errors


@dataclass(frozen=True)
class Phase16Config:
    run_id: str = "V3.3-P16-PILOT"
    seed: int = 20260918
    pilot_repetitions: int = 1
    bootstrap_iterations: int = 1000
    permutation_iterations: int = 1000
    alpha: float = 0.05
    practical_effect: float = 0.05
    accepted_run: bool = False

    def canonical(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def config_digest(self) -> str:
        return short_digest(self.canonical())


@dataclass(frozen=True)
class Observation:
    run_id: str
    task_id: str
    mode: str
    repetition: int
    split: str
    score: float
    success: bool
    knowledge_retrieved: int
    knowledge_influencing: int
    learning_applied: bool
    derived_seed: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Metrics:
    """Phase-16 instrumentation using canonical SHA-256 identities."""
    @staticmethod
    def execution_seed(config: Phase16Config, task_id: str, mode: str, repetition: int) -> int:
        raw = f"{config.seed}|{config.run_id}|{task_id}|{mode}|{repetition}".encode()
        return int(hashlib.sha256(raw).hexdigest()[:16], 16)

    @staticmethod
    def observation_digest(observations: Iterable[Observation]) -> str:
        return short_digest([o.to_dict() for o in observations])

    @staticmethod
    def aggregate(observations: Sequence[Observation]) -> Dict[str, Any]:
        n = len(observations)
        return {"n": n, "success_rate": (sum(o.success for o in observations) / n if n else None),
                "mean_score": (sum(o.score for o in observations) / n if n else None),
                "knowledge_retrieved": sum(o.knowledge_retrieved for o in observations),
                "knowledge_influencing": sum(o.knowledge_influencing for o in observations),
                "learning_applied": sum(o.learning_applied for o in observations)}


class IsolatedRunner:
    """Small deterministic runner for P16.6 infrastructure validation only."""
    def __init__(self, config: Phase16Config, benchmark=None, knowledge=None):
        self.config = config
        self.benchmark = copy.deepcopy(benchmark if benchmark is not None else load_benchmark())
        self.knowledge = copy.deepcopy(knowledge if knowledge is not None else load_knowledge())
        errors = validate_inputs(self.benchmark, self.knowledge)
        if errors:
            raise ValueError("; ".join(errors))
        self.observations: List[Observation] = []
        self.mode_state = {m: {"learned": 0, "knowledge": 0} for m in MODES}

    def run(self) -> List[Observation]:
        if self.config.accepted_run:
            raise RuntimeError("P16.7-P16.10 accepted stages are not enabled")
        for mode in MODES:
            for task in self.benchmark:
                for rep in range(1, self.config.pilot_repetitions + 1):
                    seed = Metrics.execution_seed(self.config, task["id"], mode, rep)
                    rng = random.Random(seed)
                    level = task["knowledge_level"]
                    knowledge_on = mode != "CONTROL"
                    learning_on = mode == "KNOWLEDGE_PLUS_LEARNING"
                    retrieved = (1 if knowledge_on and level != "NONE" else 0)
                    influencing = (1 if retrieved and level in ("HIGH", "MEDIUM") else 0)
                    # Synthetic runner signal is infrastructure-only, not an effectiveness claim.
                    score = max(0.0, min(1.0, 0.45 + rng.uniform(-0.12, 0.12) + (0.03 if influencing else 0.0)))
                    obs = Observation(self.config.run_id, task["id"], mode, rep, "pilot", score,
                                      score >= 0.5, retrieved, influencing, learning_on and rep > 1, seed)
                    self.observations.append(obs)
                    self.mode_state[mode]["knowledge"] += retrieved
                    self.mode_state[mode]["learned"] += int(obs.learning_applied)
        return list(self.observations)

    def isolation_check(self) -> Dict[str, Any]:
        by_mode = {m: [o for o in self.observations if o.mode == m] for m in MODES}
        return {"control_knowledge": sum(o.knowledge_retrieved for o in by_mode["CONTROL"]),
                "control_learning": sum(o.learning_applied for o in by_mode["CONTROL"]),
                "knowledge_only_learning": sum(o.learning_applied for o in by_mode["KNOWLEDGE_ONLY"]),
                "test_training": sum(o.learning_applied for o in self.observations if o.split == "test"),
                "distinct_mode_state": len({id(self.mode_state[m]) for m in MODES}) == 3}

    def artifact(self) -> Dict[str, Any]:
        return {"schema_version": "v3.3-phase16-pilot", "status": "PILOT_ONLY",
                "accepted_run_executed": False, "config": self.config.canonical(),
                "config_digest": self.config.config_digest,
                "benchmark_digest": short_digest(self.benchmark), "knowledge_digest": short_digest(self.knowledge),
                "observation_digest": Metrics.observation_digest(self.observations),
                "observations": [o.to_dict() for o in self.observations],
                "isolation": self.isolation_check()}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def analyze_effect(baseline: Sequence[float], comparison: Sequence[float], *, practical_effect: float = 0.05) -> Dict[str, Any]:
    """Predeclared descriptive effect classifier; no real-result tuning."""
    if not baseline or not comparison:
        return {"n_baseline": len(baseline), "n_comparison": len(comparison), "delta": None, "classification": "INSUFFICIENT_DATA"}
    delta = _mean(comparison) - _mean(baseline)
    if delta > practical_effect:
        classification = "POSITIVE"
    elif delta < -practical_effect:
        classification = "NEGATIVE"
    else:
        classification = "NEUTRAL"
    return {"n_baseline": len(baseline), "n_comparison": len(comparison), "baseline_mean": round(_mean(baseline), 8),
            "comparison_mean": round(_mean(comparison), 8), "delta": round(delta, 8), "classification": classification,
            "method": "difference_in_means_with_predeclared_practical_threshold"}


def synthetic_golden_cases() -> Dict[str, Dict[str, Any]]:
    base = [0.50, 0.52, 0.48, 0.51]
    return {"positive": analyze_effect(base, [0.70, 0.72, 0.68, 0.71]),
            "neutral": analyze_effect(base, [0.51, 0.50, 0.49, 0.52]),
            "negative": analyze_effect(base, [0.30, 0.32, 0.28, 0.31])}


def write_pilot(config: Phase16Config | None = None) -> Path:
    config = config or Phase16Config()
    runner = IsolatedRunner(config)
    runner.run()
    PHASE16_DIR.mkdir(parents=True, exist_ok=True)
    path = PHASE16_DIR / "pilot.json"
    path.write_text(json.dumps(runner.artifact(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    path = write_pilot()
    print(json.dumps({"pilot_status": "PILOT_ONLY", "path": str(path), "accepted_run_executed": False}))
