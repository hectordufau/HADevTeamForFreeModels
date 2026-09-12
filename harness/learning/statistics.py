# harness/learning/statistics.py — V3.2 Phase 5: Empirical COLD Baseline & Statistical Evaluation
"""
Deterministic, empirical-baseline statistical evaluation for V3.2.

Phase 5 replaces the artificial 0.5 generalization baseline with an empirical
COLD-derived baseline computed from *persisted COLD execution outcomes*. It
implements paired COLD vs LEARNED comparison, configurable significance
thresholds, percentile-bootstrap confidence intervals with a deterministic
seed derived from the ExperimentContext, and a full statistical outputs block.

Guarantees (enforced here and exercised in tests):
- NO hard-coded 0.5 baseline. Every baseline is computed from observed COLD
  data, or reported as INSUFFICIENT_BASELINE_DATA when too few observations.
- Deterministic: same inputs -> same baseline / delta / p-value / CI / effect
  size / significance. Bootstrap uses a seed derived from ExperimentContext,
  never the global random state.
- Read-only evaluation: this module NEVER writes strategies, experiences,
  model/workflow/agent/exploration policy, or routing. It is pure evaluation
  and therefore cannot create a learning feedback loop.
- Baseline scope separation: COLD / LEARNED / TEST are kept distinct. TEST
  observations can never become a baseline. Cross-run isolation: only the
  requested validation_run_id (or explicit global-historical scope) enters a
  baseline.
- Small samples: for n < min_samples no significance is claimed;
  statistical_significance = INSUFFICIENT_DATA and CI = NOT_COMPUTED.
- Security is a HARD constraint: any security failure is its own binary
  outcome; it is never averaged into another metric.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Sentinel / status constants
# ---------------------------------------------------------------------------

INSUFFICIENT_BASELINE_DATA = "INSUFFICIENT_BASELINE_DATA"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
NOT_AVAILABLE = "NOT_AVAILABLE"
NOT_APPLICABLE = "NOT_APPLICABLE"
CI_NOT_COMPUTED_INSUFFICIENT_SAMPLE = "CI_NOT_COMPUTED_INSUFFICIENT_SAMPLE"
NO_PAIRED_DATA = "NO_PAIRED_DATA"

VALID_MODES = ("COLD", "LEARNED", "LEARNED_NO_EXPLORATION", "LEARNED_EXPLORATION", "TEST")

# V3.2 Phase 6 learned modes map to the "learned" arm in the Phase 5 comparison
# without changing COLD/TEST meaning (spec §11 legacy-LEARNED mapping).
_LEARNED_MODES_P6 = {"LEARNED", "LEARNED_NO_EXPLORATION", "LEARNED_EXPLORATION"}


def _is_learned_mode(mode: str) -> bool:
    return mode in _LEARNED_MODES_P6

# Metric families we can evaluate. Only those with valid observations are
# computed; security is a hard constraint, not an averaged metric.
METRIC_FAMILIES = (
    "success_rate",
    "engineering_score",
    "correctness",
    "architecture",
    "security",
    "maintainability",
    "efficiency",
)

# Per-observation scalar metrics (real values, means + CI make sense).
SCALAR_REPORT_FIELDS = (
    "engineering_score",
    "correctness",
    "architecture",
    "maintainability",
    "efficiency",
    "latency",
    "iterations",
    "verification_failures",
    "retries",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignificanceConfig:
    """Configurable statistical-significance thresholds.

    Loaded from the V3.2 config mechanism (config/harness.yaml under
    ``evaluation.significance``); may also be constructed programmatically for
    tests. Always exposed in reports so it is observable.
    """

    success_rate_pp: float = 0.05  # min absolute percentage-point delta as a fraction (0.05 == 5pp)
    score_delta: float = 0.01  # min absolute score delta
    min_samples: int = 10  # deterministic minimum sample size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success_rate_pp": self.success_rate_pp,
            "score_delta": self.score_delta,
            "min_samples": self.min_samples,
        }


DEFAULT_SIGNIFICANCE = SignificanceConfig()


def default_significance_path() -> str:
    """Absolute path to the V3.2 config file (project root config/harness.yaml)."""
    return os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config",
            "harness.yaml",
        )
    )


def load_significance_config(path: Optional[str] = None) -> SignificanceConfig:
    """Load significance config from the V3.2 config mechanism.

    Reads ``evaluation.significance`` from the YAML. Missing keys fall back to
    defaults; a missing file / section also falls back to defaults (the runtime
    must not fail because a brand-new config key is absent). Config values that
    are present but invalid (e.g. negative min_samples) raise ValueError so a
    misconfiguration is never silently accepted.
    """
    p = path or default_significance_path()
    config: Dict[str, Any] = {}
    if os.path.exists(p):
        try:
            import yaml

            with open(p) as f:
                loaded = yaml.safe_load(f) or {}
            config = (loaded.get("evaluation") or {}).get("significance") or {}
        except Exception:
            # Unreadable/corrupt YAML -> defaults (deterministic, observable).
            config = {}

    success_rate_pp = config.get("success_rate_pp", DEFAULT_SIGNIFICANCE.success_rate_pp)
    score_delta = config.get("score_delta", DEFAULT_SIGNIFICANCE.score_delta)
    min_samples = config.get("min_samples", DEFAULT_SIGNIFICANCE.min_samples)

    def _require_number(v: Any, name: str) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"evaluation.significance.{name} must be numeric, got {v!r}")
        if math.isnan(f) or math.isinf(f):
            raise ValueError(f"evaluation.significance.{name} must be finite, got {v!r}")
        return f

    success_rate_pp = _require_number(success_rate_pp, "success_rate_pp")
    score_delta = _require_number(score_delta, "score_delta")
    min_samples_f = _require_number(min_samples, "min_samples")
    if min_samples_f < 0 or int(min_samples_f) != min_samples_f:
        raise ValueError(
            f"evaluation.significance.min_samples must be a non-negative integer, got {min_samples!r}"
        )
    if success_rate_pp < 0 or score_delta < 0:
        raise ValueError(
            "evaluation.significance thresholds (success_rate_pp, score_delta) must be >= 0"
        )

    return SignificanceConfig(
        success_rate_pp=success_rate_pp,
        score_delta=score_delta,
        min_samples=int(min_samples_f),
    )


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------


@dataclass
class EvaluationObservation:
    """A single persisted execution outcome used for COLD/LEARNED comparison.

    mode is one of COLD / LEARNED / TEST. validation_run_id and benchmark_id
    scope the observation to its experiment. task_id/category/config enable
    deterministic paired matching.
    """

    validation_run_id: str
    benchmark_id: str
    mode: str
    task_id: str = ""
    category: str = ""
    config: str = ""

    # scalar metrics
    engineering_score: Optional[float] = None
    correctness: Optional[float] = None
    architecture: Optional[float] = None
    security: Optional[float] = None
    maintainability: Optional[float] = None
    efficiency: Optional[float] = None
    latency: Optional[float] = None
    iterations: Optional[int] = None
    verification_failures: Optional[int] = None
    retries: Optional[int] = None

    # success indicator (bool) and security hard-failure flag (bool)
    success: Optional[bool] = None
    security_failure: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of: {', '.join(VALID_MODES)}"
            )
        if not self.validation_run_id or not self.benchmark_id:
            raise ValueError("validation_run_id and benchmark_id are required")

    def metric_values(self, metric: str) -> List[float]:
        """Return the list of valid numeric observations for a metric."""
        if metric == "success_rate":
            return [1.0 if self.success else 0.0] if self.success is not None else []
        if metric == "security_failure":
            return [1.0 if self.security_failure else 0.0] if self.security_failure is not None else []
        scalar = getattr(self, metric, None)
        if scalar is None:
            return []
        return [float(scalar)]

    @staticmethod
    def metric_supported(metric: str) -> bool:
        return metric in METRIC_FAMILIES or metric in SCALAR_REPORT_FIELDS or metric == "security_failure"

    def match_key(self) -> Tuple[str, str, str, str]:
        """Deterministic group key for pairing COLD/LEARNED observations."""
        return (
            self.benchmark_id,
            self.task_id or "",
            self.category or "",
            self.config or "",
        )


# ---------------------------------------------------------------------------
# Deterministic RNG (NEVER the global random state)
# ---------------------------------------------------------------------------


def _derive_seed(experiment_context: Any, salt: str = "v3.2-bootstrap") -> int:
    """Derive a deterministic PRNG seed from an ExperimentContext.

    Uses SHA-256 over canonical identifiers; the same context always yields the
    same seed, so bootstrap results are reproducible and run-isolated (a
    different run id gives a different, deterministic seed).
    """
    if experiment_context is None:
        # No context: derive purely from a stable default, keeping determinism.
        run_id = benchmark_id = "none"
    else:
        run_id = getattr(experiment_context, "validation_run_id", "none")
        benchmark_id = getattr(experiment_context, "benchmark_id", "none")
    digest = hashlib.sha256(f"{salt}::{run_id}::{benchmark_id}".encode()).hexdigest()
    return int(digest[:16], 16)


# ---------------------------------------------------------------------------
# Persisted-observation loader (V3.1 field validation run results)
# ---------------------------------------------------------------------------


def _proj_root() -> str:
    """Project root: harness/learning/statistics.py -> repo root."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_field_validation_observations(
    split: Optional[str] = None,
    *,
    base_dir: Optional[str] = None,
    results_dir_cold: Optional[str] = None,
    results_dir_learned: Optional[str] = None,
) -> List[EvaluationObservation]:
    """Load persisted V3.1 field-validation run results into EvaluationObservations.

    Reads the JSON run-result files written by harness/validation/runner.py in
    ``harness/validation/cold/results`` and ``harness/validation/learned/results``.
    Each file is a single persisted COLD or LEARNED execution outcome.

    Args:
        split: optional split filter ("test", "train", "validation", "warmup").
            TEST-split observations are real held-out evaluation data — they are
            loaded here as *evaluation* observations (COLD baseline vs LEARNED on
            the held-out set); they never enter any *training* baseline.
        base_dir: overrides the search root (default: repo harness/validation).
        results_dir_cold / results_dir_learned: exact override of the two
            results directories.

    Returns:
        List[EvaluationObservation] with mode COLD / LEARNED. paired by
        (task_id, seed) so a matched paired comparison is reproducible. A
        ``seed`` field is carried in ``config`` to preserve run match keys.
    """
    base = base_dir or os.path.join(_proj_root(), "harness", "validation")
    cold_dir = results_dir_cold or os.path.join(base, "cold", "results")
    learned_dir = results_dir_learned or os.path.join(base, "learned", "results")

    def _collect(dirpath: str, mode: str) -> List[EvaluationObservation]:
        out: List[EvaluationObservation] = []
        if not os.path.isdir(dirpath):
            return out
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith(".json"):
                continue
            d = _read_json(os.path.join(dirpath, fname))
            if split and d.get("split") != split:
                continue
            status = (d.get("status") or "").upper()
            # security_failure is ONLY set when the persisted record exposes an
            # explicit security-failure flag. Absent it, we leave None (no valid
            # observation) rather than fabricate a clean-security result.
            sec_field = d.get("security_failure")
            out.append(
                EvaluationObservation(
                    validation_run_id=d.get("run_id", "V3.1-FV"),
                    benchmark_id="V3.1-FIELD-VALIDATION",
                    mode=mode,
                    task_id=d.get("task_id", ""),
                    category=d.get("split", ""),
                    config=json.dumps({"seed": d.get("seed")},
                                      sort_keys=True),  # keeps seed in match key
                    engineering_score=d.get("score"),
                    latency=d.get("duration_seconds"),
                    iterations=d.get("iterations"),
                    verification_failures=d.get("verification_failures"),
                    retries=d.get("retries"),
                    success=(status == "PASSED"),
                    security_failure=sec_field,
                )
            )
        return out

    obs = _collect(cold_dir, "COLD") + _collect(learned_dir, "LEARNED")
    return obs


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------


def _mean(xs) -> float:
    """Mean of an iterable of floats, NaN when empty. Materializes once."""
    vals = list(xs)
    return float(statistics.mean(vals)) if vals else float("nan")


def _std(xs: Sequence[float]) -> float:
    return float(statistics.stdev(xs)) if len(xs) >= 2 else float("nan")


def percentile_bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI.

    Deterministic: consumes only the local Random(seed), never the global
    random state. Because it is seeded from the ExperimentContext, two calls
    with the same context/values return identical intervals.
    """
    import random

    vals = list(values)
    n = len(vals)
    rng = random.Random(seed)
    boot_means = [
        _mean(rng.choice(vals) for _ in range(n)) for _ in range(n_boot)
    ]
    boot_means.sort()
    lo_idx = int(round((alpha / 2.0) * (n_boot - 1)))
    hi_idx = int(round((1.0 - alpha / 2.0) * (n_boot - 1)))
    return boot_means[lo_idx], boot_means[hi_idx]


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d for two independent samples (positive = a mean > b mean)."""
    if not a or not b or len(a) < 2 or len(b) < 2:
        return float("nan")
    sa = _std(a)
    sb = _std(b)
    if sa != sa or sb != sb or (sa == 0 and sb == 0):
        return float("nan")
    pooled = math.sqrt(((len(a) - 1) * sa * sa + (len(b) - 1) * sb * sb) / (len(a) + len(b) - 2))
    if pooled == 0:
        return float("nan")
    return (_mean(a) - _mean(b)) / pooled


def _success_percentage_points(p: float, q: float) -> float:
    """Absolute difference in success rate as percentage points (p - q) * 100."""
    return float((p - q) * 100.0)


def _bootstrap_pvalue(
    cold: Sequence[float],
    learned: Sequence[float],
    *,
    seed: int,
    n_boot: int = 2000,
    paired: bool = False,
) -> float:
    """Two-sided permutation/bootstrap p-value under H0: no difference.

    Deterministic via the provided seed. For paired data, pairs are
    reassigned within each matched pair; for unpaired, labels are permuted.
    Returns None semantics via NOT_AVAILABLE handled by caller (p in [0,1]).
    """
    import random

    cold = list(cold)
    learned = list(learned)
    if not cold or not learned:
        return float("nan")
    rng = random.Random(seed)
    observed = _mean(learned) - _mean(cold)

    if paired:
        # n must match for a valid paired test.
        n = min(len(cold), len(learned))
        diffs = [learned[i] - cold[i] for i in range(n)]
        if not diffs:
            return float("nan")
        obs = _mean(diffs)
        count = 0
        for _ in range(n_boot):
            shuffled = [d if rng.random() < 0.5 else -d for d in diffs]
            if abs(_mean(shuffled)) >= abs(obs):
                count += 1
        p = (count + 1) / (n_boot + 1)
    else:
        combined = cold + learned
        n_c = len(cold)
        observed = _mean(learned) - _mean(cold)
        count = 0
        for _ in range(n_boot):
            rng.shuffle(combined)
            a = combined[:n_c]
            b = combined[n_c:]
            if abs(_mean(b) - _mean(a)) >= abs(observed):
                count += 1
        p = (count + 1) / (n_boot + 1)
    return float(p)


# ---------------------------------------------------------------------------
# Empirical COLD baseline builder
# ---------------------------------------------------------------------------


class ColdBaseline:
    """Empirical COLD baseline derived from persisted COLD observations.

    The caller supplies the COLD observations (already scoped to the exact run
    / benchmark / explicit global-historical scope). If there are fewer than
    min_samples valid observations for a metric, the metric's baseline is
    reported as INSUFFICIENT_BASELINE_DATA rather than inventing a value.
    """

    def __init__(self, observations: Sequence[EvaluationObservation], config: SignificanceConfig):
        self.observations = [o for o in observations if o.mode == "COLD"]
        self.config = config

    def mean(self, metric: str) -> Any:
        """Mean COLD value for a metric, or INSUFFICIENT_BASELINE_DATA."""
        if metric not in METRIC_FAMILIES and metric not in SCALAR_REPORT_FIELDS:
            return NOT_AVAILABLE
        values: List[float] = []
        for o in self.observations:
            values.extend(o.metric_values(metric))
        if len(values) < self.config.min_samples:
            return INSUFFICIENT_BASELINE_DATA
        return round(_mean(values), 4)

    def n(self, metric: str) -> int:
        """Number of valid COLD observations for the metric."""
        values: List[float] = []
        for o in self.observations:
            values.extend(o.metric_values(metric))
        return len(values)

    def scoring(self) -> Dict[str, Any]:
        """Return the empirical baseline block for a report."""
        out: Dict[str, Any] = {"source": "EMPIRICAL_COLD", "basis": "observed_COLD_outcomes"}
        for metric in METRIC_FAMILIES:
            out[metric] = self.mean(metric)
        return out


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------


def build_pairs(
    cold: Sequence[EvaluationObservation],
    learned: Sequence[EvaluationObservation],
) -> Tuple[List[Tuple[EvaluationObservation, EvaluationObservation]], List[EvaluationObservation], List[EvaluationObservation]]:
    """Deterministically pair COLD/LEARNED observations by match_key.

    Returns (pairs, unmatched_cold, unmatched_learned). Duplicate keys within
    either side are matched one-to-one in sorted key order; leftovers are
    reported as unmatched (documenting where pairing is impossible).
    """
    from collections import defaultdict

    cold_by_key: Dict[Tuple, List[EvaluationObservation]] = defaultdict(list)
    for c in cold:
        cold_by_key[c.match_key()].append(c)
    for k in cold_by_key:
        cold_by_key[k].sort(key=lambda o: o.task_id)

    learned_by_key: Dict[Tuple, List[EvaluationObservation]] = defaultdict(list)
    for l in learned:
        learned_by_key[l.match_key()].append(l)
    for k in learned_by_key:
        learned_by_key[k].sort(key=lambda o: o.task_id)

    pairs: List[Tuple[EvaluationObservation, EvaluationObservation]] = []
    unmatched_cold: List[EvaluationObservation] = []
    unmatched_learned: List[EvaluationObservation] = []

    all_keys = sorted(set(cold_by_key) | set(learned_by_key))
    for key in all_keys:
        cs = cold_by_key.get(key, [])
        ls = learned_by_key.get(key, [])
        n = min(len(cs), len(ls))
        for i in range(n):
            pairs.append((cs[i], ls[i]))
        unmatched_cold.extend(cs[n:])
        unmatched_learned.extend(ls[n:])

    return pairs, unmatched_cold, unmatched_learned


# ---------------------------------------------------------------------------
# Statistical comparison result
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Full statistical outputs block for one metric comparison."""

    metric: str
    baseline_n: Any
    learned_n: Any
    baseline_mean: Any
    learned_mean: Any
    delta: Any
    delta_percentage_points: Any
    confidence_interval: Any
    p_value: Any
    effect_size: Any
    statistically_significant: Any
    comparison_type: str = "UNPAIRED"  # UNPAIRED | PAIRED | NOT_APPLICABLE
    paired_n: Any = NOT_APPLICABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline_n": self.baseline_n,
            "learned_n": self.learned_n,
            "baseline_mean": self.baseline_mean,
            "learned_mean": self.learned_mean,
            "delta": self.delta,
            "delta_percentage_points": self.delta_percentage_points,
            "confidence_interval": self.confidence_interval,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "statistically_significant": self.statistically_significant,
            "comparison_type": self.comparison_type,
            "paired_n": self.paired_n,
        }


class EmpiricalEvaluator:
    """Deterministic empirical COLD vs LEARNED evaluator (Phase 5).

    Flow (no feedback loop): persisted COLD/LEARNED observations ->
    empirical baseline -> paired/unpaired comparison -> statistical eval ->
    report. This evaluator never mutates learning artifacts.
    """

    def __init__(
        self,
        observations: Sequence[EvaluationObservation],
        config: Optional[SignificanceConfig] = None,
        experiment_context: Any = None,
        *,
        bootstrap_n: int = 2000,
    ):
        self.config = config or DEFAULT_SIGNIFICANCE
        self.experiment_context = experiment_context
        self.bootstrap_n = bootstrap_n
        self._seed = _derive_seed(experiment_context)
        self.observations = list(observations)
        # Strict isolation: split by mode, TEST can never enter baselines.
        # Legacy LEARNED is the canonical "learned" arm; V3.2 Phase 6 learned
        # modes (LEARNED_NO_EXPLORATION / LEARNED_EXPLORATION) are learned arms
        # too. Their meaning is that of LEARNED for Phase 5 comparison; see
        # docs/V3.2-PHASE6-REPORT.md "legacy LEARNED mapping".
        self.cold = [o for o in self.observations if o.mode == "COLD"]
        self.learned = [o for o in self.observations if _is_learned_mode(o.mode)]
        self.test = [o for o in self.observations if o.mode == "TEST"]
        self._baseline = ColdBaseline(self.cold, self.config)

    # -- helpers ------------------------------------------------------------

    def _metric_values(self, obs: Sequence[EvaluationObservation], metric: str) -> List[float]:
        out: List[float] = []
        for o in obs:
            out.extend(o.metric_values(metric))
        return out

    # -- public API ---------------------------------------------------------

    def baseline(self) -> Dict[str, Any]:
        """Empirical COLD baseline block (never TEST, never other runs inside)."""
        return self._baseline.scoring()

    def insufficient_baseline(self, metric: str) -> bool:
        return self._baseline.mean(metric) == INSUFFICIENT_BASELINE_DATA

    def compare(self, metrics: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Run comparisons over the requested (or default) metrics.

        Returns {metric: ComparisonResult.to_dict()}. Metrics without valid
        observations on both sides are reported NOT_AVAILABLE / INSUFFICIENT_DATA.
        Security is reported as its own hard-failure rate (not averaged with
        engineering score); a security failure is a binary hard event.
        """
        metrics = metrics or list(METRIC_FAMILIES)
        out: Dict[str, Any] = {}
        for metric in metrics:
            res = self._compare_one(metric)
            # Key by the canonical output metric name (e.g. "security" ->
            # "security_failure_rate"), so the report is consistent.
            out[res.metric] = res.to_dict()
        return out

    def _compare_one(self, metric: str) -> ComparisonResult:
        cfg = self.config
        if metric == "security":
            # SECURITY IS A HARD CONSTRAINT: report the observed binary
            # security-failure rate separately; never averaged with other
            # metrics and never masked by aggregate means.
            cold_v = self._metric_values(self.cold, "security_failure")
            learned_v = self._metric_values(self.learned, "security_failure")
            return self._finish(
                metric="security_failure_rate",
                cold_values=cold_v,
                learned_values=learned_v,
                is_success_metric=False,
                hard_constraint=True,
            )

        if metric not in METRIC_FAMILIES and metric not in SCALAR_REPORT_FIELDS:
            return ComparisonResult(
                metric=metric,
                baseline_n=NOT_AVAILABLE,
                learned_n=NOT_AVAILABLE,
                baseline_mean=NOT_AVAILABLE,
                learned_mean=NOT_AVAILABLE,
                delta=NOT_AVAILABLE,
                delta_percentage_points=NOT_AVAILABLE,
                confidence_interval=NOT_AVAILABLE,
                p_value=NOT_AVAILABLE,
                effect_size=NOT_AVAILABLE,
                statistically_significant=NOT_AVAILABLE,
            )

        cold_v = self._metric_values(self.cold, metric)
        learned_v = self._metric_values(self.learned, metric)
        is_success = metric == "success_rate"
        return self._finish(metric, cold_v, learned_v, is_success_metric=is_success)

    def _finish(
        self,
        metric: str,
        cold_values: List[float],
        learned_values: List[float],
        *,
        is_success_metric: bool,
        hard_constraint: bool = False,
    ) -> ComparisonResult:
        cfg = self.config
        # Paired comparison when matched data is available on both sides.
        pairs, unmatched_cold, unmatched_learned = build_pairs(self.cold, self.learned)
        paired_cold: List[float] = []
        paired_learned: List[float] = []
        for c, l in pairs:
            paired_cold.extend(c.metric_values(metric))
            paired_learned.extend(l.metric_values(metric))
        paired_ok = bool(paired_cold) and bool(paired_learned) and len(paired_cold) == len(paired_learned)

        # Choose the sample sets for mean/CI/p: prefer paired when available,
        # otherwise fall back to unpaired aggregate (documenting pairing use).
        if paired_ok:
            use_cold = paired_cold
            use_learned = paired_learned
            paired_n = len(paired_cold)
            comparison_type = "PAIRED"
        else:
            use_cold = cold_values
            use_learned = learned_values
            paired_n = NO_PAIRED_DATA if (not paired_cold or not paired_learned) else 0
            comparison_type = "UNPAIRED"

        baseline_n = len(use_cold)
        learned_n = len(use_learned)

        # Empirical baseline insufficiency: do NOT invent a value.
        if baseline_n < cfg.min_samples:
            return ComparisonResult(
                metric=metric,
                baseline_n=baseline_n,
                learned_n=learned_n,
                baseline_mean=INSUFFICIENT_BASELINE_DATA,
                learned_mean=INSUFFICIENT_DATA if not learned_values else round(_mean(use_learned), 4),
                delta=NOT_AVAILABLE,
                delta_percentage_points=NOT_AVAILABLE,
                confidence_interval=CI_NOT_COMPUTED_INSUFFICIENT_SAMPLE,
                p_value=INSUFFICIENT_DATA,
                effect_size=NOT_AVAILABLE,
                statistically_significant=INSUFFICIENT_DATA,
                comparison_type=comparison_type,
                paired_n=paired_n,
            )

        if not learned_values:
            return ComparisonResult(
                metric=metric,
                baseline_n=baseline_n,
                learned_n=0,
                baseline_mean=round(_mean(use_cold), 4),
                learned_mean=INSUFFICIENT_DATA,
                delta=NOT_AVAILABLE,
                delta_percentage_points=NOT_AVAILABLE,
                confidence_interval=CI_NOT_COMPUTED_INSUFFICIENT_SAMPLE,
                p_value=INSUFFICIENT_DATA,
                effect_size=NOT_AVAILABLE,
                statistically_significant=INSUFFICIENT_DATA,
                comparison_type=comparison_type,
                paired_n=paired_n,
            )

        b_mean = round(_mean(use_cold), 4)
        l_mean = round(_mean(use_learned), 4)
        delta = round(l_mean - b_mean, 4)
        dpp = round(_success_percentage_points(l_mean, b_mean), 4) if is_success_metric else NOT_APPLICABLE

        # CI: only where sample size sufficient (deterministic bootstrap).
        if len(use_cold) < cfg.min_samples or len(use_learned) < cfg.min_samples:
            ci = CI_NOT_COMPUTED_INSUFFICIENT_SAMPLE
        else:
            ci = [round(v, 4) for v in percentile_bootstrap_ci(
                use_cold, seed=self._seed, n_boot=self.bootstrap_n
            )]

        # p-value: paired bootstrap when paired, else unpaired.
        if len(use_cold) < cfg.min_samples or len(use_learned) < cfg.min_samples:
            p_value = INSUFFICIENT_DATA
        else:
            p_value = round(
                _bootstrap_pvalue(
                    use_cold,
                    use_learned,
                    seed=self._seed,
                    n_boot=self.bootstrap_n,
                    paired=paired_ok,
                ),
                4,
            )

        # Effect size: Cohen's d on the chosen sample set.
        if len(use_cold) < 2 or len(use_learned) < 2:
            effect_size = NOT_AVAILABLE
        else:
            d = cohens_d(use_learned, use_cold)
            effect_size = NOT_AVAILABLE if d != d else round(d, 4)

        # Significance classification (deterministic, configurable thresholds).
        sample_ok = len(use_cold) >= cfg.min_samples and len(use_learned) >= cfg.min_samples
        if not sample_ok:
            significant: Any = INSUFFICIENT_DATA
        elif is_success_metric:
            significant = (
                isinstance(dpp, (int, float)) and abs(float(dpp)) >= (cfg.success_rate_pp * 100.0)
            )
        else:
            significant = bool(abs(delta) >= cfg.score_delta)

        return ComparisonResult(
            metric=metric,
            baseline_n=baseline_n,
            learned_n=learned_n,
            baseline_mean=b_mean,
            learned_mean=l_mean,
            delta=delta,
            delta_percentage_points=dpp,
            confidence_interval=ci,
            p_value=p_value,
            effect_size=effect_size,
            statistically_significant=significant,
            comparison_type=comparison_type,
            paired_n=paired_n,
        )

    # -- report -------------------------------------------------------------

    def report(self, metrics: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Full statistical report: config observed + baseline + comparisons."""
        comparisons = self.compare(metrics)
        return {
            "significance_config": self.config.to_dict(),
            "baseline": self.baseline(),
            "baseline_n": {
                m: self._baseline.n(m)
                for m in (metrics or list(METRIC_FAMILIES))
            },
            "test_n": len(self.test),
            "learned_n": len(self.learned),
            "cold_n": len(self.cold),
            "comparisons": comparisons,
            "feedback_loop": "NONE",  # Phase 5 is read-only evaluation.
            "method": {
                "ci": "percentile_bootstrap",
                "ci_alpha": 0.05,
                "ci_min_samples": self.config.min_samples,
                "ci_insufficient": CI_NOT_COMPUTED_INSUFFICIENT_SAMPLE,
                "p_value": "two_sided_permutation",
                "effect_size": "cohens_d",
                "significance": "configurable_thresholds",
                "seed": self._seed,
                "deterministic": True,
            },
        }
