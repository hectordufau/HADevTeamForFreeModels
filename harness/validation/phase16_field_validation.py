"""P16.9 reproducibility and P16.10 analysis for the frozen Phase 16 run."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .phase16_accepted import AcceptedRunConfig, AcceptedRunRunner, MODES, cluster_aware_analysis
from .phase16_preexperiment import digest, load_benchmark, load_knowledge

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "V3.3-P16-FV-20260918-R2"
RUN_DIR = ROOT / "artifacts" / "v3.3" / "phase16" / RUN_ID


def load_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def paired_statistics(deltas: Sequence[float], config: AcceptedRunConfig, seed: int) -> Dict[str, Any]:
    """Deterministic cluster-level bootstrap CI and sign-flip permutation p-value."""
    import random
    n = len(deltas)
    if not n:
        return {"n_clusters": 0, "bootstrap_ci_95": None, "permutation_p_two_sided": None}
    rng = random.Random(seed)
    boot = [sum(rng.choice(deltas) for _ in range(n)) / n for _ in range(config.bootstrap_iterations)]
    observed = sum(deltas) / n
    extreme = 0
    for _ in range(config.permutation_iterations):
        signed = [value if rng.getrandbits(1) else -value for value in deltas]
        if abs(sum(signed) / n) >= abs(observed):
            extreme += 1
    p_value = (extreme + 1) / (config.permutation_iterations + 1)
    return {
        "n_clusters": n,
        "mean_paired_delta": round(observed, 12),
        "bootstrap_ci_95": [round(_percentile(boot, .025), 12), round(_percentile(boot, .975), 12)],
        "permutation_p_two_sided": round(p_value, 12),
        "bootstrap_iterations": config.bootstrap_iterations,
        "permutation_iterations": config.permutation_iterations,
        "statistical_unit": config.statistical_unit,
    }


def verdict(stats: Mapping[str, Any], config: AcceptedRunConfig) -> str:
    delta = stats.get("mean_paired_delta")
    p_value = stats.get("permutation_p_two_sided")
    if delta is None or p_value is None:
        return "UNAVAILABLE"
    if p_value < config.alpha and abs(delta) >= config.practical_threshold:
        return "POSITIVE" if delta > 0 else "NEGATIVE"
    return "NEUTRAL"


def contamination_purity(rows: Sequence[Mapping[str, Any]], benchmark: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    expected_ids = {str(task["id"]) for task in benchmark}
    row_ids = {str(row["task_cluster_id"]) for row in rows}
    modes = {str(row["mode"]) for row in rows}
    checks = {
        "control_knowledge_retrieved_zero": all(row["knowledge_retrieved"] == 0 for row in rows if row["mode"] == "CONTROL"),
        "control_knowledge_influencing_zero": all(row["knowledge_influencing"] == 0 for row in rows if row["mode"] == "CONTROL"),
        "learning_only_plus": all(row["learning_applied"] is (row["mode"] == "KNOWLEDGE_PLUS_LEARNING") for row in rows),
        "learning_effect_only_plus": all(row["learning_score_effect"] == (0.01 if row["mode"] == "KNOWLEDGE_PLUS_LEARNING" else 0.0) for row in rows),
        "none_tasks_not_retrieved": all(row["knowledge_retrieved"] == 0 for row in rows if row["task_id"] in {"P16-004", "P16-008", "P16-012", "P16-016", "P16-020"}),
        "accepted_split_only": all(row["split"] == "accepted" for row in rows),
        "benchmark_ids_exact": row_ids == expected_ids,
        "modes_exact": modes == set(MODES),
        "no_answer_fields": not any("answer" in row or "expected_answer" in row for row in rows),
    }
    return {"checks": checks, "passed": all(checks.values())}


def analyze_raw(rows: Sequence[Mapping[str, Any]], config: AcceptedRunConfig) -> Dict[str, Any]:
    base = cluster_aware_analysis(rows, config)
    for label in ("primary", "secondary", "exploratory"):
        comparison = base[label]
        stats = paired_statistics(comparison["paired_deltas"], config, config.seed + len(label))
        comparison["statistics"] = stats
        comparison["verdict"] = verdict(stats, config)
    return base


def reproducibility(rows: Sequence[Mapping[str, Any]], config: AcceptedRunConfig, benchmark: Sequence[Mapping[str, Any]], knowledge: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    fresh_a = AcceptedRunRunner(config, benchmark, knowledge)
    fresh_b = AcceptedRunRunner(config, benchmark, knowledge)
    expected_a = [fresh_a._observation(task, mode, rep) for mode in MODES for task in benchmark for rep in range(1, config.repetitions + 1)]
    expected_b = [fresh_b._observation(task, mode, rep) for mode in MODES for task in benchmark for rep in range(1, config.repetitions + 1)]
    return {
        "classification": "R-A_FULLY_REPRODUCIBLE" if expected_a == expected_b == list(rows) else "R-C_NOT_REPRODUCIBLE",
        "same_process_construction_equal": expected_a == expected_b,
        "fresh_process_construction_equal": _fresh_process_equal(config, benchmark, knowledge, rows),
        "raw_matches_fresh_construction": expected_a == list(rows),
    }


def _fresh_process_equal(config: AcceptedRunConfig, benchmark: Sequence[Mapping[str, Any]], knowledge: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> bool:
    code = (
        "import json; from harness.validation.phase16_accepted import AcceptedRunRunner, MODES; "
        "r=AcceptedRunRunner(__import__('harness.validation.phase16_accepted', fromlist=['AcceptedRunConfig']).AcceptedRunConfig.corrected()); b=r.benchmark; "
        "x=[r._observation(t,m,n) for m in MODES for t in b for n in range(1,r.config.repetitions+1)]; "
        "print(json.dumps(x,sort_keys=True,separators=(',',':')))")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(result.stdout) == list(rows)


def build_artifacts(run_dir: Path = RUN_DIR) -> Dict[str, Any]:
    config = AcceptedRunConfig.corrected()
    benchmark, knowledge = load_benchmark(), load_knowledge()
    rows = load_rows(run_dir / "raw_results.jsonl")
    raw_bytes = (run_dir / "raw_results.jsonl").read_bytes()
    analysis = analyze_raw(rows, config)
    checks = contamination_purity(rows, benchmark)
    reproduction = reproducibility(rows, config, benchmark, knowledge)
    result = {
        "schema_version": "v3.3-phase16-field-validation-analysis",
        "run_id": RUN_ID,
        "accepted_run_executed": True,
        "raw_observation_count": len(rows),
        "task_cluster_count": len({row["task_cluster_id"] for row in rows}),
        "repetitions": config.repetitions,
        "independent_tasks": len(benchmark),
        "config_identity": {"path": "config/phase16/accepted-r2.json", "sha256": hashlib.sha256((ROOT / "config/phase16/accepted-r2.json").read_bytes()).hexdigest(), "canonical_digest": digest(json.loads((ROOT / "config/phase16/accepted-r2.json").read_text()))},
        "benchmark_identity": {"path": "config/phase16/benchmark.json", "sha256": hashlib.sha256((ROOT / "config/phase16/benchmark.json").read_bytes()).hexdigest(), "canonical_digest": digest(benchmark)},
        "knowledge_identity": {"path": "config/phase16/knowledge.json", "sha256": hashlib.sha256((ROOT / "config/phase16/knowledge.json").read_bytes()).hexdigest(), "canonical_digest": digest(knowledge)},
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "contamination_purity": checks,
        "reproducibility": reproduction,
        "analysis": analysis,
        "metrics_scope": {"score": "instrumented synthetic score", "latency": None, "safety": None, "cost": None},
    }
    (run_dir / "derived_analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def render_report(result: Mapping[str, Any], path: Path) -> None:
    a = result["analysis"]
    lines = [
        "# V3.3 Phase 16 Field Validation Report",
        "",
        f"Run: `{result['run_id']}`  ",
        "Scope: frozen Phase 16 benchmark/configuration only; historical Phase 13/V3.2 data are not combined.",
        "",
        "## Protocol and identity",
        f"- Raw executions: **{result['raw_observation_count']}** (20 independent benchmark tasks × 3 modes × 10 repeated observations).",
        f"- Paired statistical unit: **20 task clusters**; repetitions are not independent N.",
        f"- Raw SHA-256: `{result['raw_sha256']}`",
        f"- Configuration SHA-256: `{result['config_identity']['sha256']}`",
        f"- Benchmark SHA-256: `{result['benchmark_identity']['sha256']}`",
        f"- Knowledge SHA-256: `{result['knowledge_identity']['sha256']}`",
        "",
        "## Reproducibility and purity",
        f"- Classification: **{result['reproducibility']['classification']}**.",
        f"- Same-process construction equal: `{result['reproducibility']['same_process_construction_equal']}`; fresh-process construction equal: `{result['reproducibility']['fresh_process_construction_equal']}`.",
        f"- Raw rows match fresh construction: `{result['reproducibility']['raw_matches_fresh_construction']}`.",
        f"- Contamination/purity checks passed: **{result['contamination_purity']['passed']}**.",
        "",
        "## Paired results",
        "| Comparison | Clusters | Mean paired delta | 95% bootstrap CI | permutation p (two-sided) | Verdict |",
        "|---|---:|---:|---|---:|---|",
    ]
    for label, title in (("primary", "CONTROL → KNOWLEDGE_ONLY"), ("secondary", "KNOWLEDGE_ONLY → KNOWLEDGE_PLUS_LEARNING"), ("exploratory", "CONTROL → KNOWLEDGE_PLUS_LEARNING")):
        c = a[label]; s = c["statistics"]
        lines.append(f"| {title} | {s['n_clusters']} | {s['mean_paired_delta']:.12f} | [{s['bootstrap_ci_95'][0]:.12f}, {s['bootstrap_ci_95'][1]:.12f}] | {s['permutation_p_two_sided']:.12f} | **{c['verdict']}** |")
    lines += [
        "",
        "## Metric limits and final verdict",
        "Only the instrumented synthetic `score` is analyzed. Latency, safety, and cost were not instrumented and are reported as unavailable; no absent metric is treated as zero.",
        "",
        "The verdicts above apply only to this deterministic benchmark, frozen configuration, and synthetic harness. They do not establish production effectiveness or generalize to other phases/releases.",
        "",
        "## Artifact references",
        f"- Raw results: `artifacts/v3.3/phase16/{RUN_ID}/raw_results.jsonl`",
        f"- Raw digest: `artifacts/v3.3/phase16/{RUN_ID}/raw_results.sha256`",
        f"- Derived analysis: `artifacts/v3.3/phase16/{RUN_ID}/derived_analysis.json`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = build_artifacts()
    report = ROOT / "docs" / "V3.3-PHASE16-FIELD-VALIDATION-REPORT.md"
    render_report(result, report)
    print(json.dumps({"report": str(report), "raw_sha256": result["raw_sha256"], "executions": result["raw_observation_count"], "task_clusters": result["task_cluster_count"]}, sort_keys=True))
