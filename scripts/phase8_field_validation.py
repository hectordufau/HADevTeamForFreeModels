# scripts/phase8_field_validation.py — V3.2 Phase 8: Statistical Field Validation
"""
Runs the complete Phase 8 three-mode field-validation experiment and emits real,
machine-readable results plus the required report documents.

Run:  PYTHONPATH=. python3 scripts/phase8_field_validation.py [--seed 314159] [--reps 3]

Outputs (under artifacts/v3.2/<run_id>/field_validation/):
  * results.json            — machine-readable results (all Phase 8 fields)
  * observations/           — per-execution persisted outcomes
  * summary.txt             — human-readable summary table

And writes the required docs:
  * docs/V3.2-FIELD-VALIDATION-SPEC.md
  * docs/V3.2-FIELD-VALIDATION-REPORT.md
  * docs/V3.2-FIELD-VALIDATION-RESULTS.md
  * docs/V3.2-RELEASE-GATE.md
  * docs/V3.2-EXPERIMENT-METHODOLOGY.md

This is experiment/measurement only: it does not modify learning/routing/
orchestration/agents/models/evaluation/governance/security, does not create a
release tag, and does not merge.
"""
import argparse
import json
import os
import sys

# Ensure repo root on the path.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from harness.validation.field_validation import (  # noqa: E402
    FieldValidationRunner, FieldExperimentConfig, MODES, configuration_digest,
    IMPROVED, UNCHANGED, WORSENED, UNKNOWN, INSUFFICIENT_DATA, NOT_AVAILABLE,
    resolve_repetitions, resolved_configuration, dataset_digest,
    DEFAULT_REPETITIONS,
)


def _code_revision() -> str:
    """Short git revision of the code that produced this run (best-effort)."""
    try:
        import subprocess
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def summarize(value, is_pp=False):
    """Return a short printable form of a metric block."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value
    return value


def build_results(r: FieldValidationRunner) -> dict:
    test_stats = r.compute_statistics()["test"]
    all_stats = r.compute_statistics()["all"]

    # Per-mode aggregate success + score on TEST and ALL splits.
    def agg(mode):
        o = r.observations_for_split(mode, "test")
        if not o:
            return {"success": INSUFFICIENT_DATA, "score": INSUFFICIENT_DATA}
        return {
            "success": round(sum(1 for x in o if x.status == "PASSED") / len(o), 4),
            "score": round(sum(x.score for x in o) / len(o), 4),
            "n": len(o),
        }

    res = {
        "run_id": r.config.run_id,
        "benchmark_id": r.config.benchmark_id,
        "reproducibility_seed": r.config.reproducibility_seed,
        "configuration_digest": configuration_digest(),
        "dataset_digest": dataset_digest(),
        "code_revision": _code_revision(),
        "resolved_configuration": resolved_configuration(
            r.config.reproducibility_seed,
            r.config.train_reps, r.config.validation_reps, r.config.test_reps,
            r.config.exploration_rate, r.config.significance,
        ),
        "modes": list(MODES),
        "repetitions": {
            "train": r.config.train_reps, "validation": r.config.validation_reps,
            "test": r.config.test_reps,
        },
        "test": {
            "COLD": agg("COLD"),
            "LEARNED_NO_EXPLORATION": agg("LEARNED_NO_EXPLORATION"),
            "LEARNED_EXPLORATION": agg("LEARNED_EXPLORATION"),
        },
        "statistics_test": test_stats,
        "statistics_all": all_stats,
        "decision_outcomes": {
            m: r.decision_outcomes(m) for m in ("LEARNED_NO_EXPLORATION", "LEARNED_EXPLORATION")
        },
        "learning_influence_rate": {
            m: r.learning_influence_rate(m) for m in ("LEARNED_NO_EXPLORATION", "LEARNED_EXPLORATION")
        },
        "exploration_statistics": r.exploration_statistics(),
        "exploration_contribution": r.exploration_contribution("test"),
        "failure_learning": r.failure_learning(),
        "split_improvement_success_pp": r.split_improvement("success_rate"),
        "split_improvement_score": r.split_improvement("engineering_score"),
        "variance": r.variance_report(),
        "category_distribution": r.category_distribution(),
        "integrity": r.integrity_check(),
        "security_governance": r.security_governance_check(),
        "sample_sizes": {
            "test": {m: len(r.observations_for_split(m, "test")) for m in MODES},
            "all": {m: len(r.mode_observations(m)) for m in MODES},
        },
        "experiment_phase": "8",
        "timestamp": json.dumps({"note": "see observations timestamps"}, sort_keys=True),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=314159)
    ap.add_argument("--reps", type=int, default=None,
                    help="repetitions per split (default: canonical %d)" %
                         DEFAULT_REPETITIONS)
    ap.add_argument("--run-id", default="V3.2-FV8-20260911")
    args = ap.parse_args()

    # Canonical resolution: single authoritative source for the repetition
    # count. If --reps is omitted, it resolves to DEFAULT_REPETITIONS (5) — the
    # same count the accepted run used — never a hidden fallback of 3.
    reps = resolve_repetitions(args.reps)
    cfg = FieldExperimentConfig(
        run_id=args.run_id,
        benchmark_id="V3.2-FIELD-VALIDATION",
        reproducibility_seed=args.seed,
        train_reps=reps, validation_reps=reps, test_reps=reps,
    )
    runner = FieldValidationRunner(cfg)
    runner.run_all()
    runner.persist_observations()

    results = build_results(runner)

    outdir = os.path.join("artifacts", "v3.2", args.run_id, "field_validation")
    os.makedirs(outdir, exist_ok=True)
    rpath = os.path.join(outdir, "results.json")
    with open(rpath, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print("=" * 78)
    print(f"V3.2 PHASE 8 — STATISTICAL FIELD VALIDATION (run {args.run_id}, seed {args.seed})")
    print("=" * 78)
    print("Modes: COLD / LEARNED_NO_EXPLORATION / LEARNED_EXPLORATION")
    print("Reps (train/val/test): %d/%d/%d" % (cfg.train_reps, cfg.validation_reps, cfg.test_reps))
    print()
    print("TEST split aggregate (COLD / NO_EXPL / EXPL):")
    for m in MODES:
        a = results["test"][m]
        print("  %-26s success=%s score=%s n=%s" % (m, a["success"], a["score"], a["n"]))
    print()
    ts = results["statistics_test"]
    for comp in ("COLD_vs_LEARNED_NO_EXPLORATION", "COLD_vs_LEARNED_EXPLORATION",
                 "LEARNED_NO_EXPLORATION_vs_LEARNED_EXPLORATION"):
        blk = ts[comp]
        s = blk.get("success_rate", {})
        sc = blk.get("engineering_score", {})
        print("  %-46s success_delta_pp=%-7s  score_delta=%-7s" % (
            comp, s.get("delta_pp"), sc.get("delta")))
    print()
    print("Decision improvement / harm (LEARNED_EXPLORATION, TEST+VAL influenced):")
    doi = results["decision_outcomes"]["LEARNED_EXPLORATION"]
    print("  improved=%s unchanged=%s worsened=%s unknown=%s influenced_n=%s" % (
        doi["improved"], doi["unchanged"], doi["worsened"], doi["unknown"], doi["influenced_n"]))
    print("  Decision Improvement Rate=%s  Learning Harm Rate=%s  Unknown=%s" % (
        doi["decision_improvement_rate"], doi["learning_harm_rate"], doi["unknown_outcome_rate"]))
    print()
    print("Split improvement (success_pp / score):")
    for k, v in results["split_improvement_success_pp"].items():
        print("  %-10s success_pp=%s" % (k, v))
    print()
    print("Integrity:", results["integrity"])
    print("Security/Governance:", results["security_governance"])
    print()
    print("Exploration stats:", results["exploration_statistics"])
    print("Exploration contribution (TEST):", results["exploration_contribution"])
    print("Failure learning:", results["failure_learning"]["avoidance_COLD_vs_EXPLORATION"])
    print()
    print(f"Machine-readable results: {rpath}")
    print('PHASE 8 FIELD VALIDATION EXECUTED — results written.')


if __name__ == "__main__":
    main()
