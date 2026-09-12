#!/usr/bin/env python3
"""V3.2 Phase 5 — reproducible empirical COLD baseline & statistical evaluation.

Runs the Phase 5 statistical pipeline against the persisted V3.1 field-validation
COLD/LEARNED execution outcomes and prints the concrete empirical numbers that
are reported in docs/V3.2-PHASE5-REPORT.md.

Usage:
    PYTHONPATH=. python3 scripts/phase5_empirical_evaluation.py [--split test]

This is READ-ONLY evaluation: it never writes strategies, experiences, policies,
or routing, and never modifies learning artifacts. Bootstrap/CIs are seeded
deterministically from the ExperimentContext (no global random state), so the
same inputs always produce the same numbers.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from harness.learning.context import ExperimentContext  # noqa: E402
from harness.learning.statistics import (  # noqa: E402
    load_field_validation_observations,
    EmpiricalEvaluator,
    SignificanceConfig,
)

CFG = SignificanceConfig(success_rate_pp=0.05, score_delta=0.01, min_samples=10)
RUN_ID = "V3.1-FV-20260911"
BENCH = "V3.1-FIELD-VALIDATION"


def run(splitlabel: str, split_filter):
    obs = load_field_validation_observations(split=split_filter)
    cold = [o for o in obs if o.mode == "COLD"]
    learned = [o for o in obs if o.mode == "LEARNED"]
    ctx = ExperimentContext(
        validation_run_id=RUN_ID, benchmark_id=BENCH,
        task_id=splitlabel, execution_id=f"p5-{splitlabel}", mode="COLD",
    )
    ev = EmpiricalEvaluator(cold + learned, config=CFG, experiment_context=ctx)
    compar = ev.compare([
        "success_rate", "engineering_score", "latency",
        "correctness", "security", "maintainability", "efficiency",
        "iterations", "verification_failures", "retries",
    ])
    print(f"\n=== Benchmark '{splitlabel}' ===  (cold_n={len(cold)}, learned_n={len(learned)})")
    print("Empirical COLD baseline:", json.dumps(ev.baseline()))
    print("Significance config:", json.dumps(CFG.to_dict()))
    print("Method:", json.dumps(ev.report()["method"]))
    print("feedback_loop:", ev.report()["feedback_loop"])
    for key, v in compar.items():
        print(f"  {key}: n_cold={v['baseline_n']} n_learned={v['learned_n']} "
              f"cold={v['baseline_mean']} learned={v['learned_mean']} "
              f"delta={v['delta']} dpp={v['delta_percentage_points']} "
              f"ci={v['confidence_interval']} p={v['p_value']} "
              f"effect={v['effect_size']} sig={v['statistically_significant']} "
              f"type={v['comparison_type']}")


def main():
    split = None
    if len(sys.argv) > 1 and sys.argv[1].startswith("--split"):
        split = sys.argv[2]
    label = split if split else "FULL"
    run(f"split={label}", split)


if __name__ == "__main__":
    main()
