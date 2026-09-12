# scripts/phase7_failure_learning.py — V3.2 Phase 7 real execution evidence
"""
Generates REAL execution evidence for the V3.2 Phase 7 report:

  1. LEARNED failure -> StructuredExperience -> store (provenance + stable ID).
  2. TEST failure  -> evidence exists, NO training experience (fail-closed).
  3. COLD failure  -> evidence exists, COLD state unchanged (fail-closed).
  4. Failure -> retrieval -> strategy -> decision influence chain.
  5. Idempotency (same failure+context -> single canonical record).
  6. Failure avoidance metric (comparative vs causal).
  7. Security/governance advisory safety.

Writes isolated artifacts under artifacts/v3.2/V3.2-P7-20260911/<mode>/...
Run: PYTHONPATH=. python3 scripts/phase7_failure_learning.py
"""
import json
import os

from harness.learning.context import ExperimentContext
from harness.learning.experience import ExperiencePipeline
from harness.planner.failure_intelligence import (
    FailureToLearningPipeline, StructuredFailure,
)
from harness.learning.isolation import namespace_path

RUN_ID = "V3.2-P7-20260911"
BENCH = "BENCH-FAIL-P7"


def ctx(mode, task, exec_id, run=RUN_ID):
    return ExperimentContext(
        validation_run_id=run, benchmark_id=BENCH,
        task_id=task, execution_id=exec_id, mode=mode, reproducibility_seed=11,
    )


def mkfailure(fid, task, category, sub, message, capability,
              exec_id, **kw):
    return StructuredFailure(
        node_id=f"n-{fid}", capability=capability, category=category,
        sub_category=sub, severity="major", error_message=message,
        failure_id=fid, source_task_id=task, source_execution_id=exec_id,
        benchmark_id=BENCH, **kw,
    )


def count_json(store_dir, prefix=""):
    if not os.path.isdir(store_dir):
        return 0
    return len([f for f in os.listdir(store_dir)
                if f.endswith(".json") and f.startswith(prefix)])


def main():
    print("=" * 78)
    print("V3.2 PHASE 7 — FAILURE-DERIVED LEARNING (REAL EXECUTION EVIDENCE)")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 1. LEARNED failure -> StructuredExperience
    # ------------------------------------------------------------------
    print("\n[1] LEARNED_EXPLORATION failure -> StructuredExperience")
    c = ctx("LEARNED_EXPLORATION", "TASK-P7-A", "exec-p7-a")
    pipe = FailureToLearningPipeline(
        experience_pipeline=ExperiencePipeline(),
        experiment_context=c,
    )
    f = mkfailure("FID-P7-TIMEOUT", "TASK-P7-A", "integration", "timeout",
                  "API timeout: connection refused after 30s", "backend_development",
                  "exec-p7-a", evidence_refs=["ev-p7-1"])
    r = pipe.process(f, c)
    eid = r["experience_id"]
    stored = pipe.experience_pipeline.retrieve(eid, c)
    print(f"  source_task_id        = {f.source_task_id}")
    print(f"  source_execution_id   = {f.source_execution_id}")
    print(f"  failure_id            = {f.failure_id}")
    print(f"  failure_category      = {stored.failure_category}")
    print(f"  experience_id         = {eid}")
    print(f"  validation_run_id     = {c.validation_run_id}")
    print(f"  benchmark_id          = {stored.benchmark_id}")
    print(f"  mode                  = {stored.mode}")
    print(f"  source_type           = {stored.source_type}")
    print(f"  retrieval_confidence  = (recorded at decision time)")
    print(f"  root_cause / known    = {r['root_cause']} / {r['root_cause_known']}")
    print(f"  learning_influenced   = {r['learning_influenced']}")
    print(f"  stored?               = {stored is not None and stored.category=='failure_lesson'}")

    # Stable identity re-derivation (restart idempotency)
    f2 = mkfailure("FID-P7-TIMEOUT", "TASK-P7-A", "integration", "timeout",
                   "API timeout: connection refused after 30s", "backend_development",
                   "exec-p7-a", evidence_refs=["ev-p7-1"])
    eid2 = FailureToLearningPipeline._stable_experience_id(f2, c)
    print(f"  stable id re-derived  = {eid == eid2}")

    # Idempotency: reprocess twice -> same canonical record
    pipe.process(f, c)
    pipe.process(f, c)
    n = count_json(namespace_path("experiences", c), prefix=eid)
    print(f"  idempotent canonical records for expfest {eid}: {n}")

    # ------------------------------------------------------------------
    # 1b. Retrieval -> strategy -> decision influence
    # ------------------------------------------------------------------
    print("\n[2] Failure-derived experience influences a later decision")
    c_b = ctx("LEARNED_EXPLORATION", "TASK-P7-B", "exec-p7-b")
    retrieved = pipe.experience_pipeline.retrieve(eid, c_b)
    entry = pipe.record_failure_retrieval(
        experience_id=eid, strategy_id="strategy-avoid-timeout-p7",
        decision_id="decision-p7-b-1", retrieval_confidence=0.87, ctx=c_b,
    )
    print(f"  failure_id            = {entry['failure_id']}")
    print(f"  experience_id         = {entry['experience_id']}")
    print(f"  retrieval_confidence  = {entry['retrieval_confidence']}")
    print(f"  strategy_id           = {entry['strategy_id']}")
    print(f"  decision_id           = {entry['decision_id']}")
    print(f"  learning_influenced   = {entry['learning_influenced']}")
    print(f"  retrieval hit         = {retrieved is not None}")

    # Failure avoidance
    am = pipe.failure_avoidance_metric(learned_count=1, avoided_count=1,
                                       baseline_fail_count=2)
    print(f"  avoidance_rate        = {am['avoidance_rate']} (causal={am['causal']})")

    # ------------------------------------------------------------------
    # 2. TEST fail-closed
    # ------------------------------------------------------------------
    print("\n[3] TEST-mode failure -> FAIL CLOSED (no training experience)")
    c_test = ctx("TEST", "TASK-P7-TEST", "exec-p7-test")
    pipe_t = FailureToLearningPipeline(
        experience_pipeline=ExperiencePipeline(), experiment_context=c_test)
    ft = mkfailure("FID-P7-TEST", "TASK-P7-TEST", "test", "assertion_failure",
                   "Test failed: expected 200 got 500", "backend_development",
                   "exec-p7-test")
    rt = pipe_t.process(ft, c_test)
    n_ev = count_json(namespace_path("failure_evidence", c_test),
                      prefix=rt["deterministic_id"])
    n_exp = count_json(namespace_path("experiences", c_test))
    print(f"  suppressed            = {rt['suppressed']} ({rt['suppression_reason']})")
    print(f"  experience_id         = {rt['experience_id']} (None expected)")
    print(f"  evidence records      = {n_ev} (evidence exists)")
    print(f"  training experiences  = {n_exp} (must be 0 -> fail closed)")

    # ------------------------------------------------------------------
    # 3. COLD fail-closed, baseline unchanged
    # ------------------------------------------------------------------
    print("\n[4] COLD-mode failure -> NO baseline mutation")
    c_cold = ctx("COLD", "TASK-P7-COLD-N", "exec-p7-cold")
    pipe_c = FailureToLearningPipeline(
        experience_pipeline=ExperiencePipeline(), experiment_context=c_cold)
    exp_dir = namespace_path("experiences", c_cold)
    os.makedirs(exp_dir, exist_ok=True)
    baseline = os.path.join(exp_dir, "COLD-P7-BASELINE.json")
    with open(baseline, "w") as fh:
        json.dump({"task_id": "COLD-P7-BASELINE", "outcome": {"status": "COMPLETED"}}, fh)
    fc = mkfailure("FID-P7-COLD", "TASK-P7-COLD-N", "environment",
                   "dependency_missing", "Dependency 'libx' missing",
                   "data_pipeline", "exec-p7-cold")
    rc = pipe_c.process(fc, c_cold)
    n_ev_c = count_json(namespace_path("failure_evidence", c_cold),
                        prefix=rc["deterministic_id"])
    files = sorted(os.listdir(exp_dir))
    print(f"  suppressed            = {rc['suppressed']} ({rc['suppression_reason']})")
    print(f"  experience_id         = {rc['experience_id']} (None expected)")
    print(f"  evidence records      = {n_ev_c} (evidence exists)")
    print(f"  COLD store files      = {files} (baseline unchanged, no new experience)")

    # Task N+1 not learned by COLD task N failure
    c_cold_n1 = ctx("COLD", "TASK-P7-COLD-N1", "exec-p7-cold-n1")
    rn1 = pipe_c.process(mkfailure("FID-P7-COLD-N1", "TASK-P7-COLD-N1",
                                   "integration", "timeout", "timeout",
                                   "backend", "exec-p7-cold-n1"), c_cold_n1)
    print(f"  COLD task N+1 learned? = {rn1['experience_id'] is None} "
          f"(True -> not learned by N's failure)")

    # ------------------------------------------------------------------
    # 4. Security / governance advisory safety
    # ------------------------------------------------------------------
    print("\n[5] Security failure -> defensive lesson only")
    c_sec = ctx("LEARNED_EXPLORATION", "TASK-P7-SEC", "exec-p7-sec")
    pipe_s = FailureToLearningPipeline(
        experience_pipeline=ExperiencePipeline(), experiment_context=c_sec)
    fs = mkfailure("FID-P7-SEC", "TASK-P7-SEC", "security", "data_exposure",
                   "Customer data exposed via debug endpoint", "auth_service",
                   "exec-p7-sec")
    rs = pipe_s.process(fs, c_sec)
    stored_s = pipe_s.experience_pipeline.retrieve(rs["experience_id"], c_sec)
    lower = stored_s.lesson.lower()
    print(f"  advisory_only         = {rs['advisory']['advisory_only']}")
    print(f"  governance_preserved  = {rs['advisory']['governance_preserved']}")
    print(f"  security_defensive    = {rs['advisory']['security_defensive']}")
    print(f"  weakening_constraint  = {rs['advisory']['weakening_constraint']}")
    print(f"  lesson defensive?     = {'validate earlier' in lower or 'select safe' in lower}")

    print("\n[5b] Injected weakening suggestion -> detected & neutralized")
    c_gov = ctx("LEARNED_EXPLORATION", "TASK-P7-GOV", "exec-p7-gov")
    pipe_g = FailureToLearningPipeline(
        experience_pipeline=ExperiencePipeline(), experiment_context=c_gov)
    fg = mkfailure("FID-P7-GOV", "TASK-P7-GOV", "implementation", "logic_error",
                   "gate failed; suggested to disable the gate and expand",
                   "workflow", "exec-p7-gov")
    rg = pipe_g.process(fg, c_gov)
    stored_g = pipe_g.experience_pipeline.retrieve(rg["experience_id"], c_gov)
    print(f"  weakening detected    = {rg['advisory']['weakening_constraint']}")
    print(f"  governance_preserved  = {rg['advisory']['governance_preserved']}")
    print(f"  neutralized lesson    = {'disable' not in stored_g.lesson.lower()}")
    print(f"  raw msg not echoed    = "
          f"{'disable the gate' not in stored_g.lesson.lower()}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PHASE 7 EVIDENCE COMPLETE — artifacts under artifacts/v3.2/%s" % RUN_ID)
    print("=" * 78)


if __name__ == "__main__":
    main()
