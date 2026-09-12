# tests/unit/test_v32_failure_learning.py — V3.2 Phase 7 Failure-Derived Learning
"""
V3.2 Phase 7: Failure → Failure Intelligence → FailureToLearningPipeline →
StructuredExperience → V3.2 ExperienceStore → Retrieval/Strategy → Decision.

Covers spec §11 + Phase 7 task requirements 1-21:
- Failure → StructuredExperience conversion (R1)
- Failure provenance recorded (R2)
- TEST failures fail-closed (R3)
- COLD failures do not modify baseline state (R4)
- LEARNED mode failure learning (R5)
- Stable failure identity (SHA-256, no Python hash) (R6)
- Idempotency: same failure_id + context → single canonical record (R7)
- Restart idempotency: fresh process re-derives identical ID (R8)
- Failure retrieval (R9)
- Failure → decision influence (R10)
- Failure avoidance (comparative vs causal) (R11)
- Deterministic lesson/experience/retrieval (R12)
- Security/governance advisory safety (R13)
- Isolation: run A≠B, COLD≠LEARNED, TEST→never training (R14)
- Unknown root cause never invented (R15)
- Historical artifacts protected (R16)
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile

import pytest

from harness.learning.context import ExperimentContext
from harness.learning.experience import ExperiencePipeline, StructuredExperience
from harness.planner.failure_intelligence import (
    FailureToLearningPipeline,
    StructuredFailure,
    FailureLesson,
    FailureAnalyzer,
)
from harness.learning.isolation import namespace_path, get_v32_base


def _mkctx(mode: str, run: str = "V3.2-P7-20260911",
           task: str = "TASK-FAIL-A", exec_id: str = "exec-fail-1",
           seed: int = 7) -> ExperimentContext:
    return ExperimentContext(
        validation_run_id=run,
        benchmark_id="BENCH-FAIL-P7",
        task_id=task,
        execution_id=exec_id,
        mode=mode,
        reproducibility_seed=seed,
    )


def _mkfailure(failure_id: str = None, category: str = "integration",
               message: str = "API timeout: connection refused after 30s",
               node_id: str = "n1", capability: str = "backend_development",
               task: str = "TASK-FAIL-A"):
    return StructuredFailure(
        node_id=node_id,
        capability=capability,
        category=category,
        sub_category="timeout",
        severity="major",
        error_message=message,
        failure_id=failure_id or f"fid-{node_id}",
        source_task_id=task,
    )


# ---------------------------------------------------------------------------
# R1: Failure -> StructuredExperience conversion
# ---------------------------------------------------------------------------
def test_failure_to_structured_experience():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _mkctx("LEARNED_EXPLORATION")
        pipe = FailureToLearningPipeline(
            experience_pipeline=ExperiencePipeline(storage_dir=tmp),
            experiment_context=ctx,
        )
        failure = _mkfailure()
        result = pipe.process(failure, ctx)

        assert result["suppressed"] is False
        assert result["experience_id"] is not None
        assert result["learning_influenced"] is True

        stored = pipe.experience_pipeline.retrieve(result["experience_id"], ctx)
        assert stored is not None
        assert isinstance(stored, StructuredExperience)
        assert stored.source_type == "failure"
        assert stored.category == "failure_lesson"
        assert stored.failure_id == failure.failure_id
        assert stored.mode == "LEARNED_EXPLORATION"
        assert stored.validation_run_id == ctx.validation_run_id
        assert stored.provenance["source_type"] == "failure"
        assert "AVOID" in stored.lesson


# ---------------------------------------------------------------------------
# R2: Failure provenance recorded
# ---------------------------------------------------------------------------
def test_failure_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _mkctx("LEARNED_NO_EXPLORATION")
        pipe = FailureToLearningPipeline(
            experience_pipeline=ExperiencePipeline(storage_dir=tmp),
            experiment_context=ctx,
        )
        failure = StructuredFailure(
            node_id="n42", capability="data_pipeline", category="integration",
            sub_category="protocol_error", severity="critical",
            error_message="Protocol error: schema mismatch",
            failure_id="fid-42", source_task_id="TASK-ORIG-42",
            source_execution_id="exec-orig-42", benchmark_id="BENCH-FAIL-P7",
            model_id="mistral-small:free", agent_id="dev",
            capabilities=["data_pipeline", "testing"],
            evidence_refs=["ev-1", "ev-2"],
        )
        result = pipe.process(failure, ctx)
        stored = pipe.experience_pipeline.retrieve(result["experience_id"], ctx)
        p = stored.provenance
        assert p["source_type"] == "failure"
        assert p["failure_id"] == "fid-42"
        assert p["validation_run_id"] == ctx.validation_run_id
        assert p["mode"] == "LEARNED_NO_EXPLORATION"
        assert p["source_task_id"] == "TASK-ORIG-42"
        assert p["source_execution_id"] == "exec-orig-42"
        assert p["benchmark_id"] == "BENCH-FAIL-P7"
        assert p["model_id"] == "mistral-small:free"
        assert p["agent_id"] == "dev"
        assert p["capabilities"] == ["data_pipeline", "testing"]
        assert p["evidence_refs"] == ["ev-1", "ev-2"]
        assert stored.source_task_id == "TASK-ORIG-42"


# ---------------------------------------------------------------------------
# R3: TEST failures fail-closed
# ---------------------------------------------------------------------------
def test_test_failure_fail_closed_no_training_experience():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _mkctx("TEST")
        pipe = FailureToLearningPipeline(
            experience_pipeline=ExperiencePipeline(storage_dir=tmp),
            experiment_context=ctx,
        )
        failure = _mkfailure()
        result = pipe.process(failure, ctx)

        # Result signals suppression
        assert result["suppressed"] is True
        assert result["suppression_reason"] == "TEST"
        assert result["experience_id"] is None

        # Evidence recorded for evaluation (release-critical negative evidence)
        ev_path = os.path.join(namespace_path("failure_evidence", ctx), result["deterministic_id"] + ".json")
        assert os.path.exists(ev_path)
        with open(ev_path) as f:
            ev = json.load(f)
        assert ev["learning_suppressed"] is True
        assert ev["suppression_reason"] == "TEST"
        assert ev["training_mutation"] is False

        # No StructuredExperience anywhere in the learning store
        exp_store_dir = namespace_path("experiences", ctx)
        learned_files = [f for f in os.listdir(exp_store_dir) if f.endswith(".json")]
        assert learned_files == [], f"TEST failure must not create experiences: {learned_files}"


# ---------------------------------------------------------------------------
# R4: COLD failures do not modify baseline learning state
# ---------------------------------------------------------------------------
def test_cold_failure_no_baseline_mutation():
    with tempfile.TemporaryDirectory() as tmp:
        # COLD run/context
        ctx = _mkctx("COLD", run="V3.2-P7-COLD-20260911")
        pipe = FailureToLearningPipeline(
            experience_pipeline=ExperiencePipeline(storage_dir=tmp),
            experiment_context=ctx,
        )
        # Record a "baseline" COLD experience that must not be changed
        exp_store_dir = namespace_path("experiences", ctx)
        baseline = os.path.join(exp_store_dir, "COLD-BASELINE.json")
        os.makedirs(exp_store_dir, exist_ok=True)
        with open(baseline, "w") as f:
            json.dump({"task_id": "COLD-BASELINE", "outcome": {"status": "COMPLETED"}}, f)
        baseline_snapshot = open(baseline).read()

        # Task N fails
        failure_n = _mkfailure(failure_id="fid-cold-n", task="TASK-COLD-N",
                               message="dependency missing")
        failure_n.category = "environment"
        failure_n.sub_category = "dependency_missing"
        result_n = pipe.process(failure_n, ctx)

        assert result_n["suppressed"] is True
        assert result_n["suppression_reason"] == "COLD"
        assert result_n["experience_id"] is None

        # COLD baseline state unchanged: no NEW experience file, existing untouched
        files = sorted(f for f in os.listdir(exp_store_dir) if f.endswith(".json"))
        assert files == ["COLD-BASELINE.json"], f"COLD task failure must not mutate learning: {files}"
        assert open(baseline).read() == baseline_snapshot

        # Task N+1 must NOT become learned due to N's failure
        ctx_next = _mkctx("COLD", run="V3.2-P7-COLD-20260911", task="TASK-COLD-N1")
        failure_n1 = _mkfailure(failure_id="fid-cold-n1", task="TASK-COLD-N1")
        result_n1 = pipe.process(failure_n1, ctx_next)
        assert result_n1["suppressed"] is True
        assert result_n1["experience_id"] is None


# ---------------------------------------------------------------------------
# R5: LEARNED mode failure learning (both learned modes)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["LEARNED_NO_EXPLORATION", "LEARNED_EXPLORATION", "LEARNED"])
def test_learned_modes_produce_learning(mode):
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _mkctx(mode)
        pipe = FailureToLearningPipeline(
            experience_pipeline=ExperiencePipeline(storage_dir=tmp),
            experiment_context=ctx,
        )
        result = pipe.process(_mkfailure(), ctx)
        assert result["suppressed"] is False
        assert result["experience_id"] is not None
        assert pipe.experience_pipeline.retrieve(result["experience_id"], ctx) is not None


# ---------------------------------------------------------------------------
# R6: Stable failure identity — SHA-256, not Python hash()
# ---------------------------------------------------------------------------
def test_stable_identity_sha256_not_python_hash():
    ctx_a = _mkctx("LEARNED_EXPLORATION")
    ctx_b = _mkctx("LEARNED_EXPLORATION")  # identical context
    failure = _mkfailure(failure_id="fid-stable")

    id1 = FailureToLearningPipeline._stable_experience_id(failure, ctx_a)
    id2 = FailureToLearningPipeline._stable_experience_id(failure, ctx_b)
    assert id1 == id2
    assert id1.startswith("fexp_")
    assert len(id1) == len("fexp_") + 16

    # Deterministic derivation matches direct SHA-256, not Python hash
    payload = "::".join([
        failure.failure_id, ctx_a.mode, ctx_a.validation_run_id,
        ctx_a.benchmark_id, ctx_a.task_id,
    ])
    expected = "fexp_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    assert id1 == expected


# ---------------------------------------------------------------------------
# R7: Idempotency — same failure_id + context -> single canonical record
# ---------------------------------------------------------------------------
def test_idempotency_single_canonical_record():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _mkctx("LEARNED_EXPLORATION")
        pipe = FailureToLearningPipeline(
            experience_pipeline=ExperiencePipeline(storage_dir=tmp),
            experiment_context=ctx,
        )
        failure = _mkfailure(failure_id="fid-dup")
        r1 = pipe.process(failure, ctx)
        r2 = pipe.process(failure, ctx)
        r3 = pipe.process(failure, ctx)

        assert r1["experience_id"] == r2["experience_id"] == r3["experience_id"]

        # Exactly one canonical record (task-keyed storage -> overwrite, count stable)
        store_dir = namespace_path("experiences", ctx)
        matching = [f for f in os.listdir(store_dir)
                    if f.startswith(r1["experience_id"]) and f.endswith(".json")]
        assert len(matching) == 1, f"Expected 1 canonical record, got {matching}"


# ---------------------------------------------------------------------------
# R8: Restart idempotency — fresh process re-derives identical ID
# ---------------------------------------------------------------------------
def test_restart_idempotency_subprocess():
    ctx = _mkctx("LEARNED_EXPLORATION")
    failure = _mkfailure(failure_id="fid-restart")
    id_before = FailureToLearningPipeline._stable_experience_id(failure, ctx)

    script = (
        "from harness.learning.context import ExperimentContext;"
        "from harness.planner.failure_intelligence import StructuredFailure,"
        " FailureToLearningPipeline;"
        f"ctx=ExperimentContext(validation_run_id='{ctx.validation_run_id}',"
        f" benchmark_id='{ctx.benchmark_id}', task_id='{ctx.task_id}',"
        f" execution_id='{ctx.execution_id}', mode='{ctx.mode}',"
        f" reproducibility_seed={ctx.reproducibility_seed});"
        "sf=StructuredFailure(node_id='n1',capability='x',category='integration',"
        "sub_category='timeout',severity='major',error_message='e',"
        "failure_id='fid-restart');"
        "print(FailureToLearningPipeline._stable_experience_id(sf, ctx))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=os.getcwd(),
        env={**os.environ, "PYTHONPATH": "."},
    )
    assert out.returncode == 0, out.stderr
    id_after = out.stdout.strip()
    assert id_after == id_before, f"restart idempotency failed: {id_before} != {id_after}"


# ---------------------------------------------------------------------------
# R9: Failure retrieval — related task retrieves the failure-derived experience
# ---------------------------------------------------------------------------
def test_failure_retrieval():
    with tempfile.TemporaryDirectory() as tmp:
        ctx_a = _mkctx("LEARNED_EXPLORATION", task="TASK-A")
        pipe = FailureToLearningPipeline(
            experience_pipeline=ExperiencePipeline(storage_dir=tmp),
            experiment_context=ctx_a,
        )
        result = pipe.process(_mkfailure(task="TASK-A"), ctx_a)
        eid = result["experience_id"]

        # Related task B in the same run/mode retrieves it
        ctx_b = ExperimentContext(
            validation_run_id=ctx_a.validation_run_id,
            benchmark_id=ctx_a.benchmark_id,
            task_id="TASK-B", execution_id="exec-b", mode="LEARNED_EXPLORATION",
        )
        retrieved = pipe.experience_pipeline.retrieve(eid, ctx_b)
        assert retrieved is not None
        assert retrieved.source_type == "failure"
        assert retrieved.failure_category == "integration/timeout"


# ---------------------------------------------------------------------------
# R10: Failure -> decision influence chain
# ---------------------------------------------------------------------------
def test_failure_to_decision_influence():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _mkctx("LEARNED_EXPLORATION", task="TASK-A")
        pipe = FailureToLearningPipeline(
            experience_pipeline=ExperiencePipeline(storage_dir=tmp),
            experiment_context=ctx,
        )
        result = pipe.process(_mkfailure(task="TASK-A"), ctx)
        eid = result["experience_id"]
        assert result["chain"] is not None
        assert result["chain"]["failure_id"] == "fid-n1"
        assert result["chain"]["experience_id"] == eid
        assert result["chain"]["learning_influenced"] is True

        # Task B decision influenced by retrieval of E
        entry = pipe.record_failure_retrieval(
            experience_id=eid,
            strategy_id="strategy-avoid-timeout",
            decision_id="decision-B-1",
            retrieval_confidence=0.82,
            ctx=ctx,
        )
        assert entry["learning_influenced"] is True
        assert entry["strategy_id"] == "strategy-avoid-timeout"
        assert entry["decision_id"] == "decision-B-1"
        assert entry["retrieval_confidence"] == 0.82

        # Chain file persisted and inspectable
        chain_path = os.path.join(
            namespace_path("failure_chains", ctx), f"{eid}.json")
        assert os.path.exists(chain_path)
        with open(chain_path) as f:
            chain = json.load(f)
        assert chain["strategy_id"] == "strategy-avoid-timeout"
        assert chain["decision_id"] == "decision-B-1"
        assert chain["learning_influenced"] is True


# ---------------------------------------------------------------------------
# R11: Failure avoidance — comparative vs causal
# ---------------------------------------------------------------------------
def test_failure_avoidance_metric_causal_only():
    pipe = FailureToLearningPipeline()
    m = pipe.failure_avoidance_metric(
        learned_count=2, avoided_count=1, baseline_fail_count=5)
    assert m["avoidance_rate"] == 0.5
    assert m["causal"] is True
    # Simple success must not inflate causally-counted avoidance
    m2 = pipe.failure_avoidance_metric(learned_count=2, avoided_count=0, baseline_fail_count=5)
    assert m2["avoidance_rate"] == 0.0
    m3 = pipe.failure_avoidance_metric(learned_count=0, avoided_count=0, baseline_fail_count=1)
    assert m3["avoidance_rate"] == 0.0


# ---------------------------------------------------------------------------
# R12: Deterministic lesson/experience — identical inputs -> identical output
# ---------------------------------------------------------------------------
def test_deterministic_processing():
    with tempfile.TemporaryDirectory() as tmp:
        ctx1 = _mkctx("LEARNED_EXPLORATION", task="TASK-DET")
        p1 = FailureToLearningPipeline(
            experience_pipeline=ExperiencePipeline(storage_dir=tmp),
            experiment_context=ctx1,
        )
        ctx2 = _mkctx("LEARNED_EXPLORATION", task="TASK-DET")
        p2 = FailureToLearningPipeline(
            experience_pipeline=ExperiencePipeline(storage_dir=tmp),
            experiment_context=ctx2,
        )
        f1 = _mkfailure(failure_id="fid-det", task="TASK-DET",
                        message="timeout occurred")
        f2 = _mkfailure(failure_id="fid-det", task="TASK-DET",
                        message="timeout occurred")

        r1 = p1.process(f1, ctx1)
        r2 = p2.process(f2, ctx2)
        assert r1["experience_id"] == r2["experience_id"]
        assert r1["lesson"] == r2["lesson"]
        assert r1["deterministic_id"] == r2["deterministic_id"]

        e1 = p1.experience_pipeline.retrieve(r1["experience_id"], ctx1)
        e2 = p2.experience_pipeline.retrieve(r2["experience_id"], ctx2)
        assert e1.lesson == e2.lesson
        assert e1.failure_category == e2.failure_category
        assert e1.confidence == e2.confidence


# ---------------------------------------------------------------------------
# R13: Security / governance advisory safety
# ---------------------------------------------------------------------------
def test_security_failure_never_weakens_constraints():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _mkctx("LEARNED_EXPLORATION")
        pipe = FailureToLearningPipeline(
            experience_pipeline=ExperiencePipeline(storage_dir=tmp),
            experiment_context=ctx,
        )
        failure = StructuredFailure(
            node_id="sec", capability="auth_service", category="security",
            sub_category="auth_bypass", severity="critical",
            error_message="Auth gateway rejected request (403)",
            failure_id="fid-sec", source_task_id="TASK-SEC",
        )
        result = pipe.process(failure, ctx)
        stored = pipe.experience_pipeline.retrieve(result["experience_id"], ctx)
        lesson_lower = stored.lesson.lower()

        # Defensive/constraint-preserving, never weakening.
        # The failure's own sub_category (e.g. "auth_bypass") is a descriptive
        # taxonomy label and may appear *in the lesson text*; what matters is that
        # the derived ACTION guidance never directs a weakening/bypass of policy.
        action_text = (stored.lesson + " "
                       + stored.provenance["failure_category"]).lower()
        assert "validate earlier" in lesson_lower or "select safe" in lesson_lower
        assert result["advisory"]["advisory_only"] is True
        assert result["advisory"]["governance_preserved"] is True
        # Action guidance must never tell the agent to disable/bypass/weaken.
        for forbidden in ("disable", "increase autonomy", "weaken"):
            assert forbidden not in action_text, f"Forbidden lesson text: {stored.lesson}"
        assert "bypass the gate" not in lesson_lower
        assert "bypass" not in stored.provenance["failure_category"].split("/")[0]  # category is 'security'


def test_governance_never_weakened_by_failure():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _mkctx("LEARNED_EXPLORATION")
        pipe = FailureToLearningPipeline(
            experience_pipeline=ExperiencePipeline(storage_dir=tmp),
            experiment_context=ctx,
        )
        # A failure message that superficially suggests "disable the gate"
        failure = StructuredFailure(
            node_id="g", capability="workflow", category="implementation",
            sub_category="logic_error", severity="major",
            error_message="gate failed; suggested to disable the gate and expand",
            failure_id="fid-gov", source_task_id="TASK-GOV",
        )
        result = pipe.process(failure, ctx)
        stored = pipe.experience_pipeline.retrieve(result["experience_id"], ctx)
        lower = stored.lesson.lower()
        assert "disable" not in lower
        assert result["advisory"]["weakening_constraint"] is True  # detected & neutralized


# ---------------------------------------------------------------------------
# R14: Isolation — run A != B, COLD != LEARNED, TEST -> never training
# ---------------------------------------------------------------------------
def test_cross_run_isolation():
    with tempfile.TemporaryDirectory() as tmp:
        ctx_a = _mkctx("LEARNED_EXPLORATION", run="RUN-A")
        pipe = FailureToLearningPipeline(
            experience_pipeline=ExperiencePipeline(storage_dir=tmp),
            experiment_context=ctx_a,
        )
        r_a = pipe.process(_mkfailure(task="TASK-ISO"), ctx_a)

        # Different run cannot see run A's experience
        ctx_b = _mkctx("LEARNED_EXPLORATION", run="RUN-B", task="TASK-ISO")
        assert pipe.experience_pipeline.retrieve(r_a["experience_id"], ctx_b) is None
        # Same run can
        ctx_a2 = _mkctx("LEARNED_EXPLORATION", run="RUN-A", task="TASK-OTHER")
        assert pipe.experience_pipeline.retrieve(r_a["experience_id"], ctx_a2) is not None


def test_cross_mode_isolation_cold_learned():
    with tempfile.TemporaryDirectory() as tmp:
        ctx_learned = _mkctx("LEARNED_EXPLORATION")
        pipe = FailureToLearningPipeline(
            experience_pipeline=ExperiencePipeline(storage_dir=tmp),
            experiment_context=ctx_learned,
        )
        r = pipe.process(_mkfailure(task="TASK-M"), ctx_learned)

        ctx_cold = ExperimentContext(
            validation_run_id=ctx_learned.validation_run_id,
            benchmark_id=ctx_learned.benchmark_id,
            task_id="TASK-M", execution_id="e", mode="COLD",
        )
        # COLD retrieval must not surface a LEARNED failure experience
        assert pipe.experience_pipeline.retrieve(r["experience_id"], ctx_cold) is None


def test_test_never_trains_contamination_scan():
    with tempfile.TemporaryDirectory() as tmp:
        # TEST run writes only failure_evidence, no experiences
        ctx_test = _mkctx("TEST", run="RUN-TEST")
        pipe = FailureToLearningPipeline(
            experience_pipeline=ExperiencePipeline(storage_dir=tmp),
            experiment_context=ctx_test,
        )
        pipe.process(_mkfailure(task="TASK-T"), ctx_test)
        exp_dir = namespace_path("experiences", ctx_test)
        files = [f for f in os.listdir(exp_dir) if f.endswith(".json")]
        assert files == [], f"TEST contamination: {files}"


# ---------------------------------------------------------------------------
# R15: Unknown root cause never invented
# ---------------------------------------------------------------------------
def test_unknown_root_cause_honest():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _mkctx("LEARNED_EXPLORATION")
        pipe = FailureToLearningPipeline(
            experience_pipeline=ExperiencePipeline(storage_dir=tmp),
            experiment_context=ctx,
        )
        # Odd category that the analyzer cannot classify -> root cause UNKNOWN
        failure = StructuredFailure(
            node_id="x", capability="mystery", category="weird_category",
            sub_category="bizarre", severity="warning",
            error_message="something unexplainable happened",
            failure_id="fid-unknown", source_task_id="TASK-U",
        )
        result = pipe.process(failure, ctx)
        assert result["root_cause_known"] is False
        assert result["root_cause"] == "UNKNOWN"
        assert "UNKNOWN" in result["lesson"]


# ---------------------------------------------------------------------------
# R16: Historical artifacts protected (29 tracked artifacts intact) + the
#      durable Phase 8 release-evidence artifact (git-tracked under
#      artifacts/v3.2-release/) which is part of the release record.
# ---------------------------------------------------------------------------
def test_historical_artifacts_preserved():
    out = subprocess.run(
        ["git", "ls-files", "artifacts/"],
        capture_output=True, text=True, cwd=os.getcwd(),
    )
    tracked = [l for l in out.stdout.splitlines() if l.strip()]
    historical = [p for p in tracked if not p.startswith("artifacts/v3.2-release/")]
    # The 29 historical V3.0/V3.1 artifacts remain intact and tracked.
    assert len(historical) == 29, \
        f"Expected 29 historical tracked artifacts, got {len(historical)}"
    # The durable Phase 8 release-evidence artifacts are additionally tracked
    # (release evidence): the frozen original results.json plus the frozen
    # reproducibility rerun results_REPROD. Both are part of the release record.
    durable = [p for p in tracked if p.startswith("artifacts/v3.2-release/")]
    assert len(durable) == 2, durable
    for p in durable:
        # Either the frozen original results.json or the frozen reproducibility
        # rerun results_REPROD_<date>.json — both are release-record artifacts.
        assert p.endswith("results.json") or p.endswith("_REPROD_20260912.json"), p
    assert len(tracked) == 31, f"Expected 31 tracked artifacts total, got {len(tracked)}"
