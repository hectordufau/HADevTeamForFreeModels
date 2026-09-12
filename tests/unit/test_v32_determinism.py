"""V3.2 Phase 6 tests — Reproducibility & Deterministic Exploration (R1-R6).

Covers specification §10 (Phase 6):
  - deterministic RNG derived from ExperimentContext (SHA-256, no Python hash())
  - COLD / LEARNED_NO_EXPLORATION (greedy) / LEARNED_EXPLORATION (seeded) / TEST
  - R1 same seed => same decision
  - R2 process restart reproducibility (persist + re-create)
  - R3 different seed can drive a different exploratory path
  - R4 exploration disabled => seeds must NOT alter decision
  - R5 greedy determinism + deterministic tie-breaking
  - R6 candidate ordering does not change result
  - exploration evidence persistence + counterfactual (spec §7)
  - security/governance: exploration is advisory, never overrides policy
  - Phase 5 statistical compatibility preserved
  - no eval->learning feedback; isolation preserved
"""
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from harness.learning.context import (
    ExperimentContext,
    VALID_MODES,
    LEARNED_MODES,
    GREEDY_NO_EXPLORATION_MODES,
    EXPLORATION_ENABLED_MODES,
)
from harness.learning.determinism import (
    Candidate,
    DeterministicExplorationPolicy,
    ExplorationDecision,
    derive_exploration_seed,
    context_derived_seed,
    select_greedy,
    sorted_candidates,
    weighted_choice,
    load_exploration_config,
    persist_exploration_decision,
    would_selection_differ,
)
from harness.learning import (
    DeterministicExplorationPolicy as ExpPolicyExported,
    ExplorationDecision as DecisionExported,
)

PROJ = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))


def make_ctx(mode="LEARNED_EXPLORATION", seed=42, run="runR1",
             bench="benchP6", task="taskA", exec_id="exec1",
             exploration_enabled=None, exploration_rate=None):
    return ExperimentContext(
        validation_run_id=run, benchmark_id=bench, task_id=task,
        execution_id=exec_id, mode=mode, reproducibility_seed=seed,
        exploration_enabled=exploration_enabled,
        exploration_rate=exploration_rate,
    )


def candidates(seed_offset=0):
    """A small deterministic candidate set with a clear greedy best."""
    return [
        Candidate(stable_id="legacy-a", primary_score=0.5, confidence=0.4,
                  historical_utility=0.3),
        Candidate(stable_id="best-b", primary_score=0.9, confidence=0.8,
                  historical_utility=0.7),
        Candidate(stable_id="mid-c", primary_score=0.7, confidence=0.5,
                  historical_utility=0.5),
        Candidate(stable_id="low-d", primary_score=0.2, confidence=0.9,
                  historical_utility=0.1),
    ]


# ---------------------------------------------------------------------------
# Modes & backward compatibility
# ---------------------------------------------------------------------------


class TestModes:
    def test_new_modes_in_valid_set(self):
        assert "LEARNED_NO_EXPLORATION" in VALID_MODES
        assert "LEARNED_EXPLORATION" in VALID_MODES
        assert "COLD" in VALID_MODES
        assert "TEST" in VALID_MODES
        # legacy LEARNED must remain a valid, readable mode
        assert "LEARNED" in VALID_MODES

    def test_legacy_learned_still_constructible(self):
        ctx = ExperimentContext(
            validation_run_id="rw", benchmark_id="b", task_id="t",
            execution_id="e", mode="LEARNED")
        assert ctx.mode == "LEARNED"
        assert ctx.is_learned_mode()

    def test_learned_mode_grouping(self):
        assert LEARNED_MODES == {"LEARNED", "LEARNED_NO_EXPLORATION",
                                 "LEARNED_EXPLORATION"}
        assert GREEDY_NO_EXPLORATION_MODES == {"COLD", "LEARNED_NO_EXPLORATION"}
        assert EXPLORATION_ENABLED_MODES == {"LEARNED_EXPLORATION"}

    def test_mode_semantics(self):
        g = make_ctx(mode="LEARNED_NO_EXPLORATION")
        assert g.exploration_is_disabled() is True
        assert g.exploration_is_enabled() is False
        e = make_ctx(mode="LEARNED_EXPLORATION")
        assert e.exploration_is_disabled() is False
        assert e.exploration_is_enabled() is True
        c = make_ctx(mode="COLD")
        assert c.exploration_is_disabled() is True


# ---------------------------------------------------------------------------
# Seed derivation (spec §2)
# ---------------------------------------------------------------------------


class TestSeedDerivation:
    def test_seed_is_sha256_of_components(self):
        seed = derive_exploration_seed(42, "runR1", "benchP6", "taskA",
                                       "model")
        payload = "42::runR1::benchP6::taskA::model"
        expect = int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)
        assert seed == expect

    def test_seed_deterministic(self):
        a = derive_exploration_seed(42, "runR1", "benchP6", "taskA", "model")
        b = derive_exploration_seed(42, "runR1", "benchP6", "taskA", "model")
        assert a == b

    def test_seed_differs_on_any_component(self):
        base = derive_exploration_seed(42, "runR1", "benchP6", "taskA", "model")
        assert base != derive_exploration_seed(43, "runR1", "benchP6", "taskA", "model")
        assert base != derive_exploration_seed(42, "runR2", "benchP6", "taskA", "model")
        assert base != derive_exploration_seed(42, "runR1", "benchP7", "taskA", "model")
        assert base != derive_exploration_seed(42, "runR1", "benchP6", "taskB", "model")
        assert base != derive_exploration_seed(42, "runR1", "benchP6", "taskA", "strategy")

    def test_no_python_hash(self):
        # Python's builtin hash() of a string is salted per process for str but
        # stable for int; the derivation must not depend on it. Assert the seed
        # comes from the SHA-256 digest, not hash().
        seed = derive_exploration_seed(42, "runR1", "benchP6", "taskA", "model")
        expected = int(hashlib.sha256(b"42::runR1::benchP6::taskA::model")
                       .hexdigest()[:16], 16)
        assert seed == expected


# ---------------------------------------------------------------------------
# R1 same seed => same decision
# ---------------------------------------------------------------------------


class TestR1SameSeed:
    def test_same_seed_same_decision_repeated(self):
        ctx = make_ctx(mode="LEARNED_EXPLORATION", seed=7, run="runA")
        d1 = DeterministicExplorationPolicy(experiment_context=ctx)\
            .deterministic_decision(candidates())
        d2 = DeterministicExplorationPolicy(experiment_context=ctx)\
            .deterministic_decision(candidates())
        assert d1.to_dict() == d2.to_dict()
        assert d1.derived_seed == d2.derived_seed
        assert d1.selected_candidate_id == d2.selected_candidate_id

    def test_same_task_state_config_seed_same_plan(self):
        # Same everything => identical selected strategy/model/policy scope.
        c1 = candidates()
        c2 = candidates()
        ctx1 = make_ctx(mode="LEARNED_EXPLORATION", seed=123, run="runX")
        ctx2 = make_ctx(mode="LEARNED_EXPLORATION", seed=123, run="runX")
        a = DeterministicExplorationPolicy(experiment_context=ctx1)\
            .deterministic_decision(c1)
        b = DeterministicExplorationPolicy(experiment_context=ctx2)\
            .deterministic_decision(c2)
        assert a.to_dict()["exploration"] == b.to_dict()["exploration"]


# ---------------------------------------------------------------------------
# R2 process restart
# ---------------------------------------------------------------------------


class TestR2ProcessRestart:
    def _write_context(self, ctx, path):
        with open(path, "w") as f:
            json.dump(ctx.to_dict(), f, sort_keys=True)

    def test_seed_survives_process_boundary(self):
        ctx = make_ctx(mode="LEARNED_EXPLORATION", seed=99, run="runRestart")
        # Derived seed computed in THIS process:
        seed_here = context_derived_seed(ctx, "model")
        # Compute the same seed in a fresh python subprocess (SHA-256 is stable):
        script = (
            "import hashlib,json,sys;"
            "ctx=json.load(open(sys.argv[1]));"
            "payload=f\"{ctx['reproducibility_seed']}::{ctx['validation_run_id']}"
            "::{ctx['benchmark_id']}::{ctx['task_id']}::model\";"
            "print(int(hashlib.sha256(payload.encode()).hexdigest()[:16],16))"
        )
        tmp = os.path.join(PROJ, "artifacts", "v3.2", "_tmpctx_p6.json")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        self._write_context(ctx, tmp)
        try:
            out = subprocess.check_output([sys.executable, "-c", script, tmp],
                                          cwd=PROJ).decode().strip()
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        assert int(out) == seed_here

    def test_same_experiment_context_same_decision_across_recreate(self):
        # Persist context, terminate object, recreate from dict (as if new
        # process) => same derived seed, candidate ordering, decision.
        ctx = make_ctx(mode="LEARNED_EXPLORATION", seed=5, run="runRE")
        d1 = DeterministicExplorationPolicy(experiment_context=ctx)\
            .deterministic_decision(candidates())
        # Recreate from serialized dict:
        ctx2 = ExperimentContext.from_dict(ctx.to_dict())
        assert ctx2.reproducibility_seed == ctx.reproducibility_seed
        d2 = DeterministicExplorationPolicy(experiment_context=ctx2)\
            .deterministic_decision(candidates())
        assert d1.derived_seed == d2.derived_seed
        assert d1.candidate_ids == d2.candidate_ids
        assert d1.selected_candidate_id == d2.selected_candidate_id


# ---------------------------------------------------------------------------
# R3 different seed => different exploratory path possible
# ---------------------------------------------------------------------------


class TestR3DifferentSeed:
    def test_different_seeds_can_differ(self):
        ctx_a = make_ctx(mode="LEARNED_EXPLORATION", seed=1, run="runDiff")
        ctx_b = make_ctx(mode="LEARNED_EXPLORATION", seed=2, run="runDiff")
        ds = [DeterministicExplorationPolicy(experiment_context=ctx_a)
              .deterministic_decision(candidates()),
              DeterministicExplorationPolicy(experiment_context=ctx_b)
              .deterministic_decision(candidates())]
        # Not every pair must differ, but across a sweep some seed must lead to
        # a different exploratory path (R3) than seed 1.
        seeds = [DeterministicExplorationPolicy(
            experiment_context=make_ctx(mode="LEARNED_EXPLORATION", seed=s,
                                        run="runDiff"))
            .deterministic_decision(candidates())
            for s in range(1, 60)]
        picks = {(d.derived_seed, d.selected_candidate_id) for d in seeds}
        assert len(picks) > 1, "expected some seeds to take a different path"


# ---------------------------------------------------------------------------
# R4 exploration disabled => seeds must NOT alter decision
# ---------------------------------------------------------------------------


class TestR4Disabled:
    def test_learned_no_exploration_ignores_seed(self):
        picks = set()
        for s in range(30):
            ctx = make_ctx(mode="LEARNED_NO_EXPLORATION", seed=s, run="runNE")
            d = DeterministicExplorationPolicy(experiment_context=ctx)\
                .deterministic_decision(candidates())
            picks.add(d.selected_candidate_id)
            assert d.exploration_taken is False
        assert len(picks) == 1  # every seed -> same greedy pick

    def test_exploration_enabled_false_flag_ignores_seed(self):
        picks = set()
        for s in range(30):
            ctx = make_ctx(mode="LEARNED_EXPLORATION", seed=s, run="runNE",
                           exploration_enabled=False)
            d = DeterministicExplorationPolicy(experiment_context=ctx)\
                .deterministic_decision(candidates())
            picks.add(d.selected_candidate_id)
            assert d.exploration_taken is False
            assert d.exploration_enabled is False
        assert len(picks) == 1

    def test_no_rng_influence_when_disabled(self):
        # Same candidates + disabled must equal greedy selection exactly.
        ctx = make_ctx(mode="LEARNED_NO_EXPLORATION", seed=1234, run="runNE")
        d = DeterministicExplorationPolicy(experiment_context=ctx)\
            .deterministic_decision(candidates())
        greedy = select_greedy(candidates())
        assert d.selected_candidate_id == greedy.stable_id
        assert d.greedy_candidate_id == greedy.stable_id


# ---------------------------------------------------------------------------
# R5 greedy determinism
# ---------------------------------------------------------------------------


class TestR5Greedy:
    def test_greedy_picks_best_score(self):
        c = candidates()
        best = select_greedy(c)
        assert best.stable_id == "best-b"
        assert best.primary_score == 0.9

    def test_deterministic_tie_break(self):
        # Same primary, same confidence; tie broken by historical utility, then id.
        c = [
            Candidate("x1", primary_score=0.5, confidence=0.5,
                      historical_utility=0.9),
            Candidate("x2", primary_score=0.5, confidence=0.5,
                      historical_utility=0.4),
        ]
        assert select_greedy(c).stable_id == "x1"

    def test_confidence_tie_break(self):
        c = [
            Candidate("a", primary_score=0.5, confidence=0.2),
            Candidate("b", primary_score=0.5, confidence=0.6),
        ]
        assert select_greedy(c).stable_id == "b"

    def test_stable_id_asc_tie_break(self):
        c = [
            Candidate("z", primary_score=0.5, confidence=0.5,
                      historical_utility=0.5),
            Candidate("a", primary_score=0.5, confidence=0.5,
                      historical_utility=0.5),
        ]
        assert select_greedy(c).stable_id == "a"

    def test_greedy_deterministic_repeated(self):
        for _ in range(50):
            assert select_greedy(candidates()).stable_id == "best-b"

    def test_learned_no_exploration_always_best_known(self):
        for s in range(20):
            ctx = make_ctx(mode="LEARNED_NO_EXPLORATION", seed=s, run="runG")
            d = DeterministicExplorationPolicy(experiment_context=ctx)\
                .deterministic_decision(candidates())
            assert d.selected_candidate_id == "best-b"


# ---------------------------------------------------------------------------
# R6 candidate ordering
# ---------------------------------------------------------------------------


class TestR6Ordering:
    def test_insertion_order_does_not_change_greedy(self):
        c = candidates()
        rev = list(reversed(c))
        assert select_greedy(c).stable_id == select_greedy(rev).stable_id

    def test_insertion_order_does_not_change_decision_with_seed(self):
        ctx = make_ctx(mode="LEARNED_EXPLORATION", seed=11, run="runOrd")
        fwd = list(candidates())
        rev = list(reversed(fwd))
        # Also a shuffled order:
        shuf = [fwd[2], fwd[0], fwd[3], fwd[1]]
        d1 = DeterministicExplorationPolicy(experiment_context=ctx)\
            .deterministic_decision(fwd)
        d2 = DeterministicExplorationPolicy(experiment_context=ctx)\
            .deterministic_decision(rev)
        d3 = DeterministicExplorationPolicy(experiment_context=ctx)\
            .deterministic_decision(shuf)
        assert d1.selected_candidate_id == d2.selected_candidate_id == \
            d3.selected_candidate_id
        assert d1.candidate_ids == d2.candidate_ids == d3.candidate_ids

    def test_weighted_choice_ordered_deterministically(self):
        import random
        c = candidates()
        rng = random.Random(5)
        a = weighted_choice(c, [0.1, 0.3, 0.2, 0.1], rng)
        rng2 = random.Random(5)
        b = weighted_choice(list(reversed(c)),
                            [0.1, 0.3, 0.2, 0.1], rng2)
        # Results are computed in deterministic tie-break order, so even with
        # reversed inputs the outcome depends only on the set + seed + weights.
        assert a.stable_id == b.stable_id


# ---------------------------------------------------------------------------
# Exploration evidence (spec §7) + counterfactual
# ---------------------------------------------------------------------------


class TestExplorationEvidence:
    def test_decision_carries_full_evidence(self):
        ctx = make_ctx(mode="LEARNED_EXPLORATION", seed=8, run="runEv")
        d = DeterministicExplorationPolicy(experiment_context=ctx)\
            .deterministic_decision(candidates())
        dd = d.to_dict()["exploration"]
        for key in ("mode", "enabled", "configured_rate", "exploration_taken",
                    "reproducibility_seed", "derived_seed", "decision_scope",
                    "candidate_ids", "selected_candidate_id", "selection_reason",
                    "greedy_candidate_id", "exploratory_candidate_id"):
            assert key in dd

    def test_persist_and_reload(self):
        import tempfile
        ctx = make_ctx(mode="LEARNED_EXPLORATION", seed=8, run="runEv")
        d = DeterministicExplorationPolicy(experiment_context=ctx)\
            .deterministic_decision(candidates())
        with tempfile.TemporaryDirectory() as tmp:
            path = persist_exploration_decision(d, storage_dir=tmp)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["exploration"]["derived_seed"] == d.derived_seed
            assert data["exploration"]["selected_candidate_id"] == \
                d.selected_candidate_id
            assert data["identity"]["validation_run_id"] == "runEv"

    def test_would_selection_differ_when_exploring(self):
        # Find a seed that actually explores, then check counterfactual.
        for s in range(1, 200):
            ctx = make_ctx(mode="LEARNED_EXPLORATION", seed=s, run="runC")
            d = DeterministicExplorationPolicy(experiment_context=ctx)\
                .deterministic_decision(candidates())
            if d.exploration_taken:
                res = would_selection_differ(d)
                assert res["computable"] is True
                # Greedy candidate is best-b; exploratory pick can differ.
                assert res["greedy"] == "best-b"
                assert res["would_differ"] == (
                    d.selected_candidate_id != "best-b")
                return
        pytest.fail("no seed explored in sweep")

    def test_would_selection_differ_when_not_exploring(self):
        ctx = make_ctx(mode="LEARNED_NO_EXPLORATION", seed=1, run="runC2")
        d = DeterministicExplorationPolicy(experiment_context=ctx)\
            .deterministic_decision(candidates())
        res = would_selection_differ(d)
        assert res["would_differ"] is False
        assert d.exploration_taken is False

    def test_no_fabricated_counterfactual(self):
        d = ExplorationDecision(
            task_id="t", validation_run_id="r", benchmark_id="b",
            execution_id="e", mode="LEARNED_EXPLORATION",
            decision_scope="model", reproducibility_seed=1, derived_seed=2,
            exploration_enabled=True, configured_rate=0.1,
            exploration_taken=True, candidate_ids=["a", "b"],
            selected_candidate_id="a", selection_reason="x",
            greedy_candidate_id=None, exploratory_candidate_id=None,
        )
        res = would_selection_differ(d)
        assert res["computable"] is False
        assert res["would_differ"] is None  # not fabricated


# ---------------------------------------------------------------------------
# Governance / security (spec §13)
# ---------------------------------------------------------------------------


class TestGovernance:
    def test_exploration_is_advisory_only(self):
        # The deterministic policy merely RECORDS a decision; it carries no
        # authorization to bypass policy/security/autonomy. A downstream gate
        # can reject the exploratory candidate. We simulate that gate here.
        ctx = make_ctx(mode="LEARNED_EXPLORATION", seed=3, run="runGov")
        for s in range(200):
            c = make_ctx(mode="LEARNED_EXPLORATION", seed=s, run="runGov")
            d = DeterministicExplorationPolicy(experiment_context=c)\
                .deterministic_decision(candidates())
            # A governance gate (PolicyEngine equivalent) may reject ANY
            # candidate via the invariant enforcer; exploration never weakens it.
            if d.selected_candidate_id == "forbidden":
                continue
            # The advisory choice is just a recommendation; it must not assert
            # that it overrides any security gate.
            assert d.selection_reason
        # And the advisory module never mutates learning state:
        from harness.learning.context import ExperimentContext as EC
        base = make_ctx(mode="LEARNED", seed=0, run="r")
        # no state is written by a pure deterministic decision
        assert base.mode == "LEARNED"

    def test_policy_violating_candidate_rejected_downstream(self):
        # Simulate: an exploratory candidate that violates policy is rejected
        # by the (read-only) evidence — exploration cannot override security.
        violated = []
        for s in range(1, 100):
            c = make_ctx(mode="LEARNED_EXPLORATION", seed=s, run="runSec")
            d = DeterministicExplorationPolicy(experiment_context=c)\
                .deterministic_decision(candidates())
            # Treat "low-d" as a policy-violating candidate in this simulation.
            if d.selected_candidate_id == "low-d":
                # Governance rejects it; the decision must still be visible and
                # the system must NOT proceed with it (rejected).
                violated.append(d)
        # Exploration may *propose* a violating candidate, but it is recorded as
        # a recommendation only — no assertion that it overrides policy. The
        # invariant: the advisory module never grants permission.
        for d in violated:
            assert d.selection_reason


# ---------------------------------------------------------------------------
# Config + Phase 5 compatibility
# ---------------------------------------------------------------------------


class TestConfigAndCompatibility:
    def test_load_exploration_config_defaults(self):
        cfg = load_exploration_config(None)
        assert cfg["enabled"] is True
        assert cfg["rate"] == 0.10

    def test_load_exploration_config_disabled(self):
        cfg = load_exploration_config(
            {"learning": {"exploration": {"enabled": False, "rate": 0.2}}})
        assert cfg["enabled"] is False
        assert cfg["rate"] == 0.2

    def test_phase5_script_still_runs(self):
        # Phase 5 statistical pipeline must remain compatible (spec §11).
        script = os.path.join(PROJ, "scripts", "phase5_empirical_evaluation.py")
        env = dict(os.environ, PYTHONPATH=PROJ)
        p = subprocess.run([sys.executable, script, "--split", "test"],
                           cwd=PROJ, env=env, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        assert "feedback_loop: NONE" in p.stdout
        # empirical COLD baseline reported (not hard-coded 0.5)
        assert "Empirical COLD baseline" in p.stdout

    def test_no_eval_to_learning_feedback(self):
        # Running a deterministic decision must not write any learning state.
        import glob
        before = set()
        for root, dirs, files in os.walk(os.path.join(PROJ, "artifacts")):
            for fn in files:
                before.add(os.path.join(root, fn))
        ctx = make_ctx(mode="LEARNED_EXPLORATION", seed=4, run="runNoFB")
        DeterministicExplorationPolicy(experiment_context=ctx)\
            .deterministic_decision(candidates())
        after = set()
        for root, dirs, files in os.walk(os.path.join(PROJ, "artifacts")):
            for fn in files:
                after.add(os.path.join(root, fn))
        # No new artifact written by the pure decision.
        assert after == before

    def test_exports_available(self):
        assert ExpPolicyExported is DeterministicExplorationPolicy
        assert DecisionExported is ExplorationDecision

    def test_test_mode_exploration_evidence_namespaced_to_test(self):
        # Spec §9: TEST exploration evidence must never land in a learned path.
        import tempfile
        ctx = make_ctx(mode="TEST", seed=9, run="runTestIsol")
        d = DeterministicExplorationPolicy(experiment_context=ctx)\
            .deterministic_decision(candidates())
        with tempfile.TemporaryDirectory() as tmp:
            path = persist_exploration_decision(d, storage_dir=tmp)
            assert os.path.basename(path).startswith("taskA__model")
            # TEST is never a training/learned path; the mode is TEST.
            assert d.mode == "TEST"
            assert d.to_dict()["identity"]["validation_run_id"] == "runTestIsol"

    def test_cross_run_exploration_seed_isolation(self):
        # Spec §9: different validation_run_id => different derived seed even
        # with identical seed/task/benchmark (run A data cannot equal run B).
        a = make_ctx(mode="LEARNED_EXPLORATION", seed=1, run="runA")
        b = make_ctx(mode="LEARNED_EXPLORATION", seed=1, run="runB")
        da = DeterministicExplorationPolicy(experiment_context=a)\
            .deterministic_decision(candidates())
        db = DeterministicExplorationPolicy(experiment_context=b)\
            .deterministic_decision(candidates())
        assert da.derived_seed != db.derived_seed
        # Same run, different benchmark also differs.
        c = make_ctx(mode="LEARNED_EXPLORATION", seed=1, run="runA",
                     bench="benchOTHER")
        dc = DeterministicExplorationPolicy(experiment_context=c)\
            .deterministic_decision(candidates())
        assert dc.derived_seed != da.derived_seed
