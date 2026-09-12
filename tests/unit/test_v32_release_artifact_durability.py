# tests/unit/test_v32_release_artifact_durability.py
"""Cleanup-regression tests for the durable Phase 8 release artifact.

Phase 8 must persist the final machine-readable result as durable release
evidence. The autouse test cleanup removes ephemeral v3.2 scratch but must
preserve:

  1. temporary test fixtures       -> may be cleaned,
  2. the accepted final validation results.json (artifacts/v3.2-release/) ->
     survives cleanup,
  3. historical V3.0/V3.1 tracked artifacts (29) -> remain protected.

These tests exercise the exact cleanup contract (clean_v32_scratch from
tests.conftest) plus the deterministic generation path in
harness.validation.field_validation. They do NOT re-run the statistical
experiment.
"""
import json
import os
import shutil
import subprocess

import pytest

from tests.conftest import clean_v32_scratch, _V32_SCRATCH, _REPO_ROOT
from harness.validation import field_validation as fv

PROTECTED_RESULTS = os.path.join(
    _REPO_ROOT, "artifacts", "v3.2-release", "field_validation", "results.json"
)


def _git_tracked_artifact_paths():
    out = subprocess.check_output(
        ["git", "ls-files", "artifacts/"], cwd=_REPO_ROOT,
        stderr=subprocess.DEVNULL).decode().splitlines()
    return sorted(p for p in out if p.strip())


# ---------------------------------------------------------------------------
# 1. Temporary fixtures may be cleaned
# ---------------------------------------------------------------------------
def test_temporary_v32_scratch_is_cleaned():
    scratch_run = os.path.join(_V32_SCRATCH, "V3.2-FV8-UNIT",
                               "field_validation", "observations", "cold", "test")
    os.makedirs(scratch_run, exist_ok=True)
    scratch_file = os.path.join(scratch_run, "task1__r1.json")
    with open(scratch_file, "w") as f:
        json.dump({"scratch": True}, f)
    assert os.path.exists(scratch_file)

    clean_v32_scratch()

    # Scratch is ephemeral: cleaned away entirely.
    assert not os.path.exists(scratch_file)
    assert not os.path.exists(_V32_SCRATCH)


# ---------------------------------------------------------------------------
# 2. Accepted final validation artifact survives cleanup
# ---------------------------------------------------------------------------
def test_final_validation_results_survives_cleanup():
    # The durable release artifact may not pre-exist; generate it first.
    if not os.path.exists(PROTECTED_RESULTS):
        fv.persist_final_validation_results()
    assert os.path.exists(PROTECTED_RESULTS)

    # Simulate scratch being present (as after a real run), then clean.
    scratch_file = os.path.join(_V32_SCRATCH, "V3.2-FV8-20260911",
                                "field_validation", "observations", "cold", "test", "t.json")
    os.makedirs(os.path.dirname(scratch_file), exist_ok=True)
    with open(scratch_file, "w") as f:
        json.dump({"scratch": True}, f)

    clean_v32_scratch()

    # Scratch gone, durable release evidence preserved.
    assert not os.path.exists(scratch_file)
    assert os.path.exists(PROTECTED_RESULTS)


def test_release_evidence_root_never_inside_scratch():
    # The protection depends on the release root living outside the scratch
    # tree. Enforce the invariant so the contract cannot silently break.
    release_root = os.path.join(_REPO_ROOT, "artifacts", "v3.2-release")
    assert not release_root.startswith(_V32_SCRATCH + os.sep)


# ---------------------------------------------------------------------------
# 3. Historical V3.0/V3.1 tracked artifacts remain protected
# ---------------------------------------------------------------------------
def test_historical_tracked_artifacts_preserved_and_count():
    # Historical V3.0/V3.1 tracked artifacts (EXCLUDING the new durable
    # v3.2-release release-evidence artifact, which is a separate tracked file).
    before_all = _git_tracked_artifact_paths()
    historical = [p for p in before_all
                  if not p.startswith("artifacts/v3.2-release/")]
    # Fixed historical expectation (29) from the V3.2 release record.
    assert len(historical) == 29, \
        f"expected 29 historical tracked artifacts, got {len(historical)}"

    # Run the cleanup as the suite does.
    clean_v32_scratch()

    after_all = _git_tracked_artifact_paths()
    # Neither the historical set nor the durable release artifact changed.
    assert before_all == after_all, \
        "cleanup altered the tracked artifact set"
    # And every historical tracked artifact still exists on disk.
    for rel in historical:
        assert os.path.exists(os.path.join(_REPO_ROOT, rel)), rel
    # Durable release artifact also still present on disk.
    assert os.path.exists(PROTECTED_RESULTS)


# ---------------------------------------------------------------------------
# 4. Durable results.json consistency with the accepted/validated report
# ---------------------------------------------------------------------------
def test_durable_results_consistent_with_accepted_figures():
    if not os.path.exists(PROTECTED_RESULTS):
        fv.persist_final_validation_results()
    with open(PROTECTED_RESULTS) as f:
        r = json.load(f)

    exp = r["experiment"]
    for key in ("validation_run_id", "benchmark_id", "configuration_digest",
                "dataset_digest", "started_at", "completed_at"):
        assert key in exp and exp[key]
    assert exp["validation_run_id"] == fv.ACCEPTED_VALIDATION_RUN == "V3.2-FV8-20260911"

    modes = r["modes"]
    assert modes["COLD"]["success_rate"] == 0.360
    assert modes["LEARNED_NO_EXPLORATION"]["success_rate"] == 0.720
    assert modes["LEARNED_EXPLORATION"]["success_rate"] == 0.780
    assert modes["COLD"]["engineering_score"] == 0.4573
    assert modes["LEARNED_NO_EXPLORATION"]["engineering_score"] == 0.589
    assert modes["LEARNED_EXPLORATION"]["engineering_score"] == 0.626

    c = r["comparisons"]
    assert c["cold_vs_learned_no_exploration"]["success_rate"]["delta_pp"] == 36.0
    assert c["cold_vs_learned_exploration"]["success_rate"]["delta_pp"] == 42.0
    assert c["no_exploration_vs_exploration"]["success_rate"]["delta_pp"] == 6.0

    gen = r["generalization"]["success_rate_delta_pp"]
    assert gen == {"train": 36.0, "validation": 48.0, "test": 42.0}

    attr = r["attribution"]
    assert attr["decision_improvement"]["improved"] == 63
    assert attr["decision_improvement"]["n_influenced"] == 94
    assert attr["learning_harm"]["worsened"] == 2
    assert attr["learning_influence"]["influenced"] == 185
    assert attr["learning_influence"]["total"] == 200

    sec = r["security"]
    assert (sec["security_violations"], sec["governance_violations"],
            sec["non_free_model_executions"]) == (0, 0, 0)

    assert r["integrity"]["critical_contamination"] is False
    assert r["release_gates"]["passed"] == 26 and r["release_gates"]["total"] == 26
    assert r["release_gates"]["all_pass"] is True
    assert r["verdict"] == "A_VALIDATED"
    assert r["release_recommendation"] == "GO"
    # all required top-level sections present
    for section in ("experiment", "modes", "comparisons", "generalization",
                    "failure_learning", "attribution", "exploration", "security",
                    "governance", "integrity", "release_gates", "verdict",
                    "release_recommendation"):
        assert section in r, section


def test_durable_generation_is_deterministic():
    r1 = fv.accepted_final_validation_results()
    r2 = fv.accepted_final_validation_results()
    assert r1 == r2
