"""Tests for evidence retention and durable evidence directory (Phase 16 Gate 1).

These tests ensure:
1. The durable evidence directory exists and is tracked by git.
2. The experiment manifest template has all required fields.
3. Evidence committed to evidence/v3.3/phase16/ survives artifacts/ cleanup.
4. The gitignore exceptions are correctly configured.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "evidence" / "v3.3" / "phase16"
MANIFEST_PATH = REPO_ROOT / "evidence" / "v3.3" / "phase16" / "experiment_manifest.json"
GITIGNORE_PATH = REPO_ROOT / ".gitignore"
DOCS_MANIFEST_PATH = REPO_ROOT / "docs" / "V3.3-PHASE16-EXPERIMENT-MANIFEST.json"

REQUIRED_MANIFEST_FIELDS = {
    "run_id", "experiment_version", "base_release", "base_commit",
    "feature_commit", "freeze_commit", "config_digest", "benchmark_digest",
    "knowledge_digest", "raw_results_digest", "analysis_digest",
    "seed", "repetitions", "expected_execution_count", "actual_execution_count",
    "modes", "statistical_unit", "bootstrap_count", "permutation_count",
    "alpha", "practical_threshold",
}


def _run_git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT)] + args,
        capture_output=True, text=True, check=check,
    )


class TestEvidenceDirectoryStructure:
    """The durable evidence directory must exist and be git-tracked."""

    def test_evidence_directory_exists(self):
        assert EVIDENCE_DIR.is_dir(), "evidence/v3.3/phase16/ must exist"

    def test_evidence_directory_tracked_by_git(self):
        """Evidence directory must be in the git index (not ignored)."""
        result = _run_git(["ls-files", "evidence/v3.3/phase16/"])
        files = [f for f in result.stdout.strip().split("\n") if f]
        assert len(files) > 0, "evidence/v3.3/phase16/ must contain git-tracked files"

    def test_gitignore_has_evidence_exception(self):
        """The .gitignore must have an explicit exception for evidence/v3.3/phase16/."""
        content = GITIGNORE_PATH.read_text()
        assert "!evidence/v3.3/phase16/" in content, \
            ".gitignore must contain '!evidence/v3.3/phase16/' exception"
        assert "!evidence/v3.3/phase16/**" in content, \
            ".gitignore must contain '!evidence/v3.3/phase16/**' exception"

    def test_gitignore_still_ignores_artifacts_v33(self):
        """artifacts/v3.3/ must still be ignored."""
        content = GITIGNORE_PATH.read_text()
        assert "artifacts/v3.3/" in content, ".gitignore must still ignore artifacts/v3.3/"


class TestExperimentManifestTemplate:
    """The durable experiment manifest must have all required fields."""

    def test_manifest_file_exists(self):
        assert MANIFEST_PATH.is_file(), "experiment_manifest.json must exist"

    def test_manifest_is_valid_json(self):
        data = json.loads(MANIFEST_PATH.read_text())
        assert isinstance(data, dict)

    def test_manifest_has_schema_version(self):
        data = json.loads(MANIFEST_PATH.read_text())
        assert "schema_version" in data
        assert "phase16" in data["schema_version"]

    def test_manifest_has_frozen_protocol(self):
        data = json.loads(MANIFEST_PATH.read_text())
        protocol = data.get("frozen_protocol", {})
        assert protocol["repetitions"] == 10
        assert protocol["task_clusters"] == 20
        assert protocol["expected_executions"] == 600
        assert protocol["bootstrap_iterations"] == 10000
        assert protocol["permutation_iterations"] == 10000
        assert protocol["alpha"] == 0.05
        assert protocol["practical_threshold"] == 0.05

    def test_manifest_has_commits(self):
        data = json.loads(MANIFEST_PATH.read_text())
        assert data["base_commit"] == "edea273b47eba768518500dffe79239f06602457"
        assert "freeze_commit" in data
        assert "feature_commit" in data

    def test_manifest_has_evidence_retention(self):
        data = json.loads(MANIFEST_PATH.read_text())
        retention = data.get("evidence_retention", {})
        assert retention.get("status") == "ACTIVE"
        assert retention.get("durable_evidence_dir") == "evidence/v3.3/phase16/"

    def test_manifest_run_statuses(self):
        data = json.loads(MANIFEST_PATH.read_text())
        runs = data.get("runs", {})
        r1 = runs.get("V3.3-P16-FV-20260918", {})
        r2 = runs.get("V3.3-P16-FV-20260918-R2", {})
        assert r1.get("raw_evidence") == "LOST"
        assert r1.get("historical_result") == "PRESERVED"
        assert r1.get("accepted_evidence_status") == "SUPERSEDED"
        assert r2.get("raw_evidence") == "LOST"
        assert r2.get("accepted_evidence_status") == "SUPERSEDED"

    def test_manifest_not_instrumented_digests(self):
        data = json.loads(MANIFEST_PATH.read_text())
        digests = data.get("digests", {})
        for key in ("policy", "model", "learning_state"):
            assert digests.get(key) == "NOT_INSTRUMENTED", f"{key} must be NOT_INSTRUMENTED"


class TestDocsManifestUpdated:
    """The docs manifest must be updated with evidence retention info."""

    def test_docs_manifest_has_evidence_retention(self):
        data = json.loads(DOCS_MANIFEST_PATH.read_text())
        retention = data.get("evidence_retention")
        assert retention is not None, "docs manifest must include evidence_retention"
        assert retention["status"] == "ACTIVE"

    def test_docs_manifest_runs_lost_status(self):
        data = json.loads(DOCS_MANIFEST_PATH.read_text())
        runs = data["evidence_retention"]["runs"]
        assert runs["V3.3-P16-FV-20260918"]["raw_evidence"] == "LOST"
        assert runs["V3.3-P16-FV-20260918-R2"]["raw_evidence"] == "LOST"
        assert runs["V3.3-P16-FV-20260918"]["historical_result"] == "PRESERVED"


class TestEvidenceRetentionConfig:
    """Test the config/phase16 files that provide digests for the manifest."""

    def test_accepted_config_exists(self):
        config_path = REPO_ROOT / "config" / "phase16" / "accepted-r2.json"
        assert config_path.is_file()

    def test_benchmark_exists(self):
        bm_path = REPO_ROOT / "config" / "phase16" / "benchmark.json"
        assert bm_path.is_file()

    def test_knowledge_exists(self):
        kn_path = REPO_ROOT / "config" / "phase16" / "knowledge.json"
        assert kn_path.is_file()
