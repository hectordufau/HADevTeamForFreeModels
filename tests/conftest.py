"""Pytest configuration for V3.2 tests."""
import pytest
import os
import shutil

# Absolute repo root (two levels up from tests/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ephemeral V3.2 scratch tree: tests and auto-runs may write here; the autouse
# cleanup removes it between tests to prevent cross-contamination.
_V32_SCRATCH = os.path.join(_REPO_ROOT, "artifacts", "v3.2")

# Protected release-evidence root: the durable machine-readable FINAL
# field-validation result (results.json) lives at
# artifacts/v3.2-release/field_validation/results.json (this is the documented
# protected-marker mechanism exposed by harness.validation.field_validation:
# PROTECTED_RELEASE_ROOT == "artifacts/v3.2-release/field_validation"). The
# git-tracked historical V3.0/V3.1 artifacts also live OUTSIDE the scratch
# tree. This cleanup only ever targets _V32_SCRATCH, so the protected release
# evidence and all historical artifacts survive untouched.
_PROTECTED_RELEASE_EVIDENCE = os.path.join(_REPO_ROOT, "artifacts", "v3.2-release")


def clean_v32_scratch():
    """Remove the ephemeral V3.2 scratch tree (fail-closed catch-all).

    PROTECTED: this never touches the protected release-evidence root
    (artifacts/v3.2-release/) nor the git-tracked historical V3.0/V3.1
    artifacts — both live outside the scratch tree. Exposed as a module-level
    function so regression tests can exercise the exact cleanup contract.
    """
    # Guard: the protected release-evidence root must never sit inside the
    # scratch tree, otherwise this cleanup would delete it.
    assert not _PROTECTED_RELEASE_EVIDENCE.startswith(_V32_SCRATCH + os.sep), \
        "Protected release evidence must not live inside the v3.2 scratch tree"
    if os.path.exists(_V32_SCRATCH):
        shutil.rmtree(_V32_SCRATCH, ignore_errors=True)


@pytest.fixture(autouse=True)
def _clean_v32_artifacts():
    """Clean up ephemeral V3.2 artifact scratch space between tests."""
    yield
    clean_v32_scratch()
