# harness/learning/isolation.py — V3.2 Artifact Isolation Utilities (Phase 2)
"""
Storage namespace isolation utilities for V3.2.

Provides:
- Namespaced path construction per artifacts/v3.2/<run_id>/<mode>/<store>/
- Isolation verification across modes and runs
- Retrieval filtering helpers
- TEST protection guards
"""

import os
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .context import ExperimentContext, VALID_MODES


V3_2_BASE = "v3.2"
ARTIFACTS_BASE = "artifacts"


def get_v32_base() -> str:
    """Get the artifacts/v3.2 base path relative to the project root."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ARTIFACTS_BASE,
        V3_2_BASE,
    )


def namespace_path(
    store_type: str,
    experiment_context: Optional[ExperimentContext] = None,
    *,
    validation_run_id: str = "",
    mode: str = "",
    base_path: str = "",
) -> str:
    """Build a namespaced storage path.

    Format: <base>/<validation_run_id>/<mode>/<store_type>/

    When ExperimentContext is provided, it overrides manual run_id/mode.
    Missing context with no fallback raises ValueError.

    Args:
        store_type: Type of store (experiences, strategies, decisions, etc.)
        experiment_context: Optional ExperimentContext for namespacing
        validation_run_id: Fallback if no context provided
        mode: Fallback if no context provided
        base_path: Override base path (default: artifacts/v3.2)
    """
    run_id = validation_run_id
    md = mode

    if experiment_context is not None:
        run_id = experiment_context.validation_run_id
        md = experiment_context.mode
    elif not run_id or not md:
        raise ValueError(
            "ExperimentContext required for namespaced storage. "
            f"Got run_id={run_id!r}, mode={md!r}"
        )

    if md not in VALID_MODES:
        raise ValueError(f"Invalid mode '{md}'. Must be one of: {', '.join(sorted(VALID_MODES))}")

    base = base_path or get_v32_base()
    path = os.path.join(base, run_id, md, store_type)
    os.makedirs(path, exist_ok=True)
    return path


def find_namespaced_dirs(
    store_type: str,
    *,
    validation_run_id: str = "",
    mode: str = "",
) -> List[str]:
    """Find existing namespaced directories matching optional filters.

    If validation_run_id and mode are both empty, returns all v3.2 paths.
    If only validation_run_id, returns all modes for that run.
    If both provided, returns the exact path.
    """
    base = get_v32_base()
    if not os.path.exists(base):
        return []

    if validation_run_id and mode:
        path = os.path.join(base, validation_run_id, mode, store_type)
        return [path] if os.path.exists(path) else []

    if validation_run_id:
        run_dir = os.path.join(base, validation_run_id)
        if not os.path.exists(run_dir):
            return []
        results = []
        for m in os.listdir(run_dir):
            path = os.path.join(run_dir, m, store_type)
            if os.path.exists(path):
                results.append(path)
        return results

    # Walk all runs and modes
    results = []
    if not os.path.exists(base):
        return results
    for run_id in os.listdir(base):
        run_dir = os.path.join(base, run_id)
        if not os.path.isdir(run_dir):
            continue
        for m in os.listdir(run_dir):
            if m not in VALID_MODES:
                continue
            path = os.path.join(run_dir, m, store_type)
            if os.path.exists(path):
                results.append(path)
    return results


def get_store_type_path(store_type: str) -> str:
    """Get the full path for a store type within the v3.2 namespace."""
    return os.path.join(get_v32_base(), store_type)


def verify_isolation(base_path: str = "", run_id: str = "") -> Dict[str, Any]:
    """Verify isolation across modes and runs.

    Scans the v3.2 artifact tree and checks for contamination.

    Args:
        base_path: Override base path (default: artifacts/v3.2)
        run_id: Optional specific run to check

    Returns:
        Dict with isolation verification results
    """
    if not base_path:
        base_path = get_v32_base()

    if not os.path.exists(base_path):
        return {
            "passed": True,
            "contamination_count": 0,
            "checks": {
                "cold_learned": 0,
                "learned_cold": 0,
                "test_training": 0,
                "cross_run": 0,
                "unattributed": 0,
            },
        }

    contamination_count = 0
    cold_learned = 0
    learned_cold = 0
    test_training = 0
    cross_run = 0
    unattributed = 0

    # Scan all run dirs
    run_ids_to_check = [run_id] if run_id else [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]

    seen_files: Dict[str, Set[str]] = {}  # content_hash -> set of (run_id, mode)

    for rid in run_ids_to_check:
        run_dir = os.path.join(base_path, rid)
        if not os.path.isdir(run_dir):
            continue
        for mode in os.listdir(run_dir):
            if mode not in VALID_MODES:
                continue
            mode_dir = os.path.join(run_dir, mode)
            if not os.path.isdir(mode_dir):
                continue
            for store_type in os.listdir(mode_dir):
                store_dir = os.path.join(mode_dir, store_type)
                if not os.path.isdir(store_dir):
                    continue
                for fname in os.listdir(store_dir):
                    if not fname.endswith(".json"):
                        continue
                    fpath = os.path.join(store_dir, fname)
                    # Check cross-run: same filename in different run = potential contamination
                    key = f"{store_type}/{fname}"
                    if key not in seen_files:
                        seen_files[key] = set()
                    seen_files.setdefault(key, set()).add(f"{rid}:{mode}")

    # Analyze cross-contamination
    for key, locations in seen_files.items():
        if len(locations) > 1:
            modes_present = {loc.split(":")[1] for loc in locations}
            runs_present = {loc.split(":")[0] for loc in locations}

            if len(runs_present) > 1:
                cross_run += 1

            if "COLD" in modes_present and "LEARNED" in modes_present:
                cold_learned += 1
                learned_cold += 1
            if "TEST" in modes_present and "COLD" in modes_present:
                test_training += 1
            if "TEST" in modes_present and "LEARNED" in modes_present:
                test_training += 1

    contamination_count = (
        max(cold_learned, learned_cold)
        + test_training
        + cross_run
        + unattributed
    )

    # Check unattributed artifacts
    for rid in run_ids_to_check:
        run_dir = os.path.join(base_path, rid)
        if not os.path.isdir(run_dir):
            continue
        for mode in os.listdir(run_dir):
            if mode not in VALID_MODES:
                unattributed += 1  # directory not in valid modes

    return {
        "passed": contamination_count == 0,
        "contamination_count": contamination_count,
        "checks": {
            "cold_learned": cold_learned,
            "learned_cold": learned_cold,
            "test_training": test_training,
            "cross_run": cross_run,
            "unattributed": unattributed,
        },
    }


def retrieve_filtered(
    items: List[Any],
    experiment_context: Optional[ExperimentContext] = None,
) -> List[Any]:
    """Filter a list of items by ExperimentContext mode and validation_run_id.

    When ExperimentContext is None, returns all items (backward compatible).
    """
    if experiment_context is None:
        return items

    filtered = []
    for item in items:
        item_mode = getattr(item, "mode", "")
        item_run_id = getattr(item, "validation_run_id", "")

        if item_mode and item_mode != experiment_context.mode:
            continue
        if item_run_id and item_run_id != experiment_context.validation_run_id:
            continue
        filtered.append(item)

    return filtered


def check_test_protection(
    experiment_context: Optional[ExperimentContext],
    operation: str = "store",
) -> None:
    """Guard against TEST-mode artifacts entering training/learning stores.

    Raises ValueError if TEST mode is used in a forbidden operation.
    The 'store' operation is always allowed (TEST artifacts can be stored).
    The 'learn' or 'train' operation is blocked for TEST mode.
    """
    if experiment_context is None:
        return

    if experiment_context.mode == "TEST" and operation in ("learn", "train"):
        raise ValueError(
            f"TEST-mode artifacts cannot enter training/learning stores. "
            f"Attempted {operation} in TEST mode."
        )
