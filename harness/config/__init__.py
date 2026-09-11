# HADevTeamForFreeModels — AI Engineering Harness
"""
Core configuration loading and validation for the V2 Harness.
"""

import os
import yaml
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "config"))

# --- Schema defaults ---

DEFAULT_HARNESS_CONFIG = {
    "version": "2.0",
    "harness": {
        "name": "HADevTeamForFreeModels",
        "mode": "software-engineering",
    },
    "models": {
        "provider": "nous",
        "cost": {"maximum": 0},
    },
    "execution": {
        "max_iterations": 3,
        "require_task_contract": True,
        "require_verification": True,
        "require_evidence": True,
        "require_evaluation": True,
    },
    "autonomy": {"default_level": 2},
    "workspace": {"isolated": True},
    "human_approval": {
        "required_for": ["production", "deployment", "destructive_operations", "security_override"]
    },
}


class ConfigError(Exception):
    """Raised when Harness configuration is invalid."""


def load_config(path: Optional[str] = None) -> dict:
    """Load and validate a Harness configuration YAML file.

    Args:
        path: Path to the YAML config file. Defaults to config/harness.yaml.

    Returns:
        Validated configuration dictionary.

    Raises:
        ConfigError: If the configuration is missing required fields, has
                     invalid types, or violates the cost=0 policy.
    """
    if path is None:
        path = os.path.join(CONFIG_DIR, "harness.yaml")

    if not os.path.exists(path):
        raise ConfigError(f"Configuration file not found: {path}")

    with open(path) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {path}: {e}")

    if not isinstance(config, dict):
        raise ConfigError("Configuration must be a top-level mapping.")

    _validate(config)
    return config


def _validate(config: dict):
    """Validate a loaded configuration dictionary against the schema."""
    errors = []

    # Must have version
    if "version" not in config:
        errors.append("Missing required field: version")

    # Must have harness section
    harness = config.get("harness", {})
    if not isinstance(harness, dict):
        errors.append("'harness' must be a mapping")
    else:
        if "name" not in harness:
            errors.append("Missing harness.name")
        if "mode" not in harness:
            errors.append("Missing harness.mode")

    # Models section
    models = config.get("models", {})
    if not isinstance(models, dict):
        errors.append("'models' must be a mapping")
    else:
        if "provider" not in models:
            errors.append("Missing models.provider")
        cost = models.get("cost", {})
        if isinstance(cost, dict):
            maximum = cost.get("maximum", 0)
            if maximum != 0:
                errors.append(f"models.cost.maximum must be 0 (free only policy), got {maximum}")
        else:
            errors.append("models.cost must be a mapping")

    # Execution section
    execution = config.get("execution", {})
    if not isinstance(execution, dict):
        errors.append("'execution' must be a mapping")

    # Autonomy section
    autonomy = config.get("autonomy", {})
    if not isinstance(autonomy, dict):
        errors.append("'autonomy' must be a mapping")

    # Workspace section
    workspace = config.get("workspace", {})
    if not isinstance(workspace, dict):
        errors.append("'workspace' must be a mapping")

    if errors:
        raise ConfigError("Configuration validation failed:\n  - " + "\n  - ".join(errors))


def merge_config(config: dict, defaults: Optional[dict] = None) -> dict:
    """Merge a loaded config with defaults, filling missing keys."""
    if defaults is None:
        defaults = DEFAULT_HARNESS_CONFIG

    merged = defaults.copy()
    merged.update(config)
    return merged
