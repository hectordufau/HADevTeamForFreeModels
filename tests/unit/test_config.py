# Tests for harness/config
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.config import load_config, ConfigError


def test_load_valid_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "harness.yaml")
    config = load_config(config_path)
    assert config["version"] == "2.0"
    assert config["harness"]["name"] == "HADevTeamForFreeModels"
    assert config["models"]["provider"] == "nous"
    assert config["models"]["cost"]["maximum"] == 0


def test_load_missing_file():
    try:
        load_config("/nonexistent/path.yaml")
        assert False, "Should have raised ConfigError"
    except ConfigError:
        pass


def test_invalid_cost_policy():
    import tempfile
    import yaml
    bad_config = {
        "version": "2.0",
        "harness": {"name": "test", "mode": "test"},
        "models": {"provider": "nous", "cost": {"maximum": 5}},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(bad_config, f)
        f.flush()
        try:
            load_config(f.name)
            assert False, "Should have raised ConfigError for non-zero cost"
        except ConfigError:
            pass
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    test_load_valid_config()
    test_load_missing_file()
    test_invalid_cost_policy()
    print("All config tests passed!")
