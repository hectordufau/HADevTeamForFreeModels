# Tests for CapabilityTaxonomy dependency loading
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.capabilities.taxonomy import CapabilityTaxonomy


def _get_taxonomy():
    repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    path = os.path.join(repo_root, "config", "capability_taxonomy.yaml")
    return CapabilityTaxonomy(taxonomy_path=path)


def test_dependencies_loaded_from_config():
    """Verify dependencies are loaded from capability_dependencies section in config."""
    taxonomy = _get_taxonomy()

    # Architecture/design capabilities have no dependencies
    assert taxonomy.get_dependencies("api_design") == []
    assert taxonomy.get_dependencies("system_design") == []
    assert taxonomy.get_dependencies("domain_modeling") == []

    # Implementation depends on design
    deps = taxonomy.get_dependencies("backend_development")
    assert "api_design" in deps
    assert "system_design" in deps

    deps = taxonomy.get_dependencies("api_implementation")
    assert "api_design" in deps

    # Testing depends on implementation
    deps = taxonomy.get_dependencies("testing")
    assert "backend_development" in deps
    assert "api_implementation" in deps

    # Review depends on testing and security
    deps = taxonomy.get_dependencies("review")
    assert "testing" in deps
    assert "security_analysis" in deps

    # Security depends on implementation
    deps = taxonomy.get_dependencies("security_analysis")
    assert "backend_development" in deps
    assert "api_implementation" in deps


def test_has_dependencies_declared():
    """has_dependencies_declared flag works correctly."""
    taxonomy = _get_taxonomy()

    # Capabilities with explicit deps
    assert taxonomy.has_dependencies_declared("backend_development") is True
    assert taxonomy.has_dependencies_declared("testing") is True
    assert taxonomy.has_dependencies_declared("review") is True

    # Capabilities with empty deps list
    assert taxonomy.has_dependencies_declared("api_design") is True

    # Non-existent capability
    assert taxonomy.has_dependencies_declared("nonexistent_cap") is False


def test_dependency_filtering():
    """Dependencies only include capabilities that are valid taxonomy entries."""
    taxonomy = _get_taxonomy()
    deps = taxonomy.get_dependencies("backend_development")
    assert isinstance(deps, list)
    for d in deps:
        assert taxonomy.capability_exists(d), f"Dependency '{d}' must exist in taxonomy"


def test_unknown_capability_returns_empty():
    """Unknown capabilities return empty dependency list."""
    taxonomy = _get_taxonomy()
    assert taxonomy.get_dependencies("unknown_capability") == []
    assert taxonomy.has_dependencies_declared("unknown_capability") is False


if __name__ == "__main__":
    test_dependencies_loaded_from_config()
    test_has_dependencies_declared()
    test_dependency_filtering()
    test_unknown_capability_returns_empty()
    print("All capability_taxonomy_deps tests passed!")
