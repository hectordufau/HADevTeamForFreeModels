# Tests for CapabilityTaxonomy
import sys, os, tempfile, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.capabilities.taxonomy import CapabilityTaxonomy, CapabilityTaxonomyError


def test_load_taxonomy():
    taxonomy = CapabilityTaxonomy()
    # The default path should exist
    assert taxonomy._tree is not None


def test_get_children():
    taxonomy = CapabilityTaxonomy()
    # Setup test data
    taxonomy._all_capabilities = {"testing", "unit_testing", "integration_testing", "e2e_testing"}
    taxonomy._children = {
        "testing": ["unit_testing", "integration_testing", "e2e_testing"],
    }
    taxonomy._parents = {
        "unit_testing": ["testing"],
        "integration_testing": ["testing"],
        "e2e_testing": ["testing"],
    }

    children = taxonomy.get_children("testing")
    assert sorted(children) == sorted(["unit_testing", "integration_testing", "e2e_testing"])

    children = taxonomy.get_children("unit_testing")
    assert children == []


def test_get_parents():
    taxonomy = CapabilityTaxonomy()
    taxonomy._all_capabilities = {"testing", "unit_testing"}
    taxonomy._children = {"testing": ["unit_testing"]}
    taxonomy._parents = {"unit_testing": ["testing"]}

    parents = taxonomy.get_parents("unit_testing")
    assert parents == ["testing"]

    parents = taxonomy.get_parents("testing")
    assert parents == []


def test_get_siblings():
    taxonomy = CapabilityTaxonomy()
    taxonomy._all_capabilities = {"unit_testing", "integration_testing", "e2e_testing"}
    taxonomy._children = {"testing": ["unit_testing", "integration_testing", "e2e_testing"]}
    taxonomy._parents = {
        "unit_testing": ["testing"],
        "integration_testing": ["testing"],
        "e2e_testing": ["testing"],
    }

    siblings = taxonomy.get_siblings("unit_testing")
    assert sorted(siblings) == sorted(["integration_testing", "e2e_testing"])

    siblings = taxonomy.get_siblings("testing")
    assert siblings == []


def test_is_ancestor():
    taxonomy = CapabilityTaxonomy()
    taxonomy._all_capabilities = {"engineering", "implementation", "backend", "backend_development"}
    taxonomy._children = {
        "engineering": ["implementation"],
        "implementation": ["backend"],
        "backend": ["backend_development"],
    }
    taxonomy._parents = {
        "implementation": ["engineering"],
        "backend": ["implementation"],
        "backend_development": ["backend"],
    }

    assert taxonomy.is_ancestor("engineering", "backend_development") is True
    assert taxonomy.is_ancestor("implementation", "backend_development") is True
    assert taxonomy.is_ancestor("backend_development", "engineering") is False
    assert taxonomy.is_ancestor("engineering", "engineering") is False


def test_is_descendant():
    taxonomy = CapabilityTaxonomy()
    taxonomy._all_capabilities = {"engineering", "implementation", "backend"}
    taxonomy._children = {"engineering": ["implementation"], "implementation": ["backend"]}
    taxonomy._parents = {"implementation": ["engineering"], "backend": ["implementation"]}

    assert taxonomy.is_descendant("backend", "engineering") is True
    assert taxonomy.is_descendant("backend", "implementation") is True
    assert taxonomy.is_descendant("engineering", "backend") is False


def test_get_all_capabilities():
    taxonomy = CapabilityTaxonomy()
    taxonomy._all_capabilities = {"a", "b", "c"}
    assert taxonomy.get_all_capabilities() == {"a", "b", "c"}


def test_get_leaves():
    taxonomy = CapabilityTaxonomy()
    taxonomy._all_capabilities = {"a", "b", "c"}
    taxonomy._children = {"a": ["b"], "b": ["c"]}
    leaves = taxonomy.get_leaves()
    assert leaves == ["c"]


def test_capability_exists():
    taxonomy = CapabilityTaxonomy()
    taxonomy._all_capabilities = {"testing"}
    assert taxonomy.capability_exists("testing") is True
    assert taxonomy.capability_exists("nonexistent") is False


def test_load_from_yaml():
    taxonomy_data = {
        "engineering": {
            "testing": {
                "unit_testing": {},
                "integration_testing": {},
            },
            "implementation": {
                "backend_development": {},
            },
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(taxonomy_data, f)
        f.flush()
        taxonomy = CapabilityTaxonomy(f.name)
        assert taxonomy.capability_exists("testing") is True
        assert taxonomy.capability_exists("unit_testing") is True
        assert taxonomy.capability_exists("backend_development") is True
        assert taxonomy.get_children("testing") == sorted(["unit_testing", "integration_testing"])
        os.unlink(f.name)


def test_missing_file():
    try:
        CapabilityTaxonomy("/nonexistent/path.yaml")
    except CapabilityTaxonomyError:
        pass


if __name__ == "__main__":
    test_load_taxonomy()
    test_get_children()
    test_get_parents()
    test_get_siblings()
    test_is_ancestor()
    test_is_descendant()
    test_get_all_capabilities()
    test_get_leaves()
    test_capability_exists()
    test_load_from_yaml()
    test_missing_file()
    print("All capability_taxonomy tests passed!")
