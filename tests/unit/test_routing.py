# Tests for harness/routing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from harness.routing import ModelCatalog, ModelCandidate, ModelRouter, RouterError


def test_register_and_list():
    catalog = ModelCatalog()
    catalog.register(ModelCandidate(
        model_id="model-a:free", provider="nous", cost=0,
        capabilities={"coding": 0.9, "reasoning": 0.8},
    ))
    catalog.register(ModelCandidate(
        model_id="model-b:premium", provider="nous", cost=5,
        capabilities={"coding": 1.0, "reasoning": 0.95},
    ))
    free_models = catalog.list_free()
    assert len(free_models) == 1
    assert free_models[0].model_id == "model-a:free"


def test_select_by_capability():
    catalog = ModelCatalog()
    catalog.register(ModelCandidate(
        model_id="model-a:free", provider="nous", cost=0,
        capabilities={"coding": 0.9, "reasoning": 0.6},
    ))
    catalog.register(ModelCandidate(
        model_id="model-b:free", provider="nous", cost=0,
        capabilities={"coding": 0.6, "reasoning": 0.95},
    ))
    router = ModelRouter(catalog)

    # Prefer reasoning
    selection = router.select(["reasoning"], role="architect")
    assert selection.model_id == "model-b:free"
    assert selection.cost == 0

    # Prefer coding
    selection = router.select(["coding"], role="coder")
    assert selection.model_id == "model-a:free"


def test_paid_model_rejected():
    catalog = ModelCatalog()
    catalog.register(ModelCandidate(
        model_id="paid-model", provider="openai", cost=10,
        capabilities={"coding": 1.0},
    ))
    router = ModelRouter(catalog)
    try:
        router.select(["coding"], role="coder")
        assert False, "Should have raised RouterError (no free models)"
    except RouterError:
        pass


def test_fallback_selection():
    catalog = ModelCatalog()
    catalog.register(ModelCandidate(
        model_id="best:free", provider="nous", cost=0,
        capabilities={"coding": 0.95, "reasoning": 0.9},
    ))
    catalog.register(ModelCandidate(
        model_id="second:free", provider="nous", cost=0,
        capabilities={"coding": 0.7, "reasoning": 0.7},
    ))
    router = ModelRouter(catalog)
    selection = router.select(["coding"], role="coder", prefer_fallback=True)
    assert selection.model_id == "second:free"
    assert selection.fallback_used


if __name__ == "__main__":
    test_register_and_list()
    test_select_by_capability()
    test_paid_model_rejected()
    test_fallback_selection()
    print("All routing tests passed!")
