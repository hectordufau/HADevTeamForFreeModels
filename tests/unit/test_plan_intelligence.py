"""Tests for Plan Intelligence (Phase I)."""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.planner.plan_intelligence import (
    PlanScorer, AlternativeGenerator, PlanSelector,
    PlanLearningStore, PlanScore, WorkflowAlternative,
    PlanIntelligenceError,
)
from harness.capabilities.registry import CapabilityRegistry
from harness.capabilities.taxonomy import CapabilityTaxonomy
from harness.planner.workflow_planner import ExecutionWorkflow, WorkflowNode


def make_workflow():
    return ExecutionWorkflow(
        name="test",
        nodes=[
            WorkflowNode(id="design", capability="api_design", agent="architect",
                        model_id="m1:free", depends_on=[]),
            WorkflowNode(id="impl", capability="backend_development", agent="coder",
                        model_id="m2:free", depends_on=["design"]),
            WorkflowNode(id="test", capability="testing", agent="tester",
                        model_id="m3:free", depends_on=["impl"]),
            WorkflowNode(id="review", capability="review", agent="reviewer",
                        model_id="m4:free", depends_on=["test"]),
        ],
    )


def test_plan_scorer():
    reg = CapabilityRegistry()
    reg.register_agent("architect", ["api_design"])
    reg.register_agent("coder", ["backend_development"])
    reg.register_agent("tester", ["testing"])
    reg.register_agent("reviewer", ["review", "code_review"])

    taxo = CapabilityTaxonomy(taxonomy_path=os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config", "capability_taxonomy.yaml"
    ))
    scorer = PlanScorer(reg, taxo)

    workflow = make_workflow()
    score = scorer.score(workflow)

    assert 0 <= score.overall <= 1
    assert 0 <= score.capability_coverage <= 1
    assert 0 <= score.agent_fit <= 1


def test_alternative_generator():
    reg = CapabilityRegistry()
    reg.register_agent("architect", ["api_design", "system_design"])
    reg.register_agent("coder", ["backend_development", "frontend_development"])
    reg.register_agent("tester", ["testing"])
    reg.register_agent("reviewer", ["review", "code_review"])

    taxo = CapabilityTaxonomy(taxonomy_path=os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "config", "capability_taxonomy.yaml"
    ))

    from harness.planner.workflow_planner import WorkflowPlanner
    from harness.routing import ModelCatalog, ModelCandidate, ModelRouter

    catalog = ModelCatalog()
    catalog.register(ModelCandidate(
        model_id="m:free", provider="nous", cost=0,
        capabilities={"coding": 0.9},
    ))

    planner = WorkflowPlanner(reg, taxo, ModelRouter(catalog))
    generator = AlternativeGenerator(planner, reg, taxo)

    class MockRequirements:
        required = ["api_design", "backend_development", "testing", "review"]
        optional = []

    alternatives = generator.generate(MockRequirements(), count=2)
    assert len(alternatives) >= 2
    for alt in alternatives:
        assert isinstance(alt, WorkflowAlternative)
        assert alt.score.overall > 0


def test_plan_selector():
    scorer = PlanScorer(None, None)
    alternatives = [
        WorkflowAlternative(
            id="a1", nodes=[],
            score=PlanScore(overall=0.9, capability_coverage=0.8, agent_fit=1.0,
                          model_quality=0.7, efficiency=0.5, risk=0.2, reasoning=""),
            generated_by="planner",
        ),
        WorkflowAlternative(
            id="a2", nodes=[],
            score=PlanScore(overall=0.7, capability_coverage=0.6, agent_fit=0.8,
                          model_quality=0.7, efficiency=0.5, risk=0.3, reasoning=""),
            generated_by="mutation",
        ),
    ]
    selector = PlanSelector(scorer)
    best = selector.select(alternatives)
    assert best.id == "a1"


def test_plan_selector_empty():
    scorer = PlanScorer(None, None)
    selector = PlanSelector(scorer)
    try:
        selector.select([])
        assert False, "Should have raised"
    except PlanIntelligenceError:
        pass


def test_plan_learning_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = PlanLearningStore(storage_dir=os.path.join(tmpdir, "plans"))

        score = PlanScore(overall=0.85, capability_coverage=0.8, agent_fit=0.9,
                         model_quality=0.7, efficiency=0.6, risk=0.2, reasoning="")
        store.record_outcome("plan_test", score, "COMPLETED")

        best = store.get_best_scoring_pattern()
        assert best is not None
        assert best["score"]["overall"] == 0.85
