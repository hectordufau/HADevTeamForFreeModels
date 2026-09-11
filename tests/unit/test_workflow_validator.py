# Tests for WorkflowValidator
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.planner.workflow_validator import WorkflowValidator, WorkflowValidatorError, ValidationResult
from harness.planner.workflow_planner import WorkflowNode, ExecutionWorkflow
from harness.capabilities.registry import CapabilityRegistry
from harness.capabilities.taxonomy import CapabilityTaxonomy


def test_valid_workflow_passes():
    validator = WorkflowValidator()
    workflow = ExecutionWorkflow(
        name="test",
        nodes=[
            WorkflowNode(id="design", capability="api_design", agent="architect", model_id="mock:free", depends_on=[]),
            WorkflowNode(id="impl", capability="backend_development", agent="coder", model_id="mock:free", depends_on=["design"]),
            WorkflowNode(id="test", capability="testing", agent="tester", model_id="mock:free", depends_on=["impl"]),
            WorkflowNode(id="review", capability="review", agent="reviewer", model_id="mock:free", depends_on=["test"]),
            WorkflowNode(id="verify", capability="verification", agent="tester", model_id="mock:free", depends_on=["impl"]),
            WorkflowNode(id="eval", capability="evaluation", agent="reviewer", model_id="mock:free", depends_on=["test"], on_pass="complete"),
        ],
    )
    result = validator.validate(workflow)
    assert result.valid is True


def test_missing_mandatory_capabilities_fails():
    validator = WorkflowValidator()
    workflow = ExecutionWorkflow(
        name="test",
        nodes=[
            WorkflowNode(id="impl", capability="backend_development", agent="coder", model_id="mock:free", depends_on=[]),
        ],
    )
    result = validator.validate(workflow)
    assert result.valid is False
    assert any("MISSING_MANDATORY" in i.code for i in result.issues)


def test_cycle_detected():
    validator = WorkflowValidator()
    workflow = ExecutionWorkflow(
        name="test",
        nodes=[
            WorkflowNode(id="a", capability="arch", agent="architect", model_id="mock:free", depends_on=["c"]),
            WorkflowNode(id="b", capability="impl", agent="coder", model_id="mock:free", depends_on=["a"]),
            WorkflowNode(id="c", capability="test", agent="tester", model_id="mock:free", depends_on=["b"]),
        ],
    )
    result = validator.validate(workflow)
    assert result.valid is False
    assert any("CYCLE_DETECTED" in i.code for i in result.issues)


def test_paid_model_fails():
    validator = WorkflowValidator()
    workflow = ExecutionWorkflow(
        name="test",
        nodes=[
            WorkflowNode(id="design", capability="api_design", agent="architect", model_id="paid:model", depends_on=[]),
            WorkflowNode(id="test", capability="testing", agent="tester", model_id="mock:free", depends_on=["design"]),
            WorkflowNode(id="review", capability="review", agent="reviewer", model_id="mock:free", depends_on=["test"]),
            WorkflowNode(id="verify", capability="verification", agent="tester", model_id="mock:free", depends_on=["design"]),
            WorkflowNode(id="eval", capability="evaluation", agent="reviewer", model_id="mock:free", depends_on=["test"]),
        ],
    )
    result = validator.validate(workflow)
    assert result.valid is False
    assert any("PAID_MODEL" in i.code for i in result.issues)


def test_no_reviewer_fails():
    validator = WorkflowValidator()
    workflow = ExecutionWorkflow(
        name="test",
        nodes=[
            WorkflowNode(id="impl", capability="backend_development", agent="coder", model_id="mock:free", depends_on=[]),
            WorkflowNode(id="test", capability="testing", agent="tester", model_id="mock:free", depends_on=["impl"]),
            WorkflowNode(id="verify", capability="verification", agent="tester", model_id="mock:free", depends_on=["impl"]),
            WorkflowNode(id="eval", capability="evaluation", agent="reviewer", model_id="mock:free", depends_on=["test"]),
        ],
    )
    result = validator.validate(workflow)
    assert result.valid is False
    assert any("MISSING_MANDATORY" in i.code for i in result.issues)


def test_no_termination_path():
    validator = WorkflowValidator()
    workflow = ExecutionWorkflow(
        name="test",
        nodes=[
            WorkflowNode(id="a", capability="a", agent="coder", model_id="mock:free", depends_on=["b"]),
            WorkflowNode(id="b", capability="b", agent="coder", model_id="mock:free", depends_on=["a"]),
        ],
    )
    result = validator.validate(workflow)
    assert result.valid is False


def test_workspace_conflict_detected():
    validator = WorkflowValidator()
    workflow = ExecutionWorkflow(
        name="test",
        nodes=[
            WorkflowNode(id="impl1", capability="backend_development", agent="coder", model_id="mock:free",
                        depends_on=[], workspace={"write": ["src/**"], "read": []}),
            WorkflowNode(id="impl2", capability="api_implementation", agent="coder", model_id="mock:free",
                        depends_on=[], workspace={"write": ["src/**"], "read": []}),
        ],
    )
    result = validator.validate(workflow)
    assert result.valid is False
    assert any("WORKSPACE_CONFLICT" in i.code for i in result.issues)


def test_validate_empty_workflow():
    validator = WorkflowValidator()
    workflow = ExecutionWorkflow(name="empty", nodes=[])
    result = validator.validate(workflow)
    assert result.valid is False
    assert len(result.issues) > 0


def test_dag_validation():
    validator = WorkflowValidator()
    # Test acyclic
    node_ids = {"a", "b", "c"}
    node_deps = {"a": [], "b": ["a"], "c": ["b"]}
    assert validator._validate_dag(node_ids, node_deps) is True

    # Test cyclic
    node_ids2 = {"x", "y", "z"}
    node_deps2 = {"x": ["z"], "y": ["x"], "z": ["y"]}
    assert validator._validate_dag(node_ids2, node_deps2) is False


def test_mandatory_cap_validation():
    validator = WorkflowValidator()
    # With all mandatory caps
    assert validator._validate_mandatory_capabilities({"verification", "evaluation", "review", "coding"}) is True
    # Without one
    assert validator._validate_mandatory_capabilities({"coding", "testing"}) is False


if __name__ == "__main__":
    test_valid_workflow_passes()
    test_missing_mandatory_capabilities_fails()
    test_cycle_detected()
    test_paid_model_fails()
    test_no_reviewer_fails()
    test_no_termination_path()
    test_workspace_conflict_detected()
    test_validate_empty_workflow()
    test_dag_validation()
    test_mandatory_cap_validation()
    print("All workflow_validator tests passed!")
