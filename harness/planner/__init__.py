# harness/planner/__init__.py — V2.2 Dynamic Planner
"""Task Analyzer, Workflow Planner, Validator, Execution Graph, and adaptive replanning."""

from .task_analyzer import TaskAnalyzer, TaskAnalyzerError, CapabilityRequirements, MANDATORY_CAPABILITIES
from .workflow_planner import WorkflowPlanner, WorkflowPlannerError, WorkflowNode, ExecutionWorkflow
from .workflow_validator import WorkflowValidator, WorkflowValidatorError, ValidationResult
from .exec_graph import ExecutionGraph, ExecutionGraphError, NodeExecutionResult
from .workspace import WorkspaceManager, WorkspaceManagerError, Conflict
from .branching import BranchResolver, BranchingError
from .failure_analyzer import FailureAnalyzer, FailureAnalyzerError, FailureReport
from .replanner import Replanner, ReplannerError, ReplanReport
from .knowledge_integration import (
    KnowledgeIntegrationError,
    KnowledgeConflictError,
    KnowledgeGapError,
    CapabilityGapError,
    KnowledgeInfluence,
    KnowledgeDerivedCapability,
    KnowledgeGap,
    PlanningContext,
    KnowledgeAwareTaskAnalyzer,
    KnowledgeAwareWorkflowPlanner,
    KnowledgeAwarePlanScorer,
    KnowledgeAwareStrategyGenerator,
    KnowledgeAwareDecisionPipeline,
    validate_knowledge_constraints,
    NFR_CATEGORY_TO_CAPABILITY,
    SEC_CONSTRAINT_TO_CAPABILITY,
    TDR_DEBT_TYPE_TO_CAPABILITY,
)

__all__ = [
    "TaskAnalyzer", "TaskAnalyzerError", "CapabilityRequirements", "MANDATORY_CAPABILITIES",
    "WorkflowPlanner", "WorkflowPlannerError", "WorkflowNode", "ExecutionWorkflow",
    "WorkflowValidator", "WorkflowValidatorError", "ValidationResult",
    "ExecutionGraph", "ExecutionGraphError", "NodeExecutionResult",
    "WorkspaceManager", "WorkspaceManagerError", "Conflict",
    "BranchResolver", "BranchingError",
    "FailureAnalyzer", "FailureAnalyzerError", "FailureReport",
    "Replanner", "ReplannerError", "ReplanReport",
    # Phase 9: Knowledge Integration
    "KnowledgeIntegrationError",
    "KnowledgeConflictError",
    "KnowledgeGapError",
    "CapabilityGapError",
    "KnowledgeInfluence",
    "KnowledgeDerivedCapability",
    "KnowledgeGap",
    "PlanningContext",
    "KnowledgeAwareTaskAnalyzer",
    "KnowledgeAwareWorkflowPlanner",
    "KnowledgeAwarePlanScorer",
    "KnowledgeAwareStrategyGenerator",
    "KnowledgeAwareDecisionPipeline",
    "validate_knowledge_constraints",
    "NFR_CATEGORY_TO_CAPABILITY",
    "SEC_CONSTRAINT_TO_CAPABILITY",
    "TDR_DEBT_TYPE_TO_CAPABILITY",
]
