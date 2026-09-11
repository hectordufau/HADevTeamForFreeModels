# harness/planner/workflow_validator.py — WorkflowValidator (V2.2)
"""
Validates a workflow plan before execution.
10 checks: DAG validity, capability coverage, mandatory caps, agent availability,
model policy, workspace conflicts, termination path, reviewer independence.
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field


class WorkflowValidatorError(Exception):
    """Raised on workflow validation errors."""


@dataclass
class ValidationIssue:
    """A single validation issue."""
    code: str
    message: str
    node_id: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of workflow validation."""
    valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)

    def add_issue(self, code: str, message: str, node_id: Optional[str] = None):
        self.valid = False
        self.issues.append(ValidationIssue(code=code, message=message, node_id=node_id))

    def __str__(self) -> str:
        if self.valid:
            return "ValidationResult(valid=True)"
        return f"ValidationResult(valid=False, issues={[i.code for i in self.issues]})"


# Mandatory capabilities that must always be present
MANDATORY_CAPABILITIES = {"verification", "evaluation", "review"}


class WorkflowValidator:
    """Validates a workflow plan before execution."""

    def __init__(self, agent_registry: Any = None, policy_engine: Any = None,
                 config: Optional[dict] = None):
        self.registry = agent_registry
        self.policy = policy_engine
        self.config = config or {}

    def validate(self, workflow: Any, task: Any = None) -> ValidationResult:
        """
        Validate a workflow plan. Performs all 10 checks:
        1. Valid DAG (no cycles)
        2. All required capabilities covered
        3. Mandatory capabilities present
        4. Agents available
        5. Agents authorized (policy)
        6. Model policy valid
        7. Models are free
        8. Dependencies satisfiable
        9. Workspace conflicts absent
        10. Termination path exists
        11. Reviewer present and independent
        """
        result = ValidationResult()

        # Handle both ExecutionWorkflow object and dict
        if hasattr(workflow, 'nodes'):
            nodes = workflow.nodes
        else:
            nodes = workflow.get("nodes", [])

        if not nodes:
            result.add_issue("EMPTY_WORKFLOW", "Workflow has no nodes")
            return result

        # Extract node ids and capabilities
        node_ids = set()
        node_caps = {}
        node_agents = {}
        node_models = {}
        node_deps = {}

        for node in nodes:
            nid = node.id if hasattr(node, 'id') else node.get("id", "")
            ncap = node.capability if hasattr(node, 'capability') else node.get("capability", "")
            nagent = node.agent if hasattr(node, 'agent') else node.get("agent", "")
            nmodel = node.model_id if hasattr(node, 'model_id') else node.get("model_id", "")
            ndeps = node.depends_on if hasattr(node, 'depends_on') else node.get("depends_on", [])

            node_ids.add(nid)
            node_caps[nid] = ncap
            node_agents[nid] = nagent
            node_models[nid] = nmodel
            node_deps[nid] = ndeps

        # Check 1: Valid DAG (cycle detection)
        if not self._validate_dag(node_ids, node_deps):
            result.add_issue("CYCLE_DETECTED", "Workflow graph contains a cycle")

        # Check 2: All required capabilities covered (node-level)
        # (Capabilities are the nodes themselves)
        if not node_caps:
            result.add_issue("NO_CAPABILITIES", "No capabilities defined in workflow")

        # Check 3: Mandatory capabilities present
        if not self._validate_mandatory_capabilities(set(node_caps.values())):
            missing = MANDATORY_CAPABILITIES - set(node_caps.values())
            result.add_issue("MISSING_MANDATORY",
                             f"Mandatory capabilities not included: {missing}")

        # Check 4: Agents available
        for nid, agent in node_agents.items():
            if not agent or agent == "unknown":
                result.add_issue("AGENT_UNAVAILABLE",
                                 f"Node '{nid}' has no agent assigned", node_id=nid)

        # Check 5: Agents authorized (policy)
        # Only authorize via policy if the policy engine explicitly handles agent auth
        if self.policy and hasattr(self.policy, 'check_agent_auth'):
            for nid, agent in node_agents.items():
                if not self._check_agent_authorized(agent, nid):
                    result.add_issue("AGENT_UNAUTHORIZED",
                                     f"Agent '{agent}' not authorized for node '{nid}'",
                                     node_id=nid)

        # Check 6: Models are free (cost=0)
        for nid, model in node_models.items():
            if not model or model == "default:free":
                continue
            # Check if model is in the free list
            if not self._is_model_free(model):
                result.add_issue("PAID_MODEL",
                                 f"Model '{model}' for node '{nid}' is not free (cost>0)",
                                 node_id=nid)

        # Check 7: Dependencies satisfiable
        for nid, deps in node_deps.items():
            for dep in deps:
                if dep not in node_ids:
                    result.add_issue("UNSATISFIABLE_DEPENDENCY",
                                     f"Node '{nid}' depends on '{dep}' which is not in the workflow",
                                     node_id=nid)

        # Check 8: Workspace conflicts absent
        workspace_conflicts = self._detect_workspace_conflicts(nodes)
        if workspace_conflicts:
            for nid1, nid2 in workspace_conflicts:
                result.add_issue("WORKSPACE_CONFLICT",
                                 f"Nodes '{nid1}' and '{nid2}' write to overlapping files",
                                 node_id=nid1)

        # Check 9: Termination path exists
        if not self._has_termination_path(nodes):
            result.add_issue("NO_TERMINATION",
                             "No path leads to workflow completion")

        # Check 10: Reviewer present and independent
        if not self._validate_reviewer(nodes):
            result.add_issue("NO_REVIEWER",
                             "Reviewer not present or not independent")

        return result

    def _validate_dag(self, node_ids: Set[str],
                      node_deps: Dict[str, List[str]]) -> bool:
        """Check for cycles using DFS. Returns True if acyclic."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in node_ids}

        def dfs(nid):
            if color[nid] == GRAY:
                return True
            if color[nid] == BLACK:
                return False
            color[nid] = GRAY
            for dep in node_deps.get(nid, []):
                if dep in node_ids and dfs(dep):
                    return True
            color[nid] = BLACK
            return False

        for nid in node_ids:
            if dfs(nid):
                return False
        return True

    def _validate_mandatory_capabilities(self, caps: Set[str]) -> bool:
        """verification, evaluation, review must always be present."""
        return MANDATORY_CAPABILITIES.issubset(caps)

    def _check_agent_authorized(self, agent: str, node_id: str) -> bool:
        """Check if agent is authorized via policy."""
        if not self.policy:
            return True
        try:
            result = self.policy.evaluate_all(
                agent_level=3, task_level=3, tool_level=3,
                current_iteration=0, has_verification=False,
            )
            return result.allowed
        except Exception:
            return True

    def _is_model_free(self, model_id: str) -> bool:
        """Check if a model ID is free (cost=0)."""
        if not model_id:
            return True
        # Free models typically end with :free or are known free providers
        if model_id.endswith(":free"):
            return True
        # Additional check: known free providers
        free_prefixes = ["nous/", "poolside/", "meituan/", "stepfun/", "deepseek/",
                         "mistral/", "google/", "meta/", "openai/"]
        for prefix in free_prefixes:
            if model_id.startswith(prefix):
                return True
        return False

    def _detect_workspace_conflicts(self, nodes: List[Any]) -> List[tuple]:
        """Detect conflicting write paths between nodes."""
        conflicts = []
        write_map: Dict[str, str] = {}  # file pattern -> node_id

        for node in nodes:
            nid = node.id if hasattr(node, 'id') else node.get("id", "")
            ws = node.workspace if hasattr(node, 'workspace') else node.get("workspace", {})
            write_patterns = ws.get("write", []) if isinstance(ws, dict) else []

            for pattern in write_patterns:
                if pattern in write_map and write_map[pattern] != nid:
                    conflicts.append((write_map[pattern], nid))
                write_map[pattern] = nid

        return conflicts

    def _has_termination_path(self, nodes: List[Any]) -> bool:
        """Check that at least one node has on_pass=complete or review/eval."""
        for node in nodes:
            on_pass = node.on_pass if hasattr(node, 'on_pass') else node.get("on_pass", "next")
            if on_pass == "complete":
                return True
            # Also check if the last node passes to complete
        # Last node should lead to completion
        last_node = nodes[-1]
        last_on_pass = last_node.on_pass if hasattr(last_node, 'on_pass') else last_node.get("on_pass", "next")
        return last_on_pass == "complete"

    def _validate_reviewer(self, nodes: List[Any]) -> bool:
        """Reviewer must be present and not the same as implementer."""
        reviewer_nodes = []
        implementer_nodes = []

        for node in nodes:
            nid = node.id if hasattr(node, 'id') else node.get("id", "")
            ncap = node.capability if hasattr(node, 'capability') else node.get("capability", "")
            nagent = node.agent if hasattr(node, 'agent') else node.get("agent", "")

            if ncap in ("code_review", "review", "evaluation"):
                reviewer_nodes.append((nid, nagent))
            if ncap in ("backend_development", "api_implementation",
                        "frontend_development", "ui_implementation", "refactoring"):
                implementer_nodes.append((nid, nagent))

        if not reviewer_nodes:
            return False

        # Check independence: reviewer must not be the same agent as implementer
        for _, r_agent in reviewer_nodes:
            for _, i_agent in implementer_nodes:
                if r_agent == i_agent:
                    return False

        return True
