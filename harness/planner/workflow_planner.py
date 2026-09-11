# harness/planner/workflow_planner.py — WorkflowPlanner (V2.2)
"""
Builds an ExecutionWorkflow DAG from capability requirements.
1. Look up required capabilities
2. Find agents for each capability
3. Build dependency graph
4. Assign models via AdaptiveModelRouter
5. Return ExecutionWorkflow
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


class WorkflowPlannerError(Exception):
    """Raised on workflow planning errors."""


@dataclass
class WorkflowNode:
    """A single node in the execution workflow."""
    id: str
    capability: str
    agent: str
    model_id: str
    depends_on: List[str] = field(default_factory=list)
    on_pass: str = "next"   # next, complete, or node_id
    on_fail: str = "block"  # block, retry, feedback, or node_id
    workspace: Dict[str, List[str]] = field(default_factory=lambda: {"write": [], "read": []})


@dataclass
class ExecutionWorkflow:
    """A complete workflow plan ready for execution."""
    name: str = "dynamic"
    nodes: List[WorkflowNode] = field(default_factory=list)
    max_iterations: int = 3

    def get_node(self, node_id: str) -> Optional[WorkflowNode]:
        """Get a node by ID."""
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def to_dict(self) -> dict:
        """Serialize to dict for artifact persistence."""
        return {
            "name": self.name,
            "nodes": [
                {
                    "id": n.id,
                    "capability": n.capability,
                    "agent": n.agent,
                    "model_id": n.model_id,
                    "depends_on": n.depends_on,
                    "on_pass": n.on_pass,
                    "on_fail": n.on_fail,
                    "workspace": n.workspace,
                }
                for n in self.nodes
            ],
            "max_iterations": self.max_iterations,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionWorkflow":
        """Deserialize from dict."""
        nodes = [
            WorkflowNode(
                id=n["id"],
                capability=n["capability"],
                agent=n["agent"],
                model_id=n["model_id"],
                depends_on=n.get("depends_on", []),
                on_pass=n.get("on_pass", "next"),
                on_fail=n.get("on_fail", "block"),
                workspace=n.get("workspace", {"write": [], "read": []}),
            )
            for n in data.get("nodes", [])
        ]
        return cls(name=data.get("name", "dynamic"), nodes=nodes,
                   max_iterations=data.get("max_iterations", 3))


# Mapping from taxonomy capabilities to agent roles
# Used when the CapabilityRegistry doesn't have an agent for a capability
DEFAULT_AGENT_FOR_CAPABILITY = {
    "api_design": "architect",
    "system_design": "architect",
    "domain_modeling": "architect",
    "integration_design": "architect",
    "backend_development": "coder",
    "api_implementation": "coder",
    "database_schema": "coder",
    "frontend_development": "coder",
    "ui_implementation": "coder",
    "refactoring": "coder",
    "testing": "tester",
    "unit_testing": "tester",
    "integration_testing": "tester",
    "e2e_testing": "tester",
    "security_analysis": "reviewer",
    "vulnerability_assessment": "reviewer",
    "code_review": "reviewer",
    "verification": "tester",
    "evaluation": "reviewer",
    "review": "reviewer",
}


class WorkflowPlanner:
    """Builds an ExecutionWorkflow DAG from capability requirements."""

    def __init__(self, agent_registry: Any, taxonomy: Any,
                 model_router: Any, adaptive_router: Any = None,
                 config: Optional[dict] = None):
        self.registry = agent_registry
        self.taxonomy = taxonomy
        self.router = model_router
        self.adaptive_router = adaptive_router
        self.config = config or {}

    def plan(self, requirements: Any,
             task: Any = None) -> ExecutionWorkflow:
        """
        Build a workflow from capability requirements.

        1. Look up required capabilities
        2. Find agents for each capability
        3. Build dependency graph
        4. Assign models via router
        5. Return ExecutionWorkflow
        """
        # Get required caps (handle both CapabilityRequirements object and dict)
        if hasattr(requirements, 'required'):
            required_caps = requirements.required
            optional_caps = getattr(requirements, 'optional', [])
        else:
            required_caps = requirements.get("required", [])
            optional_caps = requirements.get("optional", [])

        # Map capabilities to agents
        node_map = {}  # capability -> node_id

        # First pass: create nodes for each required capability
        nodes = []
        for cap in required_caps:
            node_id = cap
            agent = self._find_agent_for_capability(cap)
            model_id = self._select_model(cap, agent, task)

            # Determine dependencies based on capability relationships
            deps = self._compute_dependencies(cap, required_caps)

            # Determine branching
            on_pass, on_fail = self._determine_branching(cap, required_caps)

            # Determine workspace scope
            workspace = self._determine_workspace(cap)

            node = WorkflowNode(
                id=node_id,
                capability=cap,
                agent=agent,
                model_id=model_id,
                depends_on=deps,
                on_pass=on_pass,
                on_fail=on_fail,
                workspace=workspace,
            )
            nodes.append(node)
            node_map[cap] = node_id

        # Add optional capabilities that can be covered
        for cap in optional_caps:
            if cap not in node_map:
                agent = self._find_agent_for_capability(cap)
                if agent:
                    model_id = self._select_model(cap, agent, task)
                    deps = self._compute_dependencies(cap, required_caps)
                    on_pass, on_fail = self._determine_branching(cap, required_caps)
                    workspace = self._determine_workspace(cap)

                    node = WorkflowNode(
                        id=cap,
                        capability=cap,
                        agent=agent,
                        model_id=model_id,
                        depends_on=deps,
                        on_pass=on_pass,
                        on_fail=on_fail,
                        workspace=workspace,
                    )
                    nodes.append(node)
                    node_map[cap] = cap

        return ExecutionWorkflow(
            name="dynamic",
            nodes=nodes,
            max_iterations=self.config.get("max_iterations", 3),
        )

    def plan_from_task(self, task: Any, task_analyzer: Any) -> ExecutionWorkflow:
        """Analyze task + plan in one call."""
        requirements = task_analyzer.analyze(task)
        return self.plan(requirements, task=task)

    def _find_agent_for_capability(self, capability: str) -> str:
        """Find the best agent for a capability."""
        # Try CapabilityRegistry first
        if self.registry:
            agents = self.registry.find_agents_for(capability)
            if agents:
                return agents[0]
            # Try with taxonomy
            best_agent, score = self.registry.find_best_agent([capability], self.taxonomy)
            if best_agent and score > 0:
                return best_agent

        # Fall back to default mapping
        return DEFAULT_AGENT_FOR_CAPABILITY.get(capability, "coder")

    def _select_model(self, capability: str, agent: str,
                      task: Any = None) -> str:
        """Select a model for a capability using the router."""
        try:
            if self.adaptive_router:
                selection = self.adaptive_router.select(
                    required_capabilities=[capability],
                    role=agent,
                    task_id=task.metadata.id if task else "",
                )
            else:
                selection = self.router.select(
                    required_capabilities=[capability],
                    role=agent,
                )
            return selection.model_id
        except Exception:
            return "default:free"

    def _compute_dependencies(self, capability: str,
                               required_caps: List[str]) -> List[str]:
        """Compute dependency ordering between capabilities."""
        # Architecture comes first
        if capability in ("system_design", "api_design", "domain_modeling",
                          "integration_design"):
            return []

        # Implementation depends on architecture
        if capability in ("backend_development", "api_implementation",
                          "database_schema", "frontend_development",
                          "ui_implementation", "refactoring"):
            arch_caps = [c for c in required_caps if c in (
                "system_design", "api_design", "domain_modeling",
                "integration_design"
            )]
            return arch_caps

        # Testing depends on implementation
        if capability in ("testing", "unit_testing", "integration_testing",
                          "e2e_testing"):
            impl_caps = [c for c in required_caps if c in (
                "backend_development", "api_implementation",
                "database_schema", "frontend_development",
                "ui_implementation", "refactoring"
            )]
            return impl_caps

        # Security analysis depends on implementation
        if capability in ("security_analysis", "vulnerability_assessment"):
            impl_caps = [c for c in required_caps if c in (
                "backend_development", "api_implementation",
                "frontend_development", "ui_implementation"
            )]
            return impl_caps

        # Review depends on testing and security
        if capability in ("code_review", "review", "evaluation"):
            return [c for c in required_caps if c in (
                "testing", "unit_testing", "integration_testing",
                "security_analysis"
            )]

        # Verification depends on implementation
        if capability == "verification":
            return [c for c in required_caps if c in (
                "backend_development", "api_implementation",
            )]

        return []

    def _determine_branching(self, capability: str,
                              required_caps: List[str]) -> Tuple[str, str]:
        """Determine PASS/FAIL branching for a capability node."""
        # Review → complete on pass, block on fail
        if capability in ("code_review", "review"):
            return "complete", "block"

        # Verification → complete on pass, debugging on fail
        if capability == "verification":
            return "complete", "block"

        # Evaluation → complete on pass, debug on fail
        if capability == "evaluation":
            return "complete", "block"

        # Testing → review on pass, fail on fail
        if capability in ("testing", "unit_testing", "integration_testing",
                          "e2e_testing"):
            # Next is review/evaluation if present
            next_cap = None
            for c in ("review", "evaluation", "code_review"):
                if c in required_caps:
                    next_cap = c
                    break
            return (next_cap or "complete"), "block"

        # Security → review on pass, fail on fail
        if capability in ("security_analysis", "vulnerability_assessment"):
            next_cap = None
            for c in ("review", "evaluation", "code_review"):
                if c in required_caps:
                    next_cap = c
                    break
            return (next_cap or "complete"), "block"

        # Implementation → next capability on pass, fail on fail
        return "next", "block"

    def _determine_workspace(self, capability: str) -> Dict[str, List[str]]:
        """Determine workspace scope for a capability."""
        # Architecture/design: read-only
        if capability in ("system_design", "api_design", "domain_modeling",
                          "integration_design", "code_review"):
            return {"write": [], "read": ["src/**", "docs/**"]}

        # Implementation: write to src
        if capability in ("backend_development", "api_implementation",
                          "database_schema", "frontend_development",
                          "ui_implementation", "refactoring"):
            return {"write": ["src/**", "app/**"], "read": ["**/*"]}

        # Testing: write to tests
        if capability in ("testing", "unit_testing", "integration_testing",
                          "e2e_testing"):
            return {"write": ["tests/**"], "read": ["src/**", "app/**"]}

        # Security: read-only
        if capability in ("security_analysis", "vulnerability_assessment"):
            return {"write": [], "read": ["**/*"]}

        # Verification: read-only
        if capability == "verification":
            return {"write": [], "read": ["src/**", "tests/**"]}

        # Evaluation: read-only
        if capability == "evaluation":
            return {"write": [], "read": ["**/*"]}

        return {"write": [], "read": ["**/*"]}
