# HADevTeamForFreeModels V2.2 — Dynamic Capability-Driven AI Engineering Harness

**Status:** Proposed  
**Version:** 2.2.0  
**Runtime:** Hermes Agent  
**Model Provider:** Nous Research  
**Model Cost Constraint:** Free / Cost = 0  
**Previous:** [V2.1.0](https://github.com/hectordufau/HADevTeamForFreeModels/releases/tag/v2.1.0) — Adaptive AI Engineering Harness

---

## 1. Purpose

V2.2 transforms the Harness from a configurable-step workflow engine into a **Dynamic Capability-Driven AI Engineering Harness**.

The central change:

```
V2.1
Task → Workflow Definition → Fixed Steps → Agents

V2.2
Task → Task Analysis → Required Capabilities → Capability Graph
      → Workflow Planning → Workflow Validation → Dynamic Execution
```

Instead of asking **"What is the next step in the workflow?"**, V2.2 asks:  
**"What needs to be done next, and what capabilities are required?"**

---

## 2. Core Principle

```
CAPABILITY → AGENT → MODEL
```

Not:

```
FIXED ROLE → FIXED AGENT → MODEL
```

The Harness discovers what capabilities a task needs, selects agents that possess them, and assigns models dynamically — rather than following a predetermined role sequence.

---

## 3. High-Level Architecture

```
                         TASK
                           │
                           ▼
                    TASK ANALYZER
                           │
                           ▼
                 CAPABILITY DISCOVERY
                           │
                           ▼
                    CAPABILITY GRAPH
                           │
                           ▼
                    WORKFLOW PLANNER
                           │
                           ▼
                   WORKFLOW VALIDATOR
                           │
                    ┌──────┴──────┐
                    │             │
                  VALID         INVALID
                    │             │
                    ▼             ▼
             EXECUTION GRAPH   NEEDS_HUMAN
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
    AGENT A      AGENT B      AGENT C
       │            │            │
       └────────────┼────────────┘
                    ▼
             VERIFICATION
                    │
             EVALUATION
                    │
               REVIEWER
                    │
              ┌─────┴─────┐
              │           │
             PASS        FAIL
              │           │
              ▼           ▼
          COMPLETED    FAILURE ANALYSIS
                            │
                            ▼
                      WORKFLOW REPLAN
                            │
                            └──────────↺
```

---

## 4. Repository Structure (new V2.2 additions)

```
HADevTeamForFreeModels/
│
├── config/
│   └── capability_taxonomy.yaml    ← NEW: capability hierarchy
│
├── harness/
│   └── capabilities/               ← NEW: capability engine
│       ├── __init__.py
│       ├── registry.py             ← CapabilityRegistry (agent → capabilities)
│       ├── taxonomy.py             ← CapabilityTaxonomy (hierarchy loader)
│       ├── graph.py                ← CapabilityGraph (DAG dependencies)
│       └── matching.py             ← CapabilityMatching (score agents on capabilities)
│
├── harness/
│   └── planner/                    ← NEW: workflow planner
│       ├── __init__.py
│       ├── task_analyzer.py        ← Analyzes task → required capabilities
│       ├── workflow_planner.py     ← Builds DAG from capability graph
│       ├── workflow_validator.py   ← Validates DAG, policies, agents, models
│       ├── exec_graph.py           ← ExecutionGraph (parallel-safe DAG runner)
│       ├── failure_analyzer.py     ← Classifies failures → capability gaps
│       └── replanner.py            ← Adaptive replanning after failure
│
├── tests/
│   ├── unit/
│   │   ├── test_capability_registry.py
│   │   ├── test_capability_taxonomy.py
│   │   ├── test_capability_graph.py
│   │   ├── test_capability_matching.py
│   │   ├── test_task_analyzer.py
│   │   ├── test_workflow_planner.py
│   │   ├── test_workflow_validator.py
│   │   ├── test_exec_graph.py
│   │   ├── test_failure_analyzer.py
│   │   └── test_replanner.py
│   │
│   ├── integration/
│   │   └── test_v22_integration.py
│   │
│   └── e2e/
│       └── test_harness_v22.py
│
└── docs/
    └── adr/
        └── ADR-004-dynamic-workflow.md
```

---

## 5. Capability Registry

Each agent declares its capabilities formally in PROFILE.yaml.

### `agents/coder/PROFILE.yaml` (V2.2)

```yaml
name: FreeCoder

capabilities:
  - backend_development
  - frontend_development
  - api_implementation
  - refactoring
  - database_schema

skills:
  - php/laravel
  - go/chi
  - rust/axum
  - python/fastapi

tools:
  - filesystem
  - terminal
  - git

autonomy:
  level: 3
```

### `harness/capabilities/registry.py`

```python
class CapabilityRegistry:
    """Maps agents to their declared capabilities."""

    def register_agent(self, role: str, capabilities: List[str]):
        ...

    def find_agents_for(self, capability: str) -> List[str]:
        """Find agents that declare this capability."""
        ...

    def find_best_agent(self, required_capabilities: List[str]) -> str:
        """Find the agent best matching a set of required capabilities."""
        ...
```

---

## 6. Capability Taxonomy

A hierarchical taxonomy enables matching by parent capability.

### `config/capability_taxonomy.yaml`

```yaml
engineering:
  architecture:
    system_design: {}
    api_design: {}
    integration_design: {}
    domain_modeling: {}

  implementation:
    backend:
      backend_development: {}
      api_implementation: {}
      database_schema: {}
    frontend:
      frontend_development: {}
      ui_implementation: {}
    refactoring: {}

  quality:
    testing:
      unit_testing: {}
      integration_testing: {}
      e2e_testing: {}
    security:
      security_analysis: {}
      vulnerability_assessment: {}
    performance:
      profiling: {}
      optimization: {}

  delivery:
    documentation:
      technical_writing: {}
      api_documentation: {}
    release:
      ci_cd: {}
      deployment: {}
```

### `harness/capabilities/taxonomy.py`

```python
class CapabilityTaxonomy:
    """Loads and queries the capability hierarchy."""

    def get_children(self, capability: str) -> List[str]:
        """Get more specific capabilities under this one."""
        ...

    def get_parents(self, capability: str) -> List[str]:
        """Get broader capabilities that encompass this one."""
        ...

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Check if capability A is an ancestor of capability B."""
        ...
```

---

## 7. Capability Graph

Represents dependencies between capabilities as a DAG.

### `harness/capabilities/graph.py`

```python
class CapabilityNode:
    id: str
    capability: str
    dependencies: List[str]  # capability IDs this node depends on
    status: str              # pending, running, completed, failed, skipped
    result: Optional[dict]

class CapabilityGraph:
    """DAG of capability nodes with dependency resolution."""

    def add_node(self, node: CapabilityNode):
        ...

    def get_ready_nodes(self) -> List[CapabilityNode]:
        """Nodes whose dependencies are all satisfied."""
        ...

    def has_cycle(self) -> bool:
        """Detect cycles using DFS."""
        ...

    def topological_sort(self) -> List[CapabilityNode]:
        """Return nodes in execution order."""
        ...
```

Example graph for an API feature:

```
api_design
     │
     ▼
backend_implementation
     │
 ┌───┴────┐
 ▼        ▼
testing  security
 │        │
 └───┬────┘
     ▼
   review
```

---

## 8. Capability Matching

Scores agents against required capabilities.

### `harness/capabilities/matching.py`

```python
@dataclass
class MatchResult:
    agent: str
    capability: str
    score: float          # 0.0 to 1.0
    match_type: str       # exact, parent, sibling, ancestor

class CapabilityMatcher:
    """Finds best agent(s) for required capabilities."""

    def match(self, required: List[str], agent_registry: CapabilityRegistry,
              taxonomy: CapabilityTaxonomy) -> List[MatchResult]:
        """Score all agents against required capabilities."""
        ...

    def best_match(self, required: List[str], ...) -> MatchResult:
        """Return single best agent for a capability set."""
        ...
```

Matching priorities:

| Match Type | Score | Description |
|---|---|---|
| `exact` | 1.0 | Agent declares exact capability |
| `child` | 0.9 | Agent declares a sub-capability |
| `parent` | 0.7 | Agent declares parent capability |
| `sibling` | 0.4 | Agent declares sibling capability under same parent |
| `none` | 0.0 | No match |

---

## 9. Task Analyzer

Analyzes a Task Contract and produces capability requirements.

### `harness/planner/task_analyzer.py`

```python
class TaskAnalyzer:
    """Analyzes a task contract → required capabilities."""

    def analyze(self, task: TaskContract) -> CapabilityRequirements:
        """
        Produces a structured set of required capabilities.
        The LLM proposes; the Harness validates.
        """
        ...

    def validate_proposal(self, proposal: dict) -> CapabilityRequirements:
        """Validate LLM capability proposal against taxonomy and policy."""
        ...
```

### Output

```yaml
capability_requirements:
  required:
    - api_design
    - backend_development
    - testing

  optional:
    - security_analysis

  constraints:
    minimum_coverage: 1.0  # all required must be covered

  agent_preferences:
    - role: coder
      capabilities:
        - backend_development
```

---

## 10. Workflow Planner

Builds an execution DAG from capability requirements.

### `harness/planner/workflow_planner.py`

```python
class WorkflowPlanner:
    """Builds an ExecutionWorkflow DAG from capability requirements."""

    def plan(self, requirements: CapabilityRequirements,
             agent_registry: CapabilityRegistry,
             taxonomy: CapabilityTaxonomy,
             model_catalog: ModelCatalog) -> ExecutionWorkflow:
        """
        1. Look up required capabilities
        2. Find agents for each capability
        3. Build dependency graph
        4. Assign models via AdaptiveModelRouter
        5. Return ExecutionWorkflow
        """
        ...

    def plan_from_task(self, task: TaskContract, ...) -> ExecutionWorkflow:
        """Analyze task + plan in one call."""
        ...
```

### Output

```yaml
workflow_plan:
  nodes:
    - id: design
      capability: api_design
      agent: architect
      model: poolside/laguna-s-2.1:free
      depends_on: []

    - id: implementation
      capability: backend_development
      agent: coder
      model: meituan/longcat-2.0:free
      depends_on:
        - design

    - id: tests
      capability: testing
      agent: tester
      model: stepfun/step-3.7-flash:free
      depends_on:
        - implementation

    - id: security
      capability: security_analysis
      agent: reviewer
      model: poolside/laguna-s-2.1:free
      depends_on:
        - implementation

    - id: review
      capability: code_review
      agent: reviewer
      model: poolside/laguna-s-2.1:free
      depends_on:
        - tests
        - security
```

---

## 11. Workflow Validator

Validates a planned workflow before execution.

### `harness/planner/workflow_validator.py`

```python
class WorkflowValidator:
    """Validates a workflow plan before execution."""

    def validate(self, workflow: ExecutionWorkflow,
                 task: TaskContract,
                 config: dict) -> ValidationResult:
        """
        Checks:
        - Valid DAG (no cycles)
        - All required capabilities covered
        - Mandatory capabilities present (verification, evaluation, review)
        - Agents available
        - Agents authorized (policy)
        - Model policy valid
        - Models are free
        - Dependencies satisfiable
        - Workspace conflicts absent
        - Termination path exists
        - Reviewer present and independent
        """
        ...

    def validate_dag(self, graph: CapabilityGraph) -> bool:
        """Reject cycles."""
        ...

    def validate_mandatory_capabilities(self, caps: List[str]) -> bool:
        """verification, evaluation, review must always be present."""
        ...
```

### Validation Error Codes

| Code | Description |
|---|---|
| `CYCLE_DETECTED` | Workflow graph contains a cycle |
| `MISSING_CAPABILITY` | Required capability has no agent |
| `MISSING_MANDATORY` | verification/evaluation/review not included |
| `AGENT_UNAVAILABLE` | Selected agent not available |
| `PAID_MODEL` | Selected model violates free-only policy |
| `WORKSPACE_CONFLICT` | Two parallel agents write to same files |
| `NO_TERMINATION` | No path leads to completion |

---

## 12. Execution Graph

Parallel-safe DAG runner.

### `harness/planner/exec_graph.py`

```python
class ExecutionGraph:
    """Executes a workflow DAG with parallel nodes."""

    def __init__(self, workflow: ExecutionWorkflow, orchestrator: Orchestrator):
        ...

    async def execute(self) -> Dict[str, Any]:
        """
        Execute nodes in dependency order.
        Independent nodes run in parallel.
        Returns aggregated results.
        """
        ...

    def _execute_node(self, node: WorkflowNode) -> ExecutionResult:
        """
        Execute a single workflow node:
        1. Policy check
        2. Agent execution
        3. Verification (if applicable)
        4. Evidence collection
        """
        ...
```

Parallel execution rules:

- Nodes without dependencies on each other execute concurrently
- Each node gets its own workspace scope (files it can write)
- Concurrent nodes cannot write to overlapping file paths
- Evidence is collected per-node, then merged

---

## 13. Workspace Ownership

Prevents file conflicts during parallel execution.

```yaml
node: backend_implementation
workspace:
  write:
    - src/backend/**

node: testing
workspace:
  write:
    - tests/**

node: security_analysis
workspace:
  write: []  # read-only
```

Conflict detection:

```text
Agent A writes to src/service.py
Agent B writes to src/service.py
       ↓
WORKSPACE_CONFLICT → BLOCKED
```

### `harness/planner/workspace.py`

```python
class WorkspaceManager:
    """Manages file ownership during parallel execution."""

    def acquire(self, node_id: str, files: List[str]) -> bool:
        """Try to acquire write access to files. Returns False on conflict."""
        ...

    def release(self, node_id: str):
        """Release all files owned by a node."""
        ...

    def get_conflicts(self) -> List[Conflict]:
        """Return all current workspace conflicts."""
        ...
```

---

## 14. Conditional Branching

Workflow nodes can branch based on results.

```yaml
node: tests
capability: testing
on:
  PASS: review
  FAIL: debugging

node: debugging
capability: debugging
depends_on:
  - tests
on:
  PASS: implementation  # re-implement after debug
  FAIL: needs_human
```

Conditions must be declared in the workflow plan. The LLM cannot create arbitrary branches outside the planner.

### `harness/planner/branching.py`

```python
class BranchResolver:
    """Resolves conditional branches in workflow execution."""

    def resolve(self, node: WorkflowNode, result: ExecutionResult) -> str:
        """Returns the next node ID based on result."""
        ...
```

---

## 15. Failure Analysis

When a node fails, classify the root cause.

### `harness/planner/failure_analyzer.py`

```python
class FailureAnalyzer:
    """Classifies failures and identifies capability gaps."""

    def analyze(self, node: WorkflowNode, result: ExecutionResult,
                evidence: Evidence) -> FailureReport:
        """
        Produces a failure report with:
        - Root cause category
        - Missing capabilities
        - Recommendations
        """
        ...

    def classify(self, error: str, evidence: Evidence) -> str:
        """Map error patterns to failure categories."""
        ...
```

### Failure Categories

| Category | Example |
|---|---|
| `implementation` | Logic error, API misuse |
| `test` | Test assertion failed |
| `architecture` | Wrong pattern, coupling issue |
| `security` | Vulnerability detected |
| `integration` | API contract mismatch |
| `configuration` | Missing env var, wrong setting |
| `environment` | Dependency missing, version conflict |

---

## 16. Adaptive Replanning

After failure analysis, replan the remaining workflow.

### `harness/planner/replanner.py`

```python
class Replanner:
    """Adaptive replanning after execution failure."""

    def replan(self, task: TaskContract, current_workflow: ExecutionWorkflow,
               completed_nodes: List[str], failed_node: str,
               failure_report: FailureReport) -> ExecutionWorkflow:
        """
        Create a new workflow for remaining steps.
        Can only modify PENDING nodes, never COMPLETED.
        """
        ...

    def generate_replan_report(self, original: ExecutionWorkflow,
                                new: ExecutionWorkflow,
                                reason: str) -> dict:
        """Document why and how the workflow changed."""
        ...
```

### Replan Rules

- Only PENDING nodes can be modified
- COMPLETED nodes are frozen (cannot be re-executed)
- COMPLETED evidence is preserved
- Each replan generates `workflow_replan.yaml`
- Maximum replan count: configurable (default 3)

---

## 17. Mandatory Final Capabilities

Regardless of task complexity, these capabilities are always required:

```yaml
mandatory_capabilities:
  - verification
  - evaluation
  - review
```

The WorkflowValidator enforces this before any execution begins. This prevents the planner from creating a workflow like `Implement → Complete` and skipping the V2.1 gates.

---

## 18. New Artifacts (V2.2 additions)

In addition to the 12 V2.1 artifacts:

```
artifacts/executions/TASK-XXXX/
├── capability_requirements.yaml    ← from TaskAnalyzer
├── capability_graph.yaml           ← from planner
├── workflow_plan.yaml              ← from WorkflowPlanner
├── workflow_validation.yaml        ← from WorkflowValidator
├── execution_graph.yaml            ← from ExecutionGraph
├── workspace_ownership.yaml        ← per-node file ownership
├── failure_analysis.yaml           ← from FailureAnalyzer (on failure only)
└── workflow_replan.yaml            ← from Replanner (on replan only)
```

Only applicable artifacts need to exist per execution.

---

## 19. Invariants (inherited from V2.1)

Nothing in V2.2 may weaken:

- ✅ Lifecycle gates (G1)
- ✅ Independent reviewer (G2)
- ✅ Adaptive model routing (G3)
- ✅ Policy enforcement (G4)
- ✅ Workflow bypass prevention (G5)
- ✅ Artifact integrity (G6)
- ✅ Crash recovery (G7)
- ✅ Idempotence (G8)
- ✅ Evidence integrity (G9)
- ✅ Free-model invariant (G10)

> **V2.2 adds dynamism to the workflow, not to the security rules.**

---

## 20. V2.2 Release Gates

### DG1 — Capability Coverage

All required capabilities have an agent selected. Optional capabilities are best-effort.

### DG2 — DAG Integrity

Cycles are rejected. Every node has a valid dependency path to completion.

### DG3 — Parallel Execution

Independent nodes execute correctly in parallel without data races.

### DG4 — Dependency Integrity

Node B does not start before node A if B depends on A.

### DG5 — Workspace Conflict

Concurrent writes to the same file paths are blocked.

### DG6 — Conditional Branch

PASS/FAIL follows only authorized branches declared in the plan.

### DG7 — Adaptive Replanning

Failure → analysis → capability gap → replan → recovery. Completed nodes are never re-executed.

### DG8 — Completed-Node Preservation

Nodes marked COMPLETED are frozen. Replan cannot modify them.

### DG9 — Mandatory Gates Preserved

Even dynamic workflows must include: verification, evaluation, review.

### DG10 — V2.1 Regression

All 10 V2.1 Release Gates continue to pass with V2.2 code.

---

## 21. Definition of Done

V2.2 is complete only when:

- [ ] Dynamic Capability Discovery operational
- [ ] Capability Registry operational
- [ ] Capability Taxonomy operational
- [ ] Capability Graph operational (DAG)
- [ ] Capability Matching operational
- [ ] Task Analyzer operational
- [ ] Workflow Planner operational (DAG builder)
- [ ] Workflow Validator operational (10 checks)
- [ ] Execution Graph operational (parallel DAG runner)
- [ ] Workspace Ownership operational (conflict prevention)
- [ ] Conditional Branching operational
- [ ] Failure Analysis operational
- [ ] Adaptive Replanning operational
- [ ] All V2.2 Release Gates passing (DG1–DG10)
- [ ] All V2.1 Release Gates passing (G1–G10 regression)
- [ ] All unit, integration, and E2E tests passing
- [ ] V2.1 artifacts preserved (12 per execution)
- [ ] New V2.2 artifacts created as specified
