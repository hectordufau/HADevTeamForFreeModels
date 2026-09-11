# HADevTeamForFreeModels V2.2 — Autonomous Implementation Plan

**Version:** 2.2.0  
**Status:** Ready for Implementation  
**Runtime:** Hermes Agent  
**Model Provider:** Nous Research  
**Model Cost Policy:** Free / Cost = 0  
**Specification:** SPEC-V2.2.md  
**Previous:** [V2.1.0](https://github.com/hectordufau/HADevTeamForFreeModels/releases/tag/v2.1.0)

---

## 1. Purpose

This document defines the ordered implementation tasks to build the Dynamic Capability-Driven Workflow engine.

The implementation is divided into 15 tasks (TASK-023 to TASK-037), each with defined acceptance criteria and verification steps.

---

## 2. Execution Rules (same as V2.1)

1. **Read Before Modify** — read SPEC-V2.2.md, IMPLEMENTATION-PLAN-V2.2.md, README.md
2. **Respect Task Boundaries** — only modify `allowed_changes` files
3. **Verify Before Continuing** — all acceptance criteria must pass
4. **Preserve Existing Behavior** — V2.1 must remain operational
5. **No Paid Models** — provider=nous, cost=0 invariant
6. **No Uncontrolled Refactoring** — defer architectural problems to DEFERRED-*.md
7. **Human Escalation** — stop if task cannot be safely completed

---

## 3. Task Dependency Graph

```
TASK-023 (Capability Registry)
    |
    v
TASK-024 (Capability Taxonomy)
    |
    v
TASK-025 (Capability Graph)
    |
    +----------+-----------+
    |                      |
    v                      v
TASK-026 (Matching)    TASK-027 (Task Analyzer)
    |                      |
    +----------+-----------+
               |
               v
        TASK-028 (Workflow Planner)
               |
               v
        TASK-029 (Workflow Validator)
               |
               v
        TASK-030 (Execution Graph)
               |
               v
        TASK-031 (Workspace Ownership)
               |
               v
        TASK-032 (Conditional Branching)
               |
               v
        TASK-033 (Failure Analysis)
               |
               v
        TASK-034 (Adaptive Replanning)
               |
               v
        TASK-035 (Orchestrator Integration)
               |
               v
        TASK-036 (V2.2 Tests)
               |
               v
        TASK-037 (Release Gates + Regression)
```

---

## 4. Task Definitions

### TASK-023 — Capability Registry

**Objective:** Implement agent capability declaration and lookup.

**Dependencies:** TASK-005 (Agent Loader).

**Files to create:**
- `harness/capabilities/__init__.py`
- `harness/capabilities/registry.py` — CapabilityRegistry class

**Requirements:**
- Parse capabilities from agent PROFILE.yaml
- `register_agent(role, capabilities)` — index agent
- `find_agents_for(capability)` → list of agent roles
- `find_best_agent(required_capabilities)` → single best agent
- Support scoring: exact=1.0, child=0.9, parent=0.7, sibling=0.4

**Allowed Changes:** `harness/capabilities/**`, `agents/**/PROFILE.yaml`

**Acceptance Criteria:**
- [ ] Register agent with capabilities
- [ ] Find agent by exact capability
- [ ] Score and rank agents by capability match
- [ ] Return empty list when no agent matches

**Verification:** `tests/unit/test_capability_registry.py`

---

### TASK-024 — Capability Taxonomy

**Objective:** Implement hierarchical capability classification.

**Dependencies:** TASK-023.

**Files to create/modify:**
- `config/capability_taxonomy.yaml` — hierarchy definition
- `harness/capabilities/taxonomy.py` — CapabilityTaxonomy class

**Requirements:**
- Load YAML taxonomy definition
- `get_children(capability)` → more specific caps
- `get_parents(capability)` → broader caps
- `is_ancestor(ancestor, descendant)` → boolean
- `get_siblings(capability)` → same-level caps

**Allowed Changes:** `config/**`, `harness/capabilities/**`

**Acceptance Criteria:**
- [ ] Load taxonomy from YAML
- [ ] Navigate hierarchy (parent/child/sibling)
- [ ] Ancestor detection works correctly

**Verification:** `tests/unit/test_capability_taxonomy.py`

---

### TASK-025 — Capability Graph

**Objective:** Implement DAG-based capability dependency graph.

**Dependencies:** TASK-023.

**Files to create:**
- `harness/capabilities/graph.py` — CapabilityNode, CapabilityGraph

**Requirements:**
- `add_node(id, capability, dependencies)`
- `get_ready_nodes()` — nodes whose deps are satisfied
- `has_cycle()` — DFS cycle detection
- `topological_sort()` — execution order
- `mark_completed(node_id)` — update status

**Allowed Changes:** `harness/capabilities/**`

**Acceptance Criteria:**
- [ ] Create DAG with dependencies
- [ ] Detect cycles (A→B→C→A)
- [ ] Topological sort returns valid order
- [ ] Ready nodes correctly computed
- [ ] Serialization to/from dict

**Verification:** `tests/unit/test_capability_graph.py`

---

### TASK-026 — Capability Matching

**Objective:** Score agents against required capabilities.

**Dependencies:** TASK-023, TASK-024.

**Files to create:**
- `harness/capabilities/matching.py` — MatchResult, CapabilityMatcher

**Requirements:**
- `match(required_caps, registry, taxonomy)` → all matches
- `best_match(required_caps, ...)` → single best
- Scoring: exact=1.0, child=0.9, parent=0.7, sibling=0.4
- Handle multiple required capabilities (aggregate score)

**Allowed Changes:** `harness/capabilities/**`

**Acceptance Criteria:**
- [ ] Exact match scores 1.0
- [ ] Child match scores 0.9
- [ ] Parent match scores 0.7
- [ ] No match scores 0.0
- [ ] Aggregate score across multiple capabilities

**Verification:** `tests/unit/test_capability_matching.py`

---

### TASK-027 — Task Analyzer

**Objective:** Analyze Task Contract → required capabilities.

**Dependencies:** TASK-003 (Task Contract), TASK-024 (Taxonomy).

**Files to create:**
- `harness/planner/__init__.py`
- `harness/planner/task_analyzer.py` — TaskAnalyzer, CapabilityRequirements

**Requirements:**
- Analyze task objective → extract keywords/phrases
- Map to taxonomy capabilities
- Separate required vs optional
- Validate proposal against taxonomy
- Respect policy constraints (mandatory caps)

**Allowed Changes:** `harness/planner/**`, `config/**`

**Acceptance Criteria:**
- [ ] Extract capabilities from task objective
- [ ] Separate required vs optional
- [ ] Validate against taxonomy
- [ ] Always include mandatory caps (verification, evaluation, review)

**Verification:** `tests/unit/test_task_analyzer.py`

---

### TASK-028 — Workflow Planner

**Objective:** Build execution DAG from capability requirements.

**Dependencies:** TASK-025, TASK-026, TASK-027.

**Files to create:**
- `harness/planner/workflow_planner.py` — WorkflowPlanner, ExecutionWorkflow, WorkflowNode

**Requirements:**
- Receive CapabilityRequirements
- Use CapabilityMatcher to find agents
- Use AdaptiveModelRouter to assign models
- Build CapabilityGraph from dependencies
- Return ExecutionWorkflow (ordered nodes with agent+model assignments)

**Allowed Changes:** `harness/planner/**`

**Acceptance Criteria:**
- [ ] Plan from requirements produces valid DAG
- [ ] Each node has agent + model assigned
- [ ] Mandatory caps present
- [ ] Agents matched to capabilities

**Verification:** `tests/unit/test_workflow_planner.py`

---

### TASK-029 — Workflow Validator

**Objective:** Validate workflow plan before execution.

**Dependencies:** TASK-028.

**Files to create:**
- `harness/planner/workflow_validator.py` — WorkflowValidator, ValidationResult

**Requirements:**
- Check: DAG validity (no cycles)
- Check: All required capabilities covered
- Check: Mandatory capabilities present
- Check: Agents available and authorized
- Check: Models are free (cost=0)
- Check: Workspace conflicts absent
- Check: Termination path exists
- Check: Reviewer present and independent

**Allowed Changes:** `harness/planner/**`

**Acceptance Criteria:**
- [ ] Valid workflow passes all checks
- [ ] Invalid workflow (missing cap) fails
- [ ] Workflow without mandatory caps fails
- [ ] Workflow with paid model fails
- [ ] Workflow without reviewer fails

**Verification:** `tests/unit/test_workflow_validator.py`

---

### TASK-030 — Execution Graph

**Objective:** Parallel-safe DAG runner.

**Dependencies:** TASK-008 (Agent Execution), TASK-028, TASK-029.

**Files to create:**
- `harness/planner/exec_graph.py` — ExecutionGraph

**Requirements:**
- Execute nodes in dependency order
- Independent nodes run in parallel
- Each node: policy check → execute → verify → evidence
- Collect per-node results, merge into final report
- Respect workspace ownership during parallel execution

**Allowed Changes:** `harness/planner/**`

**Acceptance Criteria:**
- [ ] Sequential dependencies execute in order
- [ ] Independent nodes execute concurrently
- [ ] Each node produces execution result
- [ ] Results aggregated correctly
- [ ] Evidence per node merged

**Verification:** `tests/unit/test_exec_graph.py`

---

### TASK-031 — Workspace Ownership

**Objective:** Prevent file conflicts during parallel execution.

**Dependencies:** TASK-030.

**Files to create:**
- `harness/planner/workspace.py` — WorkspaceManager, Conflict

**Requirements:**
- `acquire(node_id, files)` → bool (False = conflict)
- `release(node_id)` — free owned files
- `get_conflicts()` → list of current conflicts
- Block on write conflicts between parallel nodes
- Read access is not restricted

**Allowed Changes:** `harness/planner/**`

**Acceptance Criteria:**
- [ ] Acquire non-conflicting files succeeds
- [ ] Acquire conflicting files fails
- [ ] Release frees files for other nodes
- [ ] Conflict reporting works

**Verification:** `tests/unit/test_workspace.py` (new)

---

### TASK-032 — Conditional Branching

**Objective:** Support PASS/FAIL branching in workflow.

**Dependencies:** TASK-030.

**Files to create:**
- `harness/planner/branching.py` — BranchResolver

**Requirements:**
- Parse `on: { PASS: next, FAIL: other }` from workflow node
- After node execution, resolve branch based on result
- Only authorized branches (declared in plan) are allowed
- Undeclared branches → BLOCKED

**Allowed Changes:** `harness/planner/**`

**Acceptance Criteria:**
- [ ] PASS → declared target
- [ ] FAIL → declared fallback
- [ ] Undeclared branch → BLOCKED

**Verification:** `tests/unit/test_branching.py` (new)

---

### TASK-033 — Failure Analysis

**Objective:** Classify execution failures and identify capability gaps.

**Dependencies:** TASK-030.

**Files to create:**
- `harness/planner/failure_analyzer.py` — FailureAnalyzer, FailureReport

**Requirements:**
- `analyze(node, result, evidence)` → FailureReport
- `classify(error, evidence)` → failure category
- Categories: implementation, test, architecture, security, integration, configuration, environment
- Recommend missing capabilities based on failure

**Allowed Changes:** `harness/planner/**`

**Acceptance Criteria:**
- [ ] Known error patterns map to categories
- [ ] Unknown errors map to generic
- [ ] Evidence contributes to classification
- [ ] Recommendations generated

**Verification:** `tests/unit/test_failure_analyzer.py`

---

### TASK-034 — Adaptive Replanning

**Objective:** Replan workflow after failure, preserving completed nodes.

**Dependencies:** TASK-032, TASK-033.

**Files to create:**
- `harness/planner/replanner.py` — Replanner

**Requirements:**
- `replan(task, current_workflow, completed, failed, report)` → new workflow
- Only PENDING nodes can be modified
- COMPLETED nodes are frozen
- Evidence from completed nodes is preserved
- Generate `workflow_replan.yaml` artifact

**Allowed Changes:** `harness/planner/**`

**Acceptance Criteria:**
- [ ] Replan produces valid new workflow
- [ ] Completed nodes unchanged
- [ ] Pending nodes may be modified
- [ ] Replan artifact generated
- [ ] Maximum replan count enforced

**Verification:** `tests/unit/test_replanner.py`

---

### TASK-035 — Orchestrator Integration

**Objective:** Wire V2.2 planner into the Orchestrator.

**Dependencies:** TASK-028 through TASK-034.

**Files to modify:**
- `harness/orchestrator/__init__.py` — integrate planner pipeline

**Requirements:**
- Orchestrator.run() uses WorkflowPlanner instead of hard-coded steps
- TaskAnalyzer runs first → CapabilityRequirements
- WorkflowPlanner builds DAG → ExecutionWorkflow
- WorkflowValidator validates pre-execution
- ExecutionGraph executes the DAG
- FailureAnalyzer + Replanner handle failures
- All V2.2 artifacts saved

**Allowed Changes:** `harness/orchestrator/**`, `harness/planner/**`

**Acceptance Criteria:**
- [ ] Orchestrator uses planner pipeline
- [ ] V2.1 fallback still works (config flag)
- [ ] All V2.2 artifacts created
- [ ] V2.1 regression: old tests still pass

**Verification:** `tests/integration/test_v22_integration.py`

---

### TASK-036 — V2.2 Tests

**Objective:** Comprehensive test suite for all V2.2 modules.

**Dependencies:** TASK-023 through TASK-034.

**Files to create:**
- `tests/unit/test_capability_registry.py`
- `tests/unit/test_capability_taxonomy.py`
- `tests/unit/test_capability_graph.py`
- `tests/unit/test_capability_matching.py`
- `tests/unit/test_task_analyzer.py`
- `tests/unit/test_workflow_planner.py`
- `tests/unit/test_workflow_validator.py`
- `tests/unit/test_exec_graph.py`
- `tests/unit/test_workspace.py`
- `tests/unit/test_branching.py`
- `tests/unit/test_failure_analyzer.py`
- `tests/unit/test_replanner.py`

**Requirements:** Each test file must have minimum 3 tests covering happy path, edge cases, and failure modes.

**Verification:** All 12 test files pass:

---

### TASK-037 — Release Gates + Regression

**Objective:** Implement V2.2 release gates and verify V2.1 regression.

**Dependencies:** TASK-035, TASK-036.

**Files to create/modify:**
- `tests/integration/test_v22_release_gate.py` — DG1–DG10
- `tests/e2e/test_harness_v22.py` — E2E with dynamic workflow

**Requirements:**
- DG1–DG10 implemented and passing
- V2.1 regression (G1–G10) verified
- E2E test with full dynamic workflow

**Verification:** All gates pass, all tests pass, tag `v2.2.0`.

---

## 5. Definition of Done

- [ ] All 15 tasks (TASK-023 to TASK-037) implemented
- [ ] All unit tests passing (12 files)
- [ ] All integration tests passing (V2.1 + V2.2)
- [ ] All E2E tests passing
- [ ] V2.2 Release Gates (DG1–DG10) passing
- [ ] V2.1 Release Gates (G1–G10) regression passing
- [ ] Tag `v2.2.0` created
