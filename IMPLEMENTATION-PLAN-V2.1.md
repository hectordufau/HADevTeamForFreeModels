# HADevTeamForFreeModels V2.1 — Autonomous Implementation Plan

**Version:** 2.1.0
**Status:** Ready for Implementation
**Runtime:** Hermes Agent
**Model Provider:** Nous Research
**Model Cost Policy:** Free / Cost = 0
**Specification:** SPEC-V2.md (V2.1 section)

---

## 1. Purpose

This document defines the ordered implementation tasks for V2.1 features:
adaptive model routing, performance registry, persistent memory, ADR automation,
and advanced evaluation.

## 2. Task Dependency Graph

```
TASK-016
   |
   v
TASK-017
   |
   v
TASK-018
   |
   +--------+--------+
   |                  |
   v                  v
TASK-019           TASK-020
   |                  |
   +--------+--------+
            |
            v
         TASK-021
            |
            v
         TASK-022
```

## 3. Task Definitions

### TASK-016 — Model Performance Registry

**Objective:** Implement SQLite-backed storage of model execution metrics.

**Dependencies:** TASK-006 (Model Router).

**Files to create/modify:**
- `harness/routing/performance.py` — ModelPerformanceRegistry class
- `artifacts/model_performance.db` — SQLite database (auto-created)

**Requirements:**
- Store per-task records: model_id, role, success, score, iterations, latency, capabilities
- Query: get_model_stats(model_id, role) → aggregated metrics
- Query: get_best_model_for_capability(capability, min_records) → best model
- Query: get_capability_leaderboard(min_records) → all models ranked

**Allowed Changes:** `harness/routing/**`, `artifacts/**`

**Acceptance Criteria:**
- [ ] Record stores and retrieves correctly
- [ ] Aggregated stats (total, successful, avg_score, avg_iterations)
- [ ] Best model for capability query returns correct model
- [ ] SQLite database created automatically

**Verification:** `tests/unit/test_v21.py::test_registry_record_and_stats`, `test_registry_best_for_capability`

---

### TASK-017 — Adaptive Model Router

**Objective:** Model Router that learns from historical performance data.

**Dependencies:** TASK-016.

**Files to create/modify:**
- `harness/routing/performance.py` — AdaptiveModelRouter class

**Requirements:**
- Blend capability-based scoring (70%) with historical success rates (30%)
- Normalize capability scores to 0-1 range
- Fall back gracefully when insufficient data (< min_records)
- Expose reasoning: why was this model selected?

**Allowed Changes:** `harness/routing/**`

**Acceptance Criteria:**
- [ ] With sufficient history, prefers model with better success rate
- [ ] Without history, falls back to capability-based routing
- [ ] Returns clear reasoning string

**Verification:** `tests/unit/test_v21.py::test_adaptive_router_selects_best`

---

### TASK-018 — Policy Engine

**Objective:** Implement the Policy Engine that enforces execution, autonomy, and tool policies.

**Dependencies:** TASK-002 (Configuration).

**Files to create:**
- `harness/policy/__init__.py` — PolicyEngine, AutonomyPolicy, ToolPolicy, ExecutionPolicy

**Requirements:**
- **AutonomyPolicy:** checks agent level vs task level vs tool level; most restrictive wins
- **ToolPolicy:** maps tool name → required autonomy level; validates tool access
- **ExecutionPolicy:** enforces max_iterations, required artifacts, scope boundaries
- **PolicyEngine:** composable policy evaluation: execute all policies, collect violations

**Allowed Changes:** `harness/policy/**`, `config/**`

**Acceptance Criteria:**
- [ ] AutonomyPolicy: effective = min(agent, task, tool)
- [ ] ToolPolicy: tool requires level 2 but agent has level 1 → blocked
- [ ] ExecutionPolicy: exceeded max_iterations → blocked
- [ ] PolicyEngine: collects all violations, returns structured result

**Verification:** Create `tests/unit/test_policy.py` with tests for each policy.

---

### TASK-019 — Workflow Engine

**Objective:** Implement the Workflow Engine that defines and executes agent pipelines.

**Dependencies:** TASK-008 (Agent Execution).

**Files to create:**
- `harness/workflow/__init__.py` — WorkflowEngine, WorkflowStep, WorkflowDefinition

**Requirements:**
- **WorkflowDefinition:** ordered list of steps, each with: role, capabilities, model_policy, tools
- **WorkflowStep:** role, required_capabilities, model_policy_ref, allowed_tools, autonomy_override
- **WorkflowEngine.execute(workflow, task):** iterate steps, for each: select agent → select model → build context → execute → collect result
- Support branching: on PASS → next step, on FAIL → feedback step

**Allowed Changes:** `harness/workflow/**`, `config/**`

**Acceptance Criteria:**
- [ ] Workflow definition loads from YAML
- [ ] Sequential execution of steps
- [ ] Branching on pass/fail
- [ ] Each step produces execution result

**Verification:** Create `tests/unit/test_workflow.py`.

---

### TASK-020 — Advanced Evaluation (Hard Minimums)

**Objective:** Evaluation engine with hard minimums for correctness and security.

**Dependencies:** TASK-012 (Evaluation).

**Files to create/modify:**
- `harness/evaluation/advanced.py` — AdvancedEvaluationEngine

**Requirements:**
- Hard minimums: correctness >= 0.90, security >= 0.90
- These dimensions CANNOT be compensated by other scores
- Security gate: if security < minimum → fail immediately, no iteration
- Integration with EvaluationEngine base class

**Allowed Changes:** `harness/evaluation/**`

**Acceptance Criteria:**
- [ ] Correctness below 0.90 produces FAIL regardless of other scores
- [ ] Security below 0.90 produces FAIL regardless of other scores
- [ ] Both at or above minimum → normal evaluation

**Verification:** Existing `tests/unit/test_v21.py` covers this.

---

### TASK-021 — Persistent Memory and ADR Automation

**Objective:** Multi-category memory system with automatic ADR generation.

**Dependencies:** TASK-007 (Context).

**Files to create/modify:**
- `harness/memory/__init__.py` — MemoryManager (4 categories)
- `harness/memory/adr.py` — ADRAutomation

**Requirements:**
- Categories: project, task, agent, lessons
- JSON persistence per entry
- Search by substring across categories
- ADR automation: store as both memory entry + docs/adr/*.md

**Allowed Changes:** `harness/memory/**`, `docs/adr/**`

**Acceptance Criteria:**
- [ ] Store and retrieve memory entries
- [ ] Search returns matching entries across categories
- [ ] ADR creates both memory entry and markdown file

**Verification:** Manual inspection of stored files.

---

### TASK-022 — Configuration and Workflow Integration

**Objective:** Wire all V2.1 components together via configuration.

**Dependencies:** TASK-016 through TASK-021.

**Files to modify:**
- `config/harness.yaml` — V2.1 settings
- `config/model_policy.yaml` — Capability-based policies per role
- `config/workflow_definitions/` — Default workflow YAML files

**Requirements:**
- harness.yaml: evaluation.hard_minimums, model_routing.adaptive, memory.*, registry.*
- model_policy.yaml: policies per role with capabilities list
- Default workflow: manager → architect → coder → tester → reviewer (sequential)
- Wire AdaptiveModelRouter into Orchestrator

**Allowed Changes:** `config/**`, `harness/orchestrator/**`

**Acceptance Criteria:**
- [ ] Config loads without errors
- [ ] Model policies map correctly to roles
- [ ] Default workflow is executable

**Verification:** `PYTHONPATH=. python3 -c "from harness.config import load_config; print(load_config())"`

---

## 4. Global Acceptance Test

After TASK-022:

```python
# Verify all V2.1 modules import correctly
from harness.routing.performance import ModelPerformanceRegistry, AdaptiveModelRouter
from harness.memory import MemoryManager
from harness.memory.adr import ADRAutomation
from harness.evaluation.advanced import AdvancedEvaluationEngine
print("All V2.1 modules loaded successfully")
```

## 5. Version 2.1 Definition of Done

- [ ] Model Performance Registry operational
- [ ] Adaptive Model Router operational
- [ ] Policy Engine operational (autonomy, tool, execution policies)
- [ ] Workflow Engine operational (step execution, branching)
- [ ] Advanced Evaluation operational (hard minimums)
- [ ] Persistent Memory operational (4 categories)
- [ ] ADR Automation operational
- [ ] Configuration integration complete
- [ ] All tests passing
