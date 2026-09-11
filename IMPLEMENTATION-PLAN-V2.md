# HADevTeamForFreeModels V2.0 — Autonomous Implementation Plan

**Version:** 2.0.0  
**Status:** Ready for Implementation  
**Target Runtime:** Hermes Agent  
**Model Provider:** Nous Research  
**Model Cost Policy:** Free / Cost = 0  
**Specification:** SPEC-V2.md

---

## 1. Purpose

This document defines the ordered implementation tasks required to transform HADevTeamForFreeModels into an AI Engineering Harness as specified by SPEC-V2.md.

The implementation is divided into independent, verifiable tasks. The Hermes Agent MUST execute tasks sequentially unless a dependency explicitly allows parallel execution.

---

## 2. Autonomous Execution Rules

### Rule 1 — Read Before Modify
Before executing any task, the agent MUST read: SPEC-V2.md, IMPLEMENTATION-PLAN-V2.md, README.md, and all files listed in the task's relevant_files.

### Rule 2 — Respect Task Boundaries
The agent MUST modify only files permitted by `allowed_changes`. The agent MUST NOT modify files listed under `forbidden_changes` unless the task explicitly authorizes it.

### Rule 3 — Verify Before Continuing
A task MUST NOT be considered complete until all mandatory acceptance criteria pass.

### Rule 4 — Preserve Existing Behavior
Existing V1 functionality MUST remain operational unless a task explicitly replaces it.

### Rule 5 — No Paid Models
The implementation MUST preserve: `provider = Nous`, `cost = 0`. No paid model may become a mandatory dependency.

### Rule 6 — No Uncontrolled Refactoring
The agent MUST NOT perform unrelated refactoring. If an architectural problem outside the current task is discovered, create `artifacts/decisions/DEFERRED-*.md` and continue only if the current task remains valid.

### Rule 7 — Human Escalation
The agent MUST stop and request human intervention if:
- a task cannot be completed without violating its constraints;
- an existing behavior would be broken;
- a security issue is discovered;
- a required external dependency is unavailable;
- a decision would change the architecture defined in SPEC-V2.md;
- the maximum retry count is reached.

---

## 3. Task Execution Protocol

Every task follows: `READ → ANALYZE → PLAN → IMPLEMENT → TEST → VERIFY → REVIEW → COMMIT ARTIFACTS → NEXT TASK`

If verification fails: `VERIFY → FAIL → ANALYZE FAILURE → FIX → VERIFY`

Maximum implementation retries: **3**. After three unsuccessful attempts: `NEEDS_HUMAN`.

---

## 4. Task Dependency Graph

```
TASK-001
   |
   v
TASK-002
   |
   v
TASK-003
   |
   +----------------+
   |                |
   v                v
TASK-004          TASK-005
   |                |
   +-------+--------+
           |
           v
        TASK-006
           |
           v
        TASK-007
           |
           v
        TASK-008
           |
           v
        TASK-009
           |
           v
        TASK-010
           |
           v
        TASK-011
           |
           v
        TASK-012
           |
           v
        TASK-013
           |
           v
        TASK-014
           |
           v
        TASK-015
```

---

## 5. Task Definitions

### TASK-001 — Repository Baseline and V2 Foundation

**Objective:** Create the minimum V2 directory structure and establish a clean implementation baseline without changing existing V1 behavior.

**Dependencies:** None.

**Requirements:**
- Create directories: `config/`, `agents/`, `team/`, `harness/`, `tasks/`, `artifacts/`, `skills/`, `tests/`, `docs/`
- Create `config/harness.yaml` with basic V2 configuration

**Allowed Changes:** `config/**`, `harness/**`, `tasks/**`, `artifacts/**`, `agents/**`, `team/**`, `skills/**`, `tests/**`, `docs/**`, `README.md`

**Forbidden Changes:** Existing V1 implementation logic, `setup.py` (unless required for package discovery)

**Acceptance Criteria:**
- [ ] V2 directory structure exists
- [ ] Python package structure is importable
- [ ] Existing V1 behavior remains functional
- [ ] `harness.yaml` loads successfully
- [ ] No existing V1 tests fail

**Verification:** `python -m compileall .` and existing project test suite.

---

### TASK-002 — Configuration and Schema Layer

**Objective:** Implement configuration loading and validation for V2.

**Dependencies:** TASK-001.

**Requirements:**
- Implement `harness/config/` or equivalent configuration module
- Support sections: `harness`, `models`, `execution`, `autonomy`, `workspace`, `human_approval`
- Validate required fields, types, supported values, model cost constraint

**Acceptance Criteria:**
- [ ] Invalid configuration MUST produce a clear error
- [ ] Valid configuration MUST load into a typed internal representation
- [ ] Loader MUST reject configurations that violate `models.cost.maximum = 0`

**Verification:** Create tests for valid config, missing fields, invalid values, invalid cost policy.

---

### TASK-003 — Task Contract Engine

**Objective:** Implement the Task Contract defined in SPEC-V2.md.

**Dependencies:** TASK-002.

**Requirements:**
- Support: `apiVersion`, `kind`, `metadata`, `spec` including objective, requirements, constraints, acceptance_criteria, allowed_changes, forbidden_changes, verification, autonomy
- Implement `TaskManager.create()`, `TaskManager.load()`, `TaskManager.validate()`, `TaskManager.update_state()`

**Acceptance Criteria:**
- [ ] Load valid task YAML
- [ ] Reject invalid tasks
- [ ] Generate unique task ID when requested
- [ ] Preserve task metadata
- [ ] Validate autonomy limits

**Verification:** Test valid task, missing objective, missing acceptance criteria, invalid autonomy, invalid YAML.

---

### TASK-004 — Task State Machine

**Objective:** Implement persistent task lifecycle management.

**Dependencies:** TASK-003.

**Required States:** RECEIVED, ANALYZING, PLANNED, IMPLEMENTING, VERIFYING, EVALUATING, REVIEWING, ITERATING, COMPLETED, FAILED, BLOCKED, NEEDS_HUMAN, CANCELLED.

**Requirements:**
- Implement `StateManager.transition()`, `StateManager.current()`, `StateManager.history()`, `StateManager.persist()`, `StateManager.restore()`
- State machine MUST reject invalid transitions

**Acceptance Criteria:**
- [ ] Valid transition (RECEIVED → ANALYZING) must succeed
- [ ] Invalid transition (COMPLETED → IMPLEMENTING) must fail
- [ ] State MUST survive process restart

---

### TASK-005 — Agent Definition Loader

**Objective:** Implement loading and validation of SOUL and PROFILE definitions.

**Dependencies:** TASK-002.

**Requirements:**
- Support `agents/<role>/SOUL.md` and `agents/<role>/PROFILE.yaml`
- PROFILE MUST support: `name`, `capabilities`, `skills`, `tools`, `autonomy`
- Loader MUST combine SOUL + PROFILE into an internal Agent Definition

**Acceptance Criteria:**
- [ ] Discover agents
- [ ] Load profiles
- [ ] Load SOUL files
- [ ] Validate capabilities
- [ ] Validate autonomy
- [ ] Reject malformed profiles
- [ ] Agent definitions MUST NOT require a hard-coded model

---

### TASK-006 — Model Catalog and Model Router

**Objective:** Implement dynamic model selection according to capabilities and the free-model policy.

**Dependencies:** TASK-002, TASK-005.

**Requirements:**
- Implement ModelCatalog, ModelCandidate, ModelRouter
- Router MUST evaluate: provider, cost, capabilities, availability, role, task requirements
- Router MUST enforce: `provider = Nous`, `cost = 0`

**Selection Flow:** `Task → Required Capabilities → Agent Profile → Model Policy → Free Model Candidates → Ranking → Selected Model`

**Acceptance Criteria:**
- [ ] Reject paid models
- [ ] Reject unavailable models
- [ ] Rank candidates
- [ ] Provide fallback
- [ ] Explain why a model was selected

---

### TASK-007 — Context Engineering Layer

**Objective:** Implement controlled context generation.

**Dependencies:** TASK-003, TASK-005.

**Requirements:**
- Create ContextManager, ContextPack
- Context MUST support: task context, project context, agent context, team context, relevant files, architectural decisions, memory
- Context Manager MUST avoid automatically injecting the entire repository

**Acceptance Criteria:**
- [ ] Given a task and coder agent, produce a deterministic Context Pack containing only relevant information

---

### TASK-008 — Agent Execution Engine

**Objective:** Implement controlled execution of an agent using SOUL + PROFILE + TEAM + TASK + CONTEXT + MODEL + TOOLS.

**Dependencies:** TASK-004, TASK-005, TASK-006, TASK-007.

**Requirements:**
- Implement `AgentExecutor.execute()`
- Execution MUST produce structured results with status, changes, artifacts, claims, requested_verification
- Execution engine MUST enforce autonomy policy

**Acceptance Criteria:**
- [ ] Agent receives complete execution context
- [ ] Cannot use unauthorized tools
- [ ] Cannot exceed autonomy level
- [ ] Produces structured execution results

---

### TASK-009 — Workspace and Scope Enforcement

**Objective:** Prevent agents from modifying files outside the Task Contract.

**Dependencies:** TASK-003, TASK-008.

**Requirements:**
- Implement `allowed_changes` and `forbidden_changes` validation
- Before execution: validate intended scope. After execution: inspect actual changes.

**Acceptance Criteria:**
- [ ] With `allowed_changes: [src/auth/**]`, agent MUST NOT modify `infrastructure/**`
- [ ] Violation MUST produce `SCOPE_VIOLATION` and prevent task completion

---

### TASK-010 — Verification Engine

**Objective:** Implement objective verification of the generated implementation.

**Dependencies:** TASK-008, TASK-009.

**Requirements:**
- Create pluggable verification interfaces: Verifier, VerificationEngine
- Initial verifiers: UnitTestVerifier, IntegrationTestVerifier, BuildVerifier, LintVerifier, AcceptanceVerifier

**Acceptance Criteria:**
- [ ] Execute configured checks
- [ ] Capture exit codes and output
- [ ] Determine pass/fail
- [ ] Produce structured results

---

### TASK-011 — Evidence Collector

**Objective:** Implement evidence-backed execution reporting.

**Dependencies:** TASK-008, TASK-010.

**Requirements:**
- Collect: changed files, commands executed, exit codes, test results, verification results, execution metadata
- Create `artifacts/evidence/<TASK-ID>/`

**Acceptance Criteria:**
- [ ] Every successful verification MUST have corresponding evidence
- [ ] System MUST distinguish agent claim from verified evidence

---

### TASK-012 — Evaluation and Acceptance Engine

**Objective:** Implement evaluation of task acceptance criteria and overall quality.

**Dependencies:** TASK-010, TASK-011.

**Requirements:**
- Implement AcceptanceEvaluator, EvaluationEngine
- Evaluate: correctness, architecture, security, maintainability, requirement coverage, efficiency

**Acceptance Criteria:**
- [ ] Evaluate mandatory acceptance criteria
- [ ] Calculate a score
- [ ] Produce PASS/FAIL
- [ ] Explain failures

---

### TASK-013 — Feedback and Iteration Engine

**Objective:** Implement controlled correction loops.

**Dependencies:** TASK-004, TASK-010, TASK-012.

**Requirements:**
- Support failure classes: MODEL_FAILURE, AGENT_FAILURE, TOOL_FAILURE, BUILD_FAILURE, TEST_FAILURE, SECURITY_FAILURE, SCOPE_VIOLATION, ARCHITECTURE_FAILURE, REQUIREMENT_FAILURE, HARNESS_FAILURE
- Maximum: 3 iterations

**Acceptance Criteria:**
- [ ] Retry recoverable failures
- [ ] Preserve previous evidence
- [ ] Provide feedback to next execution
- [ ] Stop after maximum iterations
- [ ] Escalate unrecoverable failures

---

### TASK-014 — Orchestrator and End-to-End Harness

**Objective:** Integrate all V2 components into the complete Harness.

**Dependencies:** TASK-001 through TASK-013.

**Requirements:**
- Implement `Orchestrator.run(task)`
- Complete workflow: RECEIVED → ANALYZING → PLANNED → IMPLEMENTING → VERIFYING → EVALUATING → REVIEWING → COMPLETED
- Failure path: VERIFYING → FAILED → FEEDBACK → ITERATING → IMPLEMENTING

**Acceptance Criteria:**
- [ ] A complete synthetic task MUST execute successfully from Task Contract to COMPLETED
- [ ] Must include: selected free model, selected agent, Context Pack, implementation, verification, evidence, evaluation, final report

---

### TASK-015 — Autonomous Bootstrap, Regression and Documentation

**Objective:** Make the V2 Harness usable by Hermes Agent as a self-directed implementation environment.

**Dependencies:** TASK-014.

**Requirements:**
- Create: `docs/architecture/`, `docs/specifications/`, `artifacts/reports/`
- Update README.md with V2 architecture, installation, configuration, task creation, execution, verification, model selection, troubleshooting
- Create `tests/e2e/` with a complete end-to-end scenario

**End-to-End Scenario:** User submits "Implement a small software feature" → Harness creates task, selects agents, selects free model, builds context, executes, verifies, collects evidence, evaluates, completes.

**Acceptance Criteria:**
- [ ] All automated tests pass
- [ ] README contains enough information for a new user to execute the Harness
- [ ] Repository has no broken imports
- [ ] V2 workflow is reproducible from clean environment

---

## 6. Global Acceptance Test

After TASK-015, execute the following scenario:

**Task:** Implement a simple authentication endpoint.

Expected Harness behavior:
1. Parse Task Contract
2. Validate constraints
3. Create task state
4. Select Manager
5. Select Architect
6. Build architecture plan
7. Select Coder
8. Select free Nous model
9. Build Context Pack
10. Implement
11. Run tests
12. Collect evidence
13. Evaluate acceptance criteria
14. Review implementation
15. Mark COMPLETED

If implementation fails: `FAIL → FEEDBACK → ITERATION → RE-EXECUTION` (max 3).

---

## 7. Global Definition of Done

V2.0 implementation is complete only when:

- [ ] SPEC-V2.md implemented
- [ ] Task Contracts operational
- [ ] State Machine operational
- [ ] Agent loader operational
- [ ] Model Router operational
- [ ] Context Manager operational
- [ ] Execution Engine operational
- [ ] Workspace enforcement operational
- [ ] Verification operational
- [ ] Evidence operational
- [ ] Evaluation operational
- [ ] Iteration operational
- [ ] Orchestrator operational
- [ ] E2E tests passing
- [ ] V1 behavior preserved

---

## 8. Implementation Priority

| Priority | Tasks |
|----------|-------|
| P0 — Mandatory | TASK-001, TASK-002, TASK-003, TASK-004, TASK-006, TASK-007, TASK-008, TASK-009, TASK-010, TASK-011, TASK-013, TASK-014 |
| P1 — Important | TASK-005, TASK-012, TASK-015 |

---

## 9. Autonomous Agent Directive

> You are implementing HADevTeamForFreeModels V2.0.
>
> Read SPEC-V2.md and IMPLEMENTATION-PLAN-V2.md before modifying the repository.
>
> Execute implementation tasks sequentially.
>
> For each task:
> 1. Read the task definition.
> 2. Inspect all relevant existing files.
> 3. Identify dependencies.
> 4. Create an implementation plan.
> 5. Implement only the requested scope.
> 6. Run the required verification.
> 7. Fix failures when possible.
> 8. Generate the required artifacts.
> 9. Confirm every acceptance criterion.
> 10. Only then proceed to the next task.
>
> Never claim a task is complete without verification evidence.
> Never use paid models.
> Never violate allowed/forbidden file boundaries.
> Never silently modify the architecture defined by SPEC-V2.md.
>
> If a task cannot be safely completed, stop and report: current state, attempted actions, failure, evidence, required human decision.
>
> Do not continue to subsequent tasks after an unresolved blocking failure.

---

## 10. Version 2.0 Success Definition

HADevTeamForFreeModels V2.0 succeeds when the Hermes Agent can take a formally defined software engineering task and autonomously execute the complete engineering lifecycle while remaining within defined model, tool, scope, autonomy, verification, and human-approval boundaries.

The system must be able to answer not only "What did the AI generate?" but:

> **What was requested, who performed it, which model was used, what context was provided, what changed, what was verified, what evidence proves it, how was it evaluated, and why was the task considered complete?**

That is the fundamental transition from AI Agent orchestration to AI Engineering Harness.
