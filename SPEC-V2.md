# HADevTeamForFreeModels V2.0 — AI Engineering Harness

**Status:** Proposed  
**Version:** 2.0.0  
**Runtime:** Hermes Agent  
**Model Provider:** Nous Research  
**Model Cost Constraint:** Free / Cost = 0

---

## 1. Purpose

HADevTeamForFreeModels V2.0 transforms the project from a multi-agent software development team into an **AI Engineering Harness** capable of coordinating, executing, verifying, evaluating, and iterating software engineering tasks.

The Harness provides the environment and control mechanisms required for AI agents to perform software engineering work in a controlled, observable, verifiable, and repeatable way.

The system must preserve the project's original principle:

> Use free Nous Research models dynamically according to the capabilities required by each agent and task.

V2.0 extends this principle by introducing:

- Task Contracts
- State Machine
- Context Engineering
- Agent execution control
- Model routing
- Verification
- Evidence collection
- Evaluation
- Controlled iteration
- Autonomy policies
- Human-in-the-loop escalation
- Persistent engineering artifacts

---

## 2. Core Architectural Principle

The system MUST treat the following concepts as separate entities:

```
MODEL
  ↓
AGENT
  ↓
ROLE
  ↓
TASK
  ↓
HARNESS
```

- **Model** — The LLM used to perform reasoning or generation.
- **Agent** — A software-engineering actor with identity, behavior, capabilities, tools, and autonomy.
- **Role** — The responsibility performed by an agent in the development workflow.
- **Task** — A concrete unit of engineering work.
- **Harness** — The execution environment controlling the complete lifecycle of the task.

---

## 3. High-Level Architecture

```
                         USER
                           |
                           v
                    +-------------+
                    | Task Intake |
                    +------+------+
                           |
                           v
                  +------------------+
                  |  Task Contract   |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  |   Orchestrator   |
                  +--------+---------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
      +-------------+             +-------------+
      |   Context   |             |    Model    |
      |   Manager   |             |   Router    |
      +------+------+             +------+------+
             |                           |
             +-------------+-------------+
                           |
                           v
                  +------------------+
                  | Agent Execution  |
                  +--------+---------+
                           |
                           v
                     WORKSPACE
                           |
                           v
                  +------------------+
                  |   Verification   |
                  +--------+---------+
                           |
                           v
                     +----------+
                     | Evidence |
                     +----+-----+
                          |
                          v
                    +-----------+
                    | Evaluation|
                    +-----+-----+
                          |
                 +--------+--------+
                 |                 |
                 v                 v
               PASS              FAIL
                 |                 |
                 v                 v
             REVIEW            FEEDBACK
                 |                 |
                 |                 v
                 |             ITERATION
                 |                 |
                 +-----------------+
                          |
                          v
                       DELIVERY
```

---

## 4. Repository Structure

```
HADevTeamForFreeModels/
│
├── README.md
├── SPEC-V2.md              ← this document
├── IMPLEMENTATION-PLAN-V2.md ← ordered implementation tasks
├── LICENSE
├── setup.py
│
├── config/
│   ├── harness.yaml
│   ├── model_policy.yaml
│   ├── execution_policy.yaml
│   └── verification_policy.yaml
│
├── agents/
│   ├── manager/
│   │   ├── SOUL.md
│   │   └── PROFILE.yaml
│   ├── architect/
│   ├── coder/
│   ├── tester/
│   └── reviewer/
│
├── team/
│   └── TEAM.md
│
├── harness/
│   ├── orchestrator/
│   ├── context/
│   ├── routing/
│   ├── execution/
│   ├── verification/
│   ├── evaluation/
│   ├── evidence/
│   ├── memory/
│   └── state/
│
├── tasks/
│   ├── templates/
│   └── active/
│
├── artifacts/
│   ├── executions/
│   ├── evidence/
│   ├── evaluations/
│   ├── decisions/
│   └── reports/
│
├── skills/
├── tests/
└── docs/
    ├── architecture/
    ├── adr/
    └── specifications/
```

---

## 5. Configuration

### `config/harness.yaml`

```yaml
version: "2.0"

harness:
  name: HADevTeamForFreeModels
  mode: software-engineering

models:
  provider: nous
  cost:
    maximum: 0

execution:
  max_iterations: 3
  require_task_contract: true
  require_verification: true
  require_evidence: true
  require_evaluation: true

autonomy:
  default_level: 2

workspace:
  isolated: true

human_approval:
  required_for:
    - production
    - deployment
    - destructive_operations
    - security_override
```

---

## 6. Agent Architecture

Each agent MUST have:

```
agents/<role>/
├── SOUL.md       ← identity, behavior, principles, communication
└── PROFILE.yaml  ← capabilities, skills, tools, autonomy level
```

### SOUL.md — defines who the agent IS and how it thinks/acts

```markdown
# Coder Soul

## Identity
You are a software engineer responsible for implementing approved technical designs.

## Principles
- Follow the approved architecture.
- Do not silently change requirements.
- Prefer simple and maintainable solutions.
- Do not claim success without evidence.
- Do not modify files outside the allowed scope.

## Behavior
1. Understand the task.
2. Inspect relevant files.
3. Identify existing conventions.
4. Implement the smallest correct change.
5. Run verification.

## Communication
Always report:
- changes performed;
- files modified;
- tests executed;
- verification results;
- remaining problems.
```

### PROFILE.yaml — defines operational capabilities (NO hard-coded model)

```yaml
name: coder

capabilities:
  - software-development
  - debugging
  - refactoring

skills:
  - programming
  - testing
  - git

tools:
  - filesystem
  - terminal
  - git

autonomy:
  level: 3
```

---

## 7. Team Contract

`team/TEAM.md` defines collaboration rules.

```markdown
# Development Team

## Workflow
Manager → Architect → Coder → Tester → Reviewer

## Rules
1. Manager analyzes and coordinates the task.
2. Architect defines the technical approach.
3. Coder implements the approved design.
4. Tester independently verifies the implementation.
5. Reviewer evaluates the final result.
6. Verification failures trigger controlled iteration.
7. Security failures block completion.
8. Production-impacting actions require human approval.

## Communication
Agents communicate through task artifacts and explicit execution results.
```

---

## 8. Task Contract

Every engineering task MUST be represented by a **Task Contract**.

### `tasks/templates/task.yaml`

```yaml
apiVersion: harness/v1
kind: Task

metadata:
  id: TASK-0001
  title: ""

spec:
  objective: ""
  requirements: []
  constraints: []
  acceptance_criteria: []
  allowed_changes: []
  forbidden_changes: []
  verification:
    required: []
  autonomy:
    maximum: 3
```

### Example Task Contract

```yaml
apiVersion: harness/v1
kind: Task

metadata:
  id: TASK-0001
  title: Implement JWT authentication

spec:
  objective: Implement REST authentication using JWT.

  requirements:
    - email/password authentication
    - password hashing
    - JWT token generation
    - invalid credential handling

  constraints:
    - use existing architecture
    - no infrastructure changes

  acceptance_criteria:
    - valid credentials return JWT
    - invalid credentials return HTTP 401
    - passwords are never stored in plaintext
    - automated tests pass

  allowed_changes:
    - src/auth/**
    - tests/auth/**

  forbidden_changes:
    - infrastructure/**
    - deployment/**

  verification:
    required:
      - unit_tests
      - integration_tests
      - lint

  autonomy:
    maximum: 3
```

---

## 9. Task Lifecycle (State Machine)

```
RECEIVED → ANALYZING → PLANNED → IMPLEMENTING → VERIFYING → EVALUATING → REVIEWING
                                                                               |
                                                                         +-----+-----+
                                                                         |           |
                                                                       PASS        FAIL
                                                                         |           |
                                                                         v           v
                                                                     COMPLETED   FEEDBACK
                                                                                     |
                                                                                     v
                                                                                 ITERATING
                                                                                     |
                                                                                     v
                                                                                IMPLEMENTING
```

Additional states: `BLOCKED`, `FAILED`, `NEEDS_HUMAN`, `CANCELLED`

---

## 10. Core Interfaces

```python
class TaskManager:
    def create(self, definition): ...
    def load(self, task_id): ...
    def update_state(self, task_id, state): ...

class ContextManager:
    def build_context(self, task, agent): ...

class ModelRouter:
    def select_model(self, task, role, context): ...

class AgentExecutor:
    def execute(self, agent, model, task, context): ...

class VerificationEngine:
    def verify(self, task, workspace): ...

class EvaluationEngine:
    def evaluate(self, task, verification, evidence): ...

class EvidenceCollector:
    def collect(self, execution): ...

class MemoryManager:
    def store(self, data): ...
    def retrieve(self, query): ...

class Orchestrator:
    def run(self, task): ...
    def iterate(self, feedback): ...
```

---

## 11. Verification Engine

Pluggable verifiers:

```
VerificationEngine
    ├── UnitTestVerifier
    ├── IntegrationTestVerifier
    ├── BuildVerifier
    ├── LintVerifier
    ├── SecurityVerifier
    ├── AcceptanceVerifier
    └── CustomVerifier
```

### Verification Result

```yaml
verification:
  status: passed
  checks:
    unit_tests:
      status: passed
      passed: 42
      failed: 0
    integration_tests:
      status: passed
    lint:
      status: passed
    security:
      status: passed
```

---

## 12. Evidence System

Every important completion claim MUST be backed by evidence.

```yaml
evidence:
  changed_files:
    - src/auth/LoginService.php
    - tests/auth/LoginTest.php

  tests:
    command: "./vendor/bin/phpunit"
    exit_code: 0
    passed: 42
    failed: 0

  lint:
    command: "phpstan analyse"
    exit_code: 0

  git_diff:
    available: true
```

---

## 13. Evaluation Engine

```yaml
evaluation:
  score: 0.91
  dimensions:
    correctness: 0.95
    architecture: 0.90
    security: 0.94
    maintainability: 0.88
    efficiency: 0.87
  decision: PASS
```

### Evaluation Policy

```yaml
evaluation:
  weights:
    correctness: 0.40
    architecture: 0.20
    security: 0.20
    maintainability: 0.10
    efficiency: 0.10
  minimum_score: 0.85
```

---

## 14. Feedback & Iteration

```yaml
feedback:
  task: TASK-0001
  failures:
    - check: authentication_invalid_password
      reason: expected HTTP 401, received HTTP 500
  recommendations:
    - inspect exception handling
    - add negative-path test
  target_agent: coder
```

### Iteration Policy

```yaml
iteration:
  maximum: 3
  rules:
    verification_failure:
      action: retry
    repeated_failure:
      action: escalate
    security_failure:
      action: block
    scope_violation:
      action: block
    architecture_conflict:
      action: architect_review
```

---

## 15. Autonomy Levels

```yaml
levels:
  0: advisory
  1: read_only
  2: controlled_edit
  3: execute_tests
  4: autonomous_development
  5: production
```

### Tool Policies

```yaml
tools:
  filesystem_read:
    autonomy: 1
  filesystem_write:
    autonomy: 2
  terminal:
    autonomy: 3
  git_commit:
    autonomy: 3
  deploy:
    autonomy: 5
```

---

## 16. Execution Artifacts

Each task MUST produce an execution record:

```
artifacts/executions/TASK-0001/
├── task.yaml
├── plan.yaml
├── context.yaml
├── execution.yaml
├── verification.yaml
├── evaluation.yaml
├── feedback.yaml
└── final_report.md
```

---

## 17. Model Performance Registry (future)

```yaml
model: example-model
tasks:
  total: 37
  successful: 32
metrics:
  success_rate: 0.865
  avg_iterations: 1.4
  avg_score: 0.89
  avg_latency: 12.4
capabilities:
  coding: 0.91
  reasoning: 0.86
  testing: 0.79
```

---

## 18. Architectural Invariants

- **INV-001** — The Model Router MUST NOT bypass the configured cost policy.
- **INV-002** — Agents MUST NOT define their own unrestricted model selection.
- **INV-003** — Agents MUST NOT override Task Contract constraints.
- **INV-004** — Agent claims MUST NOT replace automated verification.
- **INV-005** — Verification failures MUST prevent successful completion.
- **INV-006** — The Harness MUST prevent uncontrolled iteration.
- **INV-007** — Production-impacting operations MUST require explicit authorization.
- **INV-008** — The implementation agent SHOULD NOT be the sole reviewer of its own work.
- **INV-009** — Task state MUST be recoverable.
- **INV-010** — Important execution decisions MUST produce persistent artifacts.

---

## 19. Definition of Done

A task is **COMPLETED** only when:

```
CONTRACT_VALID
    ∧ IMPLEMENTED
    ∧ VERIFIED
    ∧ ACCEPTED
    ∧ SECURE
    ∧ IN_SCOPE
    ∧ EVIDENCE_AVAILABLE
    ∧ EVALUATED
    ∧ REVIEWED
```

---

## 20. Final Architectural Statement

> The LLM is not the engineering system. The Harness is the engineering system; the LLM is one of its interchangeable execution components.

This architecture allows HADevTeamForFreeModels to evolve from a free-model multi-agent experiment into a reproducible, verifiable, adaptive AI software engineering platform.
