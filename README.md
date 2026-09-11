# HADevTeamForFreeModels

> An AI Engineering Harness for Hermes Agent — free models, verifiable execution, autonomous software engineering.

## What is this?

A **capability-driven AI Engineering Harness** that receives software engineering demands, analyzes what capabilities they require, selects from registered agent profiles by capability match, and executes a dynamic, verified workflow — using **only free models from the Nous Portal**.

### V2.2 Capability-Driven Architecture

Instead of asking **"What is the next step in a fixed workflow?"** , the Harness asks:

**"What needs to be done next, and what capabilities are required?"**

```
CAPABILITY → AGENT → MODEL
```

Not:

```
FIXED ROLE → FIXED AGENT → MODEL
```

Capabilities drive what must be done. Agents are selected by their declared capabilities (via `CapabilityRegistry`). Models are assigned by policy. **Roles are metadata, not workflow primitives.**

### How it works

1. **Task intake** → Formal `TaskContract` with objective, acceptance criteria, allowed changes
2. **Task analysis** (`TaskAnalyzer`) → What capabilities are needed? (e.g., `api_design`, `backend_development`, `testing`, `review`)
3. **Capability matching** (`CapabilityMatcher` + `CapabilityRegistry`) → Find agents that declare each required capability
4. **Dependency resolution** (`CapabilityGraph` + declarative `capability_dependencies` config) → Build a DAG of capability dependencies
5. **Workflow planning** (`WorkflowPlanner`) → Assign agents to capabilities, build an `ExecutionWorkflow`
6. **Workflow validation** (`WorkflowValidator`) → 10 checks: DAG integrity, mandatory gates, free models, reviewer independence, workspace conflicts
7. **Dynamic execution** (`ExecutionGraph`) → Execute the DAG in dependency order, independent nodes in parallel
8. **Verification** → Unit tests, lint, build checks
9. **Evaluation** → Acceptance criteria scoring
10. **Review** → Independent reviewer gate
11. **Iteration** → On failure: `FailureAnalyzer` → `Replanner` → re-execute (max 3 attempts)

### Mandatory gates (never bypassed)

| Gate | Enforced by |
|------|-------------|
| Verification | `WorkflowValidator`, `VerificationEngine` |
| Evaluation | `WorkflowValidator`, `AdvancedEvaluationEngine` |
| Independent review | `WorkflowValidator` (check #10) |
| Free models only | `WorkflowValidator` (cost = 0) |
| DAG integrity | `CapabilityGraph.has_cycle()`, `WorkflowValidator` |

### What's NOT in V2.2

- No hard-coded role sequences (manager → architect → coder → tester → reviewer)
- No `DEFAULT_AGENT_FOR_CAPABILITY` fallback — the `CapabilityRegistry` is **authoritative**
- No hard-coded capability dependency rules — all dependencies are **declarative** in `config/capability_taxonomy.yaml`
- No role-based workflow logic — roles are **metadata** (used for display, not for routing)

## Quick Start

```bash
git clone https://github.com/hectordufau/HADevTeamForFreeModels.git
cd HADevTeamForFreeModels
python3 setup.py
```

Then in Hermes chat:

```
/devteamfree <your demand>
```

## Documentation

| Document | Purpose |
|---|---|
| [`SPEC-V2.md`](./SPEC-V2.md) | V2.0 architectural specification |
| [`SPEC-V2.2.md`](./SPEC-V2.2.md) | V2.2 capability-driven architecture specification |
| [`IMPLEMENTATION-PLAN-V2.md`](./IMPLEMENTATION-PLAN-V2.md) | Ordered implementation tasks |
| [`setup.py`](./setup.py) | Generates profiles from Nous Portal FREE models |

## Agent Profiles (V2.1 roles — used as metadata, not workflow primitives)

| Profile | Role | Capabilities |
|---|---|---|
| `devfree-manager` | Triage/orchestration | `orchestration`, `task_classification`, `complexity_assessment` |
| `devfree-architect` | Architecture/security | `architecture`, `system_design`, `security` |
| `devfree-coder` | Implementation | `coding`, `refactoring`, `testing` |
| `devfree-tester` | Tests/debug | `testing`, `debugging`, `verification` |
| `devfree-reviewer` | Code review | `code_review`, `reasoning`, `security` |
| `devfree-worker` | Simple tasks | `coding`, `debugging` |

New specialist agents can be added by creating a `PROFILE.yaml` with their capabilities — the Harness discovers them automatically.

## Requirements

- Hermes Agent
- Nous Portal account (free)
- `hermes setup --portal` configured

## V3.1 Learning Optimization & Decision Intelligence

V3.1 improves how the harness converts accumulated experience into better engineering decisions. Built on V3.0's foundational learning modules, it adds:

- **Experience Extraction 2.0** — Structured experience representation with task context, strategy, outcome, failures, evidence links, and quality scoring
- **Memory Retrieval 2.0** — Multi-factor retrieval (similarity, usefulness, applicability, confidence, evidence quality, recency) with historical usefulness tracking and confidence calibration
- **Learned Strategy Layer** — Strategy generation from accumulated experiences, validation against policy/capabilities/evidence
- **Learning→Decision Coupling** — Standardized decision context, strategy-influenced model/agent/workflow selection with impact tracking
- **Adaptive Exploration** — Exploration rate adapts to confidence (2-20%), purposeful candidate selection maximizing information gain
- **Learning Evaluation** — Learning Gain, Generalization Gain, Failure Avoidance, Decision Influence, and Learning Efficiency metrics

### Architecture (V3.1 Additions)

```
Experience Extraction 2.0 → Memory Retrieval 2.0 → Learned Strategy → Decision Coupling
         ↓                         ↓                        ↓                  ↓
    Quality Scoring         Confidence Calibration     Validation        Impact Tracking
```

### Documentation

| Document | Purpose |
|---|---|
| [`docs/V3.1/PRE_IMPLEMENTATION_AUDIT.md`](./docs/V3.1/PRE_IMPLEMENTATION_AUDIT.md) | Pre-implementation audit of V3.0 codebase |
| [`docs/V3.1/VALIDATION_REPORT.md`](./docs/V3.1/VALIDATION_REPORT.md) | V3.1 validation results and release gates |

## License

MIT
