# Hermes Agent DevTeam For Free Models

> An AI Engineering Harness for Hermes Agent — free models, verifiable execution, autonomous software engineering, learning and optimization.

Transform your Hermes Agent into a complete, self-improving software engineering team using **only free models from the Nous Portal**.

---

## What is this?

A **capability-driven, learning-enabled AI Engineering Harness** that receives software engineering demands, analyzes what capabilities they require, selects agents by capability match, executes dynamic verified workflows, **learns from every execution**, and **optimizes future decisions** — all at zero model cost.

### Core Principle

```
           CAPABILITY → AGENT → MODEL
```

Not:

```
           FIXED ROLE → FIXED AGENT → MODEL
```

Capabilities drive what must be done. Agents are selected by their declared capabilities via `CapabilityRegistry`. Models are assigned by policy and historical performance. **Roles are metadata, not workflow primitives.**

---

## How it works

```
              ┌─────────────────────┐
              │        TASK         │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │    TASK ANALYZER    │
              └──────────┬──────────┘
                         ↓
              REQUIRED CAPABILITIES
                         ↓
              ┌─────────────────────┐
              │   EXPERIENCE +      │
              │  STRATEGY RETRIEVAL │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │ WORKFLOW PLANNER    │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │ WORKFLOW VALIDATOR  │
              └──────────┬──────────┘
                         ↓
              ┌─────────────────────┐
              │ EXECUTION GRAPH     │
              └──────────┬──────────┘
                         ↓
       ┌──────────────────┼──────────────────┐
       ↓                  ↓                  ↓
    AGENT               AGENT              AGENT
    MODEL               MODEL              MODEL
       └──────────────────┼──────────────────┘
                          ↓
                    VERIFICATION
                          ↓
                     EVALUATION
                          ↓
                      REVIEWER
                          ↓
                    EVIDENCE STORE
                          ↓
              ┌───────────┴───────────┐
              ↓                       ↓
         PERFORMANCE             ENGINEERING
         REGISTRY V3               MEMORY
              ↓                       ↓
              └───────────┬───────────┘
                          ↓
                   LEARNING ENGINE
                          ↓
                  OPTIMIZATION ENGINE
                          ↓
                    FUTURE DECISIONS
```

---

## Version History

| Version | Tag | What it delivers |
|---------|-----|-----------------|
| **V2.1.0** | `v2.1.0` | Adaptive Harness — policy enforcement, adaptive model routing, evidence, 10 release gates |
| **V2.2.0** | `v2.2.0` | Dynamic Capability-Driven Workflow — capability registry, DAG planner, parallel execution |
| **V2.2.1** | `v2.2.1` | Capability Integrity — removed role fallback, declarative dependencies, specialist agent E2E |
| **V3.0.0** | `v3.0.0` | Learning + Optimization — benchmark framework, evidence hash chain, memory confidence, model/capability/plan intelligence, optimization engine, 25 release gates |
| **V3.1.0** | `v3.1.0` | Learning Optimization & Decision Intelligence — experience extraction 2.0, multi-factor retrieval, learned strategy layer, learning→decision coupling, adaptive exploration |

---

## Validation Results

### V3.0 (100 benchmark executions)

| Metric | Baseline | After Learning | Improvement |
|--------|----------|---------------|-------------|
| Success Rate | 74.0% | 76.0% | **+2.0%** |
| Mean Score | 0.7184 | 0.7229 | **+0.0045** |
| Mean Latency | 4975ms | 4532ms | **-443ms** |
| Generalization Gain | — | +0.1787 | **PASS** |
| Failure → Learning → Recovery | — | All checks | **PASS** |
| Security Violations | 0 | 0 | **PASS** |
| Governance Violations | 0 | 0 | **PASS** |

### V3.1 (407 tests, 15 release gates)

| Gate | Description | Status |
|------|-------------|--------|
| L1–L15 | All learning optimization gates | **ALL PASS** |
| Security boundary | Learning cannot disable governance | **PASS** |
| Prompt injection | Memories treated as untrusted data | **PASS** |
| Regression | V2.1+V2.2+V2.2.1+V3.0+V3.1 | **ALL PASS** |

---

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

---

## Documentation

| Document | Purpose |
|---|---|
| [`SPEC-V2.md`](./SPEC-V2.md) | V2.0 architectural specification |
| [`SPEC-V2.2.md`](./SPEC-V2.2.md) | V2.2 capability-driven architecture specification |
| [`V3.0-VALIDATION-PLAN.md`](./V3.0-VALIDATION-PLAN.md) | V3.0 validation protocol and decision gate |
| [`V3.0-VALIDATION-REPORT.md`](./V3.0-VALIDATION-REPORT.md) | V3.0 validation results (100 executions) |
| [`V3-IMPLEMENTATION-REPORT.md`](./V3-IMPLEMENTATION-REPORT.md) | V3 implementation report |
| [`V3-ARCHITECTURE.md`](./V3-ARCHITECTURE.md) | V3 architecture documentation |
| [`V3-BENCHMARK-REPORT.md`](./V3-BENCHMARK-REPORT.md) | V3 benchmark results |
| [`V3-RELEASE-GATE.md`](./V3-RELEASE-GATE.md) | V3 release gate verification |
| [`docs/V3.1/PRE_IMPLEMENTATION_AUDIT.md`](./docs/V3.1/PRE_IMPLEMENTATION_AUDIT.md) | V3.1 pre-implementation audit |
| [`docs/V3.1/VALIDATION_REPORT.md`](./docs/V3.1/VALIDATION_REPORT.md) | V3.1 validation results |
| [`setup.py`](./setup.py) | Generates profiles from Nous Portal FREE models |

---

## Architecture Overview

### Core Modules (V2.1–V2.2)

| Module | Location |
|--------|----------|
| Capability Registry | `harness/capabilities/registry.py` |
| Capability Taxonomy | `harness/capabilities/taxonomy.py` |
| Capability Graph (DAG) | `harness/capabilities/graph.py` |
| Capability Matching | `harness/capabilities/matching.py` |
| Task Analyzer | `harness/planner/task_analyzer.py` |
| Workflow Planner | `harness/planner/workflow_planner.py` |
| Workflow Validator (10 checks) | `harness/planner/workflow_validator.py` |
| Execution Graph (parallel DAG) | `harness/planner/exec_graph.py` |
| Workspace Manager | `harness/planner/workspace.py` |
| Branching | `harness/planner/branching.py` |
| Failure Analyzer | `harness/planner/failure_analyzer.py` |
| Replanner | `harness/planner/replanner.py` |
| Policy Engine | `harness/policy/__init__.py` |
| Evidence Store | `harness/evidence/__init__.py` |
| Model Router + Catalog | `harness/routing/__init__.py` |
| State Machine | `harness/state/__init__.py` |
| Task Contract Engine | `harness/task/__init__.py` |
| Agent Loader | `harness/agents/__init__.py` |
| Agent Executor | `harness/execution/__init__.py` |
| Verification Engine | `harness/verification/__init__.py` |
| Evaluation Engine | `harness/evaluation/__init__.py` |
| Iteration Engine | `harness/iteration/__init__.py` |
| Memory Manager | `harness/memory/__init__.py` |
| ADR Automation | `harness/memory/adr.py` |
| Context Manager | `harness/context/__init__.py` |
| Orchestrator | `harness/orchestrator/__init__.py` |
| Workflow Engine | `harness/workflow/__init__.py` |

### V3.0 Intelligence Modules

| Module | Location |
|--------|----------|
| Capability Intelligence | `harness/capabilities/intelligence.py` |
| Task Intelligence | `harness/planner/task_intelligence.py` |
| Model Intelligence | `harness/planner/model_intelligence.py` |
| Plan Intelligence | `harness/planner/plan_intelligence.py` |
| Failure Intelligence | `harness/planner/failure_intelligence.py` |
| Context V3 | `harness/context/v3.py` |
| Observability | `harness/planner/observability.py` |
| Optimization Engine | `harness/planner/optimization.py` |
| Efficiency | `harness/planner/efficiency.py` |
| V3 Integration | `harness/planner/v3_integration.py` |
| Security/Governance V3 | `harness/policy/v3.py` |
| Evidence Unified (hash chain) | `harness/evidence/unified.py` |
| Performance Registry V3 | `harness/routing/performance_v3.py` |
| Memory V3 (confidence) | `harness/memory/v3.py` |
| Benchmark Engine | `harness/benchmark/engine.py` |
| Benchmark Metrics | `harness/benchmark/metrics.py` |
| Benchmark Report | `harness/benchmark/report.py` |
| Advanced Evaluation | `harness/evaluation/advanced.py` |

### V3.1 Learning Optimization Modules

| Module | Location |
|--------|----------|
| Experience Engine | `harness/learning/experience.py` |
| Retrieval Engine | `harness/learning/retrieval.py` |
| Strategy Engine | `harness/learning/strategy.py` |
| Learning Evaluation | `harness/learning/evaluation.py` |

---

## Agent Profiles

| Profile | Role | Capabilities |
|---------|------|-------------|
| `devfree-manager` | Triage/orchestration | `orchestration`, `task_classification`, `complexity_assessment` |
| `devfree-architect` | Architecture/security | `architecture`, `system_design`, `security` |
| `devfree-coder` | Implementation | `coding`, `refactoring`, `testing` |
| `devfree-tester` | Tests/debug | `testing`, `debugging`, `verification` |
| `devfree-reviewer` | Code review | `code_review`, `reasoning`, `security` |
| `devfree-worker` | Simple tasks | `coding`, `debugging` |

New specialist agents can be added by creating a `PROFILE.yaml` with their capabilities — the Harness discovers them automatically via `CapabilityRegistry`.

---

## Requirements

- Hermes Agent
- Nous Portal account (free)
- `hermes setup --portal` configured

---

## License

MIT
