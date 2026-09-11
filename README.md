# Hermes Agent DevTeam For Free Models

> An AI Engineering Harness for Hermes Agent — free models, verifiable execution, autonomous software engineering, learning and optimization.

Transform your Hermes Agent into a complete, self-improving software engineering team using **only free models from the Nous Portal**.

---

## What is this?

A **capability-driven, learning-enabled AI Engineering Harness** that receives software engineering demands, analyzes what capabilities they require, selects agents by capability match, executes dynamic verified workflows, **learns from every execution**, and **optimizes future decisions** — all at zero model cost.

### Capability-Driven Architecture

Instead of asking **"What is the next step in a fixed workflow?"**, the Harness asks: **"What needs to be done next, and what capabilities are required?"**

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
              │ CAPABILITY GRAPH    │
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
| **V3.1** | *recommended* | Improve Learning Engine — experience extraction, confidence scoring, learning-to-execution coupling |

### Current Release: V3.0.0

**312 automated tests** · **25 V3 release gates** · **10 V2.1/V2.2/V2.2.1 gates** · **100% passing**

---

## Validation Results (100 benchmark executions)

| Metric | Baseline | After Learning | Improvement |
|--------|----------|---------------|-------------|
| Success Rate | 74.0% | 76.0% | **+2.0%** |
| Mean Score | 0.7184 | 0.7229 | **+0.0045** |
| Mean Latency | 4975ms | 4532ms | **-443ms** |
| Generalization Gain | — | +0.1787 | **PASS** |
| Failure → Learning → Recovery | — | All checks | **PASS** |
| Security Violations | 0 | 0 | **PASS** |
| Governance Violations | 0 | 0 | **PASS** |

---

## Mandatory Gates (never bypassed)

| Gate | Enforced by |
|------|-------------|
| Verification | `WorkflowValidator`, `VerificationEngine` |
| Evaluation | `WorkflowValidator`, `AdvancedEvaluationEngine` |
| Independent review | `WorkflowValidator` |
| Free models only (cost = 0) | `WorkflowValidator` |
| DAG integrity | `CapabilityGraph.has_cycle()`, `WorkflowValidator` |
| Evidence integrity | `EvidenceStore` hash chain |
| Security hard minimum | `AdvancedEvaluationEngine` (≥ 0.90) |
| Policy enforcement | `PolicyEngine` (autonomy, tool, execution) |

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
| [`IMPLEMENTATION-PLAN-V2.md`](./IMPLEMENTATION-PLAN-V2.md) | V2.0 ordered implementation tasks |
| [`IMPLEMENTATION-PLAN-V2.2.md`](./IMPLEMENTATION-PLAN-V2.2.md) | V2.2 implementation plan |
| [`V3.0-VALIDATION-PLAN.md`](./V3.0-VALIDATION-PLAN.md) | Validation protocol and decision gate |
| [`V3.0-VALIDATION-REPORT.md`](./V3.0-VALIDATION-REPORT.md) | 100-execution validation results |
| [`V3-BENCHMARK-REPORT.md`](./V3-BENCHMARK-REPORT.md) | Benchmark suite results |
| [`V3-RELEASE-GATE.md`](./V3-RELEASE-GATE.md) | V3 release gate results |
| [`V3-ARCHITECTURE.md`](./V3-ARCHITECTURE.md) | V3 architecture documentation |
| [`setup.py`](./setup.py) | Generates profiles from Nous Portal FREE models |

---

## Architecture (V3.0 modules)

| Module | Location | Phase |
|--------|----------|-------|
| Capability Registry | `harness/capabilities/registry.py` | Core |
| Capability Taxonomy | `harness/capabilities/taxonomy.py` | Core |
| Capability Graph (DAG) | `harness/capabilities/graph.py` | Core |
| Capability Matching | `harness/capabilities/matching.py` | Core |
| Capability Intelligence | `harness/capabilities/intelligence.py` | V3 |
| Task Analyzer | `harness/planner/task_analyzer.py` | Core |
| Task Intelligence | `harness/planner/task_intelligence.py` | V3 |
| Workflow Planner | `harness/planner/workflow_planner.py` | Core |
| Workflow Validator (10 checks) | `harness/planner/workflow_validator.py` | Core |
| Execution Graph (parallel DAG) | `harness/planner/exec_graph.py` | Core |
| Workspace Manager | `harness/planner/workspace.py` | Core |
| Branching | `harness/planner/branching.py` | Core |
| Failure Analyzer | `harness/planner/failure_analyzer.py` | Core |
| Replanner | `harness/planner/replanner.py` | Core |
| Failure Intelligence | `harness/planner/failure_intelligence.py` | V3 |
| Plan Intelligence | `harness/planner/plan_intelligence.py` | V3 |
| Model Intelligence | `harness/planner/model_intelligence.py` | V3 |
| Context V3 | `harness/context/v3.py` | V3 |
| Observability | `harness/planner/observability.py` | V3 |
| Optimization Engine | `harness/planner/optimization.py` | V3 |
| Efficiency | `harness/planner/efficiency.py` | V3 |
| V3 Integration | `harness/planner/v3_integration.py` | V3 |
| Policy Engine | `harness/policy/__init__.py` | Core |
| Security/Governance V3 | `harness/policy/v3.py` | V3 |
| Evidence Store | `harness/evidence/__init__.py` | Core |
| Evidence Unified (hash chain) | `harness/evidence/unified.py` | V3 |
| Model Router + Catalog | `harness/routing/__init__.py` | Core |
| Performance Registry V3 | `harness/routing/performance_v3.py` | V3 |
| State Machine | `harness/state/__init__.py` | Core |
| Task Contract Engine | `harness/task/__init__.py` | Core |
| Agent Loader | `harness/agents/__init__.py` | Core |
| Context Manager | `harness/context/__init__.py` | Core |
| Agent Executor | `harness/execution/__init__.py` | Core |
| Verification Engine | `harness/verification/__init__.py` | Core |
| Evaluation Engine | `harness/evaluation/__init__.py` | Core |
| Advanced Evaluation | `harness/evaluation/advanced.py` | V2.1 |
| Iteration Engine | `harness/iteration/__init__.py` | Core |
| Memory Manager | `harness/memory/__init__.py` | Core |
| Memory V3 (confidence) | `harness/memory/v3.py` | V3 |
| ADR Automation | `harness/memory/adr.py` | V2.1 |
| Orchestrator | `harness/orchestrator/__init__.py` | Core |
| Benchmark Engine | `harness/benchmark/engine.py` | V3 |
| Benchmark Metrics | `harness/benchmark/metrics.py` | V3 |
| Benchmark Report | `harness/benchmark/report.py` | V3 |
| Workflow Engine | `harness/workflow/__init__.py` | V2.1 |

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
