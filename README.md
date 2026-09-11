# HADevTeamForFreeModels

> An AI Engineering Harness for Hermes Agent — free models, verifiable execution, autonomous software engineering.

## What is this?

A **capability-driven AI Engineering Harness** that receives software engineering demands, analyzes what capabilities they require, selects from registered agent profiles by capability match, and executes a dynamic, verified workflow — using **only free models from the Nous Portal**.

### V2.2 Capability-Driven Architecture

Instead of asking **"What is the next step in a fixed workflow?"**, the Harness asks:

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
|---------|------|-------------|
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
- **Adaptive Exploration** — Exploration rate adapts to confidence (2-50%), purposeful candidate selection maximizing information gain
- **Learning Evaluation** — Learning Gain, Generalization Gain, Failure Avoidance, Decision Influence, and Learning Efficiency metrics

### Architecture (V3.1 Additions)

```
Experience Extraction 2.0 → Memory Retrieval 2.0 → Learned Strategy → Decision Coupling
         ↓                         ↓                        ↓                  ↓
    Quality Scoring         Confidence Calibration     Validation        Impact Tracking
```

### New Files (V3.1)

| File | Purpose |
|------|---------|
| `harness/learning/__init__.py` | Module package and re-exports |
| `harness/learning/experience.py` | StructuredExperience, ExperienceExtractorV31, ExperienceQualityScorer |
| `harness/learning/retrieval.py` | MultiFactorRetriever, HistoricalUsefulnessTracker, ConfidenceCalibrator, RetrievalExplainer |
| `harness/learning/strategy.py` | Strategy, StrategyGenerator, StrategyValidator, StrategyStore |
| `harness/learning/evaluation.py` | LearningGainCalculator, GeneralizationGain, FailureAvoidanceMetric, DecisionInfluenceMetric, LearningEfficiency |
| `harness/learning/exploration.py` | AdaptiveExplorationPolicy, PurposefulCandidateSelector, ExplorationResultLearner |

### Evolved Modules (V3.1)

| Module | Changes |
|--------|---------|
| `harness/memory/v3.py` | Backward-compatible V3.1 extensions |
| `harness/planner/model_intelligence.py` | Adaptive exploration rate, exploration result learning |
| `harness/planner/v3_integration.py` | DecisionContext, DecisionImpact, DecisionImpactTracker, strategy-influenced decisions |
| `harness/planner/plan_intelligence.py` | Strategy integration |
| `harness/planner/failure_intelligence.py` | Richer failure→experience pipeline |

### Test Coverage (V3.1)

| File | Type |
|------|------|
| `tests/unit/test_learning_experience.py` | Unit tests for experience extraction |
| `tests/unit/test_learning_retrieval.py` | Unit tests for multi-factor retrieval |
| `tests/unit/test_learning_strategy.py` | Unit tests for strategy layer |
| `tests/unit/test_learning_evaluation.py` | Unit tests for learning metrics |
| `tests/unit/test_learning_exploration.py` | Unit tests for adaptive exploration |
| `tests/unit/test_model_intelligence_v31.py` | Unit tests for V3.1 model intelligence changes |
| `tests/integration/test_v31_integration.py` | Integration tests for learning→decision coupling |
| `tests/integration/test_v31_release_gate.py` | 15 release gates (L1-L15) |

### Release Gates L1-L15

| Gate | Description | Status |
|------|-------------|--------|
| L1 | StructuredExperience has required fields | PASS |
| L2 | Experience pipeline end-to-end | PASS |
| L3 | Quality scoring correct | PASS |
| L4 | Multi-factor retrieval all factors | PASS |
| L5 | Historical usefulness tracking | PASS |
| L6 | Confidence calibration buckets | PASS |
| L7 | Strategy generation from experiences | PASS |
| L8 | Strategy validation (policy/capability/evidence) | PASS |
| L9 | Decision coupling with impact tracking | PASS |
| L10 | Strategy influences decisions (advisory) | PASS |
| L11 | Adaptive exploration rate and result learning | PASS |
| L12 | Learning gain, failure avoidance, decision influence | PASS |
| L13 | Evaluator aggregation of all metrics | PASS |
| L14 | Governance boundaries: learning advisory, memories untrusted | PASS |
| L15 | Full regression — all previous tests pass | PASS |

### Documentation

| Document | Purpose |
|----------|---------|
| [`docs/V3.1/PRE_IMPLEMENTATION_AUDIT.md`](./docs/V3.1/PRE_IMPLEMENTATION_AUDIT.md) | Pre-implementation audit of V3.0 codebase |
| [`docs/V3.1/VALIDATION_REPORT.md`](./docs/V3.1/VALIDATION_REPORT.md) | V3.1 validation results and release gates |

### Benchmark Scenarios (V3.1)

| File | Purpose |
|------|---------|
| `config/benchmark_scenarios/v3_1_scenarios.json` | 13 scenarios: generalization pairs, failure learning, decision influence, exploration |

## License

MIT
