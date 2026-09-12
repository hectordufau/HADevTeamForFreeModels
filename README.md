# Hermes Agent DevTeam For Free Models

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

---

## V3.2 Learning Integrity, Causal Attribution & Experimental Reproducibility

V3.2 (`v3.2.0`) is the **hardening and measurement-integrity release**. It does not add new agent roles or LLM functionality — it makes the learning system's evidence **trustworthy, reproducible, and causally attributable**. Built on V3.1's learning architecture, it adds:

- **Experiment Context** — A canonical `ExperimentContext` (`validation_run_id`, `benchmark_id`, `task_id`, `execution_id`, `mode`) that attributes every learning artifact and survives process restart
- **Storage Isolation** — Physically namespaced artifact trees per run and mode (`COLD` / `LEARNED` / `LEARNED_NO_EXPLORATION` / `LEARNED_EXPLORATION` / `TEST`), with contamination prevention and a fail-closed cleanup guard that protects historical artifacts
- **Decision Impact Persistence** — `DecisionImpact` is persisted and reloaded across process restarts; `record_outcomes()` writes real execution outcomes to disk (never just in-memory)
- **Explicit Outcome & Attribution Semantics** — `IMPROVED / UNCHANGED / WORSENED / UNKNOWN` outcomes kept separate from `CONFIRMED / PROBABLE / POSSIBLE / NONE` attribution; `UNKNOWN` is never treated as harm or success
- **Empirical COLD Baseline** — Removes the artificial hard-coded `0.5` generalization baseline; baselines are derived from observed COLD outcomes (via `ColdBaseline`)
- **Statistical Evaluation** — Configurable significance thresholds, paired COLD↔LEARNED comparison, 95% bootstrap CIs with deterministic ExperimentContext-derived seeds, Cohen's d effect sizes, and explicit `INSUFFICIENT_DATA`/`NOT_APPLICABLE` sentinels. No evaluation→learning feedback loop
- **Deterministic Exploration** — SHA-256 seed derivation (never Python `hash()`), an experiment-scoped RNG, and greedy `LEARNED_NO_EXPLORATION` vs seeded `LEARNED_EXPLORATION` modes. Same seed + same state ⇒ same decision, across process restarts
- **Failure-Derived Learning** — Failures flow through the canonical `StructuredExperience` pipeline with full provenance; TEST failures are fail-closed (evidence-only, never training) and COLD failures never mutate the baseline
- **Statistical Field Validation** — A 3-mode experiment (COLD / LEARNED_NO_EXPLORATION / LEARNED_EXPLORATION) across TRAIN/VALIDATION/TEST, producing a machine-readable `results.json` as release evidence

### Architecture (V3.2 transversal layer)

```
                    ┌────────────────────────────────────────────┐
                    │          EXPERIMENT CONTEXT                │
                    │  validation_run_id · benchmark_id · task_id│
                    │  execution_id · mode · reproducibility_seed│
                    └────────────────────────────────────────────┘
                                 │
    Experience → Memory → Retrieval → Strategy → Decision → Execution
        ↓              ↓           ↓           ↓         ↓
   (mode/run-tagged)   quality      factors    attribution   outcome
                                 (advisory)                IMPROVED/…
        ↓
    PERSISTED with full provenance (survives restart)
```

### New Modules (V3.2)

| File | Purpose |
|------|---------|
| `harness/learning/context.py` | `ExperimentContext` dataclass: validation, serialization, mode semantics |
| `harness/learning/isolation.py` | `namespace_path`, `verify_isolation`, `retrieve_filtered`, `check_test_protection` |
| `harness/learning/attribution.py` | Deterministic attribution engine (CONFIRMED/PROBABLE/POSSIBLE/NONE), `AttributionMetrics` |
| `harness/learning/statistics.py` | Empirical statistical evaluation: `ColdBaseline`, `EmpiricalEvaluator`, `build_pairs`, `SignificanceConfig`, 95% CIs, p-values, effect sizes |
| `harness/learning/determinism.py` | SHA-256 seed derivation, `DeterministicExplorationPolicy`, `ExplorationDecision`, greedy selection + counterfactual |
| `harness/validation/field_validation.py` | 3-mode statistical field-validation engine |

### Evolved Modules (V3.2)

| Module | Changes |
|--------|---------|
| `harness/planner/v3_integration.py` | `DecisionImpact` persistence + reload, `record_outcomes()` persistence, explicit outcome/attribution semantics, COLD baseline recording |
| `harness/planner/failure_intelligence.py` | `FailureLesson`/`FailureAnalyzer`, fail-closed TEST/COLD, stable SHA-256 identity, idempotency, failure→decision chain |
| `harness/planner/model_intelligence.py` | Wired to deterministic experiment-scoped exploration (no global RNG) |
| `harness/learning/experience.py` | ExperimentContext propagation, fail-closed `process_stored()`, failure provenance |
| `harness/learning/evaluation.py` | `0.5` baseline removed; empirical/configurable baseline + significance |
| `harness/learning/exploration.py` | Seeded local RNG (no global RNG) |
| `config/harness.yaml` | `evaluation.significance` + `learning.exploration` configuration |

### V3.2 Validation (final release evidence `V3.2-FV8-20260912-reprod`)

Reproducibility classification: **R-B — Reproducible After Deterministic Bug Fix** (two defects in `load_benchmark_tasks()` and repetition configuration were found, fixed, and the benchmark rerun on the corrected code).

| Metric | COLD | LEARNED_NO_EXPL | LEARNED_EXPL |
|---|---|---|---|
| Success (TEST) | 42% | 74% | 82% |
| Engineering score (TEST) | 0.4437 | 0.5842 | 0.6331 |
| n (TEST) | 50 | 50 | 50 |

- Split improvement (success pp / score): TRAIN **+26.0 / +0.1017**, VALIDATION **+46.0 / +0.2167**, TEST **+40.0 / +0.1894**
- Decision Improvement Rate: **0.7872** · Learning Harm Rate: **0.0213** · Learning Influence Rate: **0.925**
- Exploration Contribution: **POSITIVE** (+8.0pp success; success-rate significance p=0.1254 — NOT statistically conclusive, honestly reported)
- Integrity: COLD/LEARNED, TEST, and cross-run contamination all **0**
- Security / Governance / Non-free model executions: **0** (free-model invariant preserved)
- Release gates: **26/26 PASS** · Experimental Verdict: **A — VALIDATED** · Recommendation: **GO**

### Documentation (V3.2)

| Document | Purpose |
|----------|---------|
| [`docs/V3.2-SPEC.md`](./docs/V3.2-SPEC.md) | Formal V3.2 specification (phases, 26 release gates, non-goals) |
| [`docs/V3.2-ARCHITECTURE-AUDIT.md`](./docs/V3.2-ARCHITECTURE-AUDIT.md) | Pre-implementation architecture audit |
| [`docs/V3.2-FIELD-VALIDATION-REPORT.md`](./docs/V3.2-FIELD-VALIDATION-REPORT.md) | Final statistical field-validation report (post-fix run) |
| [`docs/V3.2-FIELD-VALIDATION-RESULTS.md`](./docs/V3.2-FIELD-VALIDATION-RESULTS.md) | Machine-readable results summary |
| [`docs/V3.2-RELEASE-GATE.md`](./docs/V3.2-RELEASE-GATE.md) | G1–G26 release gate evaluation |
| [`docs/V3.2-EXPERIMENT-METHODOLOGY.md`](./docs/V3.2-EXPERIMENT-METHODOLOGY.md) | Experimental design and limitations |
| [`docs/V3.2-REPRODUCIBILITY-INVESTIGATION.md`](./docs/V3.2-REPRODUCIBILITY-INVESTIGATION.md) | R-B reproducibility divergence investigation |
| `docs/V3.2-PHASE2..8-REPORT.md` | Per-phase implementation reports |
| `artifacts/v3.2-release/field_validation/results.json` | Original (pre-fix) frozen evidence |
| `artifacts/v3.2-release/field_validation/results_REPROD_20260912.json` | **Final post-fix release evidence** (supersedes the pre-fix run) |

### Version History

```text
v2.1.0  Adaptive Harness + Integrity
v2.2.0  Dynamic Capability-Driven Workflow
v2.2.1  Capability Integrity Patch
v3.0.0  Learning + Optimization Engineering Harness
v3.1.0  Learning Optimization & Decision Intelligence
v3.2.0  Learning Integrity, Causal Attribution & Experimental Reproducibility
```

## License

MIT
