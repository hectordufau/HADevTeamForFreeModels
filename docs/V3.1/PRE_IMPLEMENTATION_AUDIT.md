# V3.1 Pre-Implementation Audit

## Baseline
- **Commit:** 78e2e7e (V3.0.0: Learning + Optimization Engineering Harness)
- **Test Count:** 312 passing
- **Learning Gain (V3.0):** +2.00% success rate (74%→76%), +0.0045 score (0.7184→0.7229)

## Existing V3 Modules Assessment

### Phase A — Experience Extraction (harness/memory/v3.py)
**Current state:** `ExperienceExtractor` produces `ExperienceRecord` with basic fields (task_id, task_objective, capabilities_used, result_status, score, duration, lessons)
**Limitations:**
- No structured model (context, strategy, outcome, failures, lesson, confidence, evidence)
- No extraction pipeline (Task → Execution Trace → Evidence → Structured Experience)
- No quality scoring (evidence_quality × outcome_quality × extraction_confidence)
- `lessons` is a simple string list, no structured lesson metadata

### Phase B — Memory Retrieval (harness/memory/v3.py)
**Current state:** `RelevanceScorer` uses keyword overlap (0.6) + capability overlap (0.3) + status bonus (0.1)
**Limitations:**
- No multi-factor retrieval (similarity, historical_usefulness, applicability, confidence, evidence_quality, recency)
- No historical usefulness tracking (did retrieved experience improve outcome?)
- Confidence calibration is flat (all entries median 0.5)
- No retrieval explainability (why was this experience selected?)

### Phase C — Learned Strategy Layer
**Current state:** No strategy representation exists. `PlanLearningStore` stores plan patterns but no concept of strategy.
**Limitations:**
- No Strategy object (task_class, capabilities, conditions, workflow, preferred_models/agents, expected_outcome)
- No strategy generation from accumulated experiences
- No strategy validation (policy check, capability check, evidence check)

### Phase D — Learning→Decision Coupling (harness/planner/v3_integration.py)
**Current state:** `UnifiedDecisionPipeline` delegates to individual components but no standardized decision context
**Limitations:**
- No `DecisionContext` (TaskContext + CapabilityContext + PolicyContext + ExperienceContext + StrategyContext + PerformanceContext)
- Model selection has no strategy influence
- Agent selection doesn't use strategy or history
- Workflow planning doesn't consider learned strategies
- No decision impact tracking (was decision changed? was outcome improved?)
- Task-level performance recording defaults to "default:free" model_id — incomplete

### Phase E — Adaptive Exploration (harness/planner/model_intelligence.py)
**Current state:** Fixed 10% exploration rate, random selection from free models
**Limitations:**
- Fixed 10% rate regardless of confidence
- Random selection — no purposeful candidate selection
- No explicit exploration result learning

### Phase F — Learning Evaluation
**Current state:** No dedicated learning evaluation module
**Limitations:**
- No `LearningGainCalculator`
- No `GeneralizationGain`
- No `FailureAvoidanceMetric`
- No `DecisionInfluenceMetric`
- No `LearningEfficiency`

### Phase G — V3.1 Benchmark
**Current state:** V3.0 benchmark dataset exists but lacks V3.1 scenarios
**Limitations:**
- No generalization pairs in benchmark
- No failure-learning scenarios
- No decision-influence scenarios

### Phase H — Governance & Regression
**Current state:** Security, governance, prompt injection boundaries exist
**Limitations:**
- No specific test that learning cannot disable security/governance
- No test that memories are treated as untrusted data

### Phase I — Release Gates
**Current state:** 25 gates (DG3-01 to DG3-25)
**Required for V3.1:** 15 additional gates (L1-L15)

## Summary of Required Changes

| Phase | New Files | Modified Files | New Tests |
|-------|-----------|----------------|-----------|
| A — Experience 2.0 | harness/learning/experience.py | — | 6 |
| B — Memory 2.0 | harness/learning/retrieval.py | harness/learning/__init__.py | 6 |
| C — Strategy | harness/learning/strategy.py | — | 5 |
| D — Coupling | — | harness/planner/v3_integration.py | 4 |
| E — Exploration | — | harness/planner/model_intelligence.py | 3 |
| F — Evaluation | harness/learning/evaluation.py | — | 6 |
| G — Benchmark | — | tests/unit/test_learning_*.py | 6+ |
| H — Governance | — | tests/integration/test_v31_release_gate.py | 15 |
| I — Release | — | run_validation.py, README.md | — |

## V3.0 Functionality That Satisfies V3.1 Requirements (partial)
- Experience storage and retrieval infrastructure exists
- Confidence tracking (basic Bayesian) exists but needs calibration
- Relevance scoring exists but needs extension
- Exploration/exploitation framework exists but needs adaptive rate
- Free model invariant enforcement exists
- Governance/policy framework exists
- Benchmark engine and report generation exist
