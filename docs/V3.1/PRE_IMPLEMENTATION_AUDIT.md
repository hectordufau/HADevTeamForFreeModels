# V3.1 Pre-Implementation Audit

## Baseline
- **Commit:** 78e2e7e (V3.0.0: Learning + Optimization Engineering Harness)
- **Test Count:** 312 passing
- **Branch:** `v3.0.0` tag (detached HEAD)

## Modules Inspected

### 1. `harness/learning/` (directory)
```yaml
module: harness.learning
location: harness/learning/
status: MISSING
existing_interfaces: []
existing_data_structures: []
current_limitations:
  - Directory exists but contains ONLY __pycache__ — zero source files
  - No experience extraction pipeline
  - No memory retrieval system
  - No learned strategy layer
  - No learning evaluation module
v31_requirements_met: []
v31_changes_needed:
  - Create harness/learning/__init__.py
  - Create harness/learning/experience.py — StructuredExperience, ExperienceExtractorV31, ExperienceQualityScorer, ExperiencePipeline
  - Create harness/learning/retrieval.py — MultiFactorRetriever, HistoricalUsefulnessTracker, ConfidenceCalibrator, RetrievalExplainer
  - Create harness/learning/strategy.py — Strategy, StrategyGenerator, StrategyValidator, StrategyStore
  - Create harness/learning/evaluation.py — LearningGainCalculator, GeneralizationGain, FailureAvoidanceMetric, DecisionInfluenceMetric, LearningEfficiency, LearningEvaluator
potential_regressions:
  - None (new package, no existing code to break)
```

### 2. `harness/memory/__init__.py`
```yaml
module: harness.memory
location: harness/memory/__init__.py
status: EXISTS
existing_interfaces:
  - MemoryManager class
  - MemoryEntry dataclass
  - store(entry), retrieve(key, category), search(query, category)
  - store_adr(adr_id, context, decision, alternatives, consequences)
existing_data_structures:
  - MemoryEntry: key, content, category, tags, timestamp
  - JSON file-based storage with subdirectories: project, task, agent, lessons
current_limitations:
  - Basic file-based JSON persistence (no indexing, no caching)
  - search() uses simple substring matching (no relevance scoring)
  - No relationship tracking between entries
  - No confidence or quality scoring
  - No deduplication
  - V2.1 era design, no integration with V3 learning modules
v31_requirements_met:
  - Basic persistent storage mechanism exists (can be extended)
  - ADR storage provides a known pattern
v31_changes_needed:
  - May need integration adapter to bridge with V3.1 ExperienceStore/ExperiencePipeline
  - Consider migration path from MemoryEntry to StructuredExperience format
potential_regressions:
  - Existing callers of MemoryManager may break if signatures change
  - 5 tests rely on MemoryManager (test_memory.py)
```

### 3. `harness/memory/v3.py`
```yaml
module: harness.memory.v3
location: harness/memory/v3.py
status: EXISTS (NEEDS_EVOLUTION)
existing_interfaces:
  - ExperienceExtractor.extract(report, task) -> ExperienceRecord
  - RelevanceScorer.score(entry, query) -> float
  - ExperienceStore class: store, retrieve, search, get_confidence, record_outcome
  - ConfidenceTracker: record_access(outcome), to_dict
existing_data_structures:
  - ExperienceRecord: task_id, task_objective, capabilities_used, result_status, score, duration_seconds, lessons
  - MemoryScore: entry_id, content, relevance, confidence, source, category
  - ConfidenceTracker: entry_id, confidence, access_count, last_accessed, success_count, failure_count
current_limitations:
  - ExperienceRecord has flat structure — no task_context, strategy, structured failures, evidence links
  - ExperienceExtractor extracts from V2 execution report format only
  - RelevanceScorer uses only keyword overlap (0.6) + capability overlap (0.3) + status bonus (0.1) — no multi-factor retrieval
  - ConfidenceTracker uses simple Bayesian average — no calibration from historical outcomes
  - ExperienceStore.search() iterates all files linearly — no indexing
  - No retrieval explainability (why was this experience selected?)
  - No historical usefulness tracking (did retrieved experience improve outcome?)
  - Duration defaults to 0.0 (not properly piped from execution)
  - model_id defaults to "default:free" in v3_integration.py callers
v31_requirements_met:
  - Experience storage/retrieval infrastructure exists
  - Basic confidence tracking exists (needs calibration extension)
  - Relevance scoring base exists (needs multi-factor extension)
  - ExperienceRecord can be converted to StructuredExperience via from_v3_experience()
v31_changes_needed:
  - Keep as backward-compatible V3 store
  - Add adapter from V3 ExperienceRecord → V3.1 StructuredExperience (already in experience.py plan)
  - V3.1's ExperiencePipeline provides the upgraded path
potential_regressions:
  - 8 tests in test_memory_v3.py — must continue passing
  - Existing callers (v3_integration.py, test_v3_integration.py) must not break
```

### 4. `harness/routing/performance_v3.py`
```yaml
module: harness.routing.performance_v3
location: harness/routing/performance_v3.py
status: EXISTS
existing_interfaces:
  - PerformanceRegistryV3: record(sample), get_stats(model_id, capability), rank_models(capability), best_model(capability)
  - PerformanceSample dataclass
  - AggregatedStats dataclass with confidence_score()
existing_data_structures:
  - PerformanceSample: task_id, model_id, capability, success, score, latency_ms, iterations, timestamp
  - AggregatedStats: model_id, capability, sample_count, success_rate, avg_score, avg_latency_ms, ...
  - MIN_SAMPLES_REQUIRED = 5
current_limitations:
  - Statistical aggregation is basic: mean/std only, no percentiles, no trends
  - Sample storage is monolithic JSON (may be slow with large datasets)
  - No decay/aging of old samples
  - No cross-capability correlation
  - rank_models() string-key parsing on ":" is fragile for model IDs containing colons
v31_requirements_met:
  - Statistical aggregation framework exists
  - Minimum sample policy exists
  - Confidence-aware ranking exists
v31_changes_needed:
  - May need performance improvements for large registries
  - Could benefit from trend analysis for exploration decisions
potential_regressions:
  - 4 tests in test_performance_v3.py — must continue passing
  - Direct integration with AdaptiveModelPolicyV3
```

### 5. `harness/planner/model_intelligence.py`
```yaml
module: harness.planner.model_intelligence
location: harness/planner/model_intelligence.py
status: EXISTS (NEEDS_EVOLUTION)
existing_interfaces:
  - AdaptiveModelPolicyV3: select(caps, role, task_id, force_exploit) -> Selection
  - ModelProfileStore: save_profile, load_profile, list_profiles
  - ExperimentTracker: record, get_results(model_id, capability)
  - FreeModelInvariantEnforcer: enforce(model_id), validate_workflow(workflow)
existing_data_structures:
  - ModelSpecializationProfile: model_id, capabilities dict, specialization_score, top_capabilities
  - ModelExperiment: experiment_id, model_id, capability, task_id, result_score, is_exploration
  - Selection (inline class): model_id, provider, cost, reason, fallback_used, is_exploration
current_limitations:
  - Fixed 10% exploration rate regardless of confidence level
  - Exploration picks random free model — no purposeful candidate selection
  - No exploration result learning (doesn't learn from what exploration found)
  - No adaptive rate adjustment based on confidence
  - ProfileStore doesn't integrate with PerformanceRegistryV3 for automatic updates
  - ModelSpecializationProfile scores are static (set once, never updated)
v31_requirements_met:
  - Exploration/exploitation framework exists
  - Free model invariant enforcement exists
  - Experiment tracking exists
  - Model profile storage exists
v31_changes_needed:
  - Add ExplorationResultLearner to learn from exploration outcomes
  - Enhance AdaptiveModelPolicyV3 with adaptive exploration rate (2-20% based on confidence)
  - Add purposeful candidate selection instead of random
potential_regressions:
  - 7 tests in test_model_intelligence.py — must continue passing
  - Tests rely on fixed 10% exploration rate
  - v3_integration.py and test_v3_integration.py call these interfaces
```

### 6. `harness/planner/optimization.py`
```yaml
module: harness.planner.optimization
location: harness/planner/optimization.py
status: EXISTS
existing_interfaces:
  - OptimizationEngine: add_target, add_constraint, optimize_workflow, get_audit_trail, get_targets, get_constraints, clear
  - OptimizationTarget dataclass
  - OptimizationConstraint dataclass
  - OptimizationResult dataclass
  - OptimizationAuditEntry dataclass
existing_data_structures:
  - OptimizationTarget: name, metric, direction, weight, current_value, target_value
  - OptimizationConstraint: name, condition, value, current_value, violated
  - OptimizationResult: target_name, original_value, optimized_value, improvement_pct, constraints_met, iterations, details
  - OptimizationAuditEntry: optimization_id, action, target, parameters, result, timestamp
current_limitations:
  - _simulate_optimization() uses hardcoded ±20% improvement — not real optimization
  - No integration with real metrics from execution results
  - No workflow parameter mutation capability
  - No multi-objective optimization (handles one primary target at a time)
  - Audit trail is in-memory only, no persistence
  - No link to PerformanceRegistryV3 for data-driven optimization
v31_requirements_met:
  - Framework structure for targets, constraints, and audit exists
  - Interface for plugging in real optimization logic in place
v31_changes_needed:
  - V3.1 likely does NOT need changes to this module directly (it's a Phase M module)
  - Learning→Decision coupling may feed optimized params to this engine
potential_regressions:
  - 4 tests in test_optimization.py — must continue passing
```

### 7. `harness/planner/v3_integration.py`
```yaml
module: harness.planner.v3_integration
location: harness/planner/v3_integration.py
status: EXISTS (NEEDS_EVOLUTION)
existing_interfaces:
  - UnifiedLearningPipeline: process(task, report), get_learning_summary()
  - UnifiedDecisionPipeline: decide_capabilities, decide_model, decide_plan, decide_replan, check_governance
  - OrchestratorV3: run(task_id, workflow_name)
  - remove_obsolete_compatibility(force)
existing_data_structures:
  - V3IntegrationError exception
  - No structured data classes (uses basic dicts for cross-component data)
current_limitations:
  - No DecisionContext data class (task + capability + policy + experience + strategy + performance context)
  - Model selection has no strategy influence
  - Agent selection doesn't use strategy or history
  - Workflow planning doesn't consider learned strategies
  - No decision impact tracking (was decision changed? was outcome improved?)
  - Task-level performance recording defaults to "default:free" model_id (incomplete)
  - Failure learning in UnifiedLearningPipeline uses hardcoded categories/severity
  - Learning pipeline silently swallows all exceptions (pass on error)
v31_requirements_met:
  - OrchestratorV3 delegation pattern is correct (coordinator, not implementer)
  - UnifiedLearningPipeline processes execution through all learning pipelines
  - UnifiedDecisionPipeline coordinates all intelligence components
  - Infrastructure for plugging in new components exists
v31_changes_needed:
  - Add DecisionContext data class
  - Add DecisionImpact and DecisionImpactTracker
  - Enhance UnifiedDecisionPipeline with strategy influence
  - Add impact tracking to all decision points
  - Fix model_id recording to use actual selected model rather than "default:free"
potential_regressions:
  - 3 tests in test_v3_integration.py — must continue passing
  - 6 tests in test_v21.py (interface tests)
  - Any change that alters OrchestratorV3.run() signature or return format
```

### 8. `harness/planner/plan_intelligence.py`
```yaml
module: harness.planner.plan_intelligence
location: harness/planner/plan_intelligence.py
status: EXISTS
existing_interfaces:
  - PlanScorer: score(workflow) -> PlanScore
  - AlternativeGenerator: generate(requirements, count) -> List[WorkflowAlternative]
  - PlanSelector: select(alternatives) -> WorkflowAlternative
  - PlanLearningStore: record_outcome(plan_id, score, result_status), get_best_scoring_pattern()
existing_data_structures:
  - PlanScore: overall, capability_coverage, agent_fit, model_quality, efficiency, risk
  - WorkflowAlternative: id, nodes, score, generated_by, timestamp
  - PlanLearningStore uses JSON file storage with timestamped entries
current_limitations:
  - PlanScorer._score_model_quality() always returns 0.7 (no real assessment)
  - AlternativeGenerator._merge_similar_nodes() is simplistic
  - Alternative generation only creates 3 alternatives (primary, reversed, consolidated)
  - No integration with Strategy layer for plan generation
  - PlanLearningStore.get_best_scoring_pattern() doesn't weight by recency
  - No plan improvement suggestion based on past failures
v31_requirements_met:
  - Plan scoring framework exists with multiple dimensions
  - Alternative generation exists (can be extended)
  - Plan selection and learning store exist
v31_changes_needed:
  - V3.1 plan may integrate Strategy objects with AlternativeGenerator
  - PlanLearningStore could participate in Strategy generation
potential_regressions:
  - 6 tests in test_plan_intelligence.py — must continue passing
```

### 9. `harness/planner/failure_intelligence.py`
```yaml
module: harness.planner.failure_intelligence
location: harness/planner/failure_intelligence.py
status: EXISTS
existing_interfaces:
  - RootCauseAnalyzer: analyze(failure, task_context), get_patterns()
  - FailureToLearningPipeline: process(failure) -> lesson dict
  - ReplanningOptimizer: should_replan(task_id, failure), get_optimal_replan_strategy(failure), record_attempt(task_id)
existing_data_structures:
  - StructuredFailure: node_id, capability, category, sub_category, severity, error_message, evidence, timestamp
  - FAILURE_TAXONOMY dict with 7 categories and sub-categories
  - FAILURE_TAXONOMY: implementation, test, architecture, security, integration, configuration, environment
current_limitations:
  - _classify_failure() uses substring matching against hardcoded keywords — fragile
  - No integration with V3.1 StructuredExperience for richer failure records
  - ReplanningOptimizer uses simple attempt counting, no state persistence
  - Failure categories and strategies are hardcoded strings
  - No failure trending or anomaly detection
  - RootCauseAnalyzer._analysis_history is in-memory only
v31_requirements_met:
  - Structured failure taxonomy exists
  - Root cause analysis framework exists
  - Failure-to-learning pipeline exists
  - Replanning optimization exists with strategy selection
v31_changes_needed:
  - Failure-to-learning pipeline should produce V3.1 StructuredExperience (it already creates ExperienceRecord)
  - May need richer failure representation to feed Strategy generation
potential_regressions:
  - 5 tests in test_failure_intelligence.py — must continue passing
  - v3_integration.py creates StructuredFailure objects that must remain compatible
```

### 10. `harness/capabilities/intelligence.py`
```yaml
module: harness.capabilities.intelligence
location: harness/capabilities/intelligence.py
status: EXISTS
existing_interfaces:
  - SynonymResolver: resolve(capability), is_synonym(cap_a, cap_b), canonical(capability)
  - CapabilityGapDetector: detect_gaps(required_capabilities, scenario_tags), get_coverage_report()
  - CapabilityRecommender: recommend(task_objective, acceptances, limit) -> List[CapabilityRecommendation]
  - CapabilityQualityAnalyzer: analyze(capability) -> CapabilityQualityMetrics
existing_data_structures:
  - CapabilityGap: capability, gap_type, severity, description, affected_scenarios, recommendation
  - CapabilityRecommendation: capability, recommended_action, priority, rationale, expected_impact
  - CapabilityQualityMetrics: capability, usage_count, success_rate, avg_score, agent_count, scenario_count, quality_score
  - SYNONYMS dict (11 canonical entries with synonym lists)
current_limitations:
  - SynonymResolver has only 11 hardcoded synonym groups
  - CapabilityRecommender uses simple keyword matching — no semantic understanding
  - CapabilityQualityMetrics.quality_score weights are arbitrary (0.3/0.4/0.3)
  - No automatic synonym discovery or learning
  - No integration with V3.1 experience store for capability quality scoring
  - No connection to Strategy layer for capability recommendations
v31_requirements_met:
  - Capability gap detection framework exists
  - Capability recommendation exists
  - Quality analysis infrastructure exists
  - Synonym resolution exists (basic but functional)
v31_changes_needed:
  - V3.1 likely does NOT need changes to this module directly
  - Could benefit from experience-driven capability recommendations
potential_regressions:
  - 8 tests in test_capability_intelligence.py — must continue passing
```

### 11. `harness/evidence/unified.py`
```yaml
module: harness.evidence.unified
location: harness/evidence/unified.py
status: EXISTS
existing_interfaces:
  - EvidenceStore: store_evidence(package), store_record(record), verify_integrity(task_id, execution_id)
  - EvidenceStore: get_record(task_id, execution_id), get_evidence(task_id, execution_id), query(query)
  - CanonicalExecutionRecord dataclass
  - EvidencePackage dataclass
  - EvidenceQuery dataclass
existing_data_structures:
  - CanonicalExecutionRecord: task_id, execution_id, workflow_name, nodes, capability_requirements, assignments, results, evidence_hashes, timestamp, previous_hash
  - EvidencePackage: node_id, capability, agent, task_id, execution_id, data, content_hash, previous_hash, timestamp
  - EvidenceQuery: task_id, execution_id, node_id, capability, limit, offset
  - Hash-chain index: chain_index.json (list of hash entries)
current_limitations:
  - No integration with V3.1 structured experiences
  - EvidencePackage stores data as generic dict — no schema enforcement
  - Query API is file-system based (no indexing for performance)
  - No evidence quality scoring or filtering
  - No evidence expiry or archival strategy
v31_requirements_met:
  - Hash-chain integrity framework is complete and robust
  - Evidence storage and retrieval exists
  - Evidence integrity verification works
  - Can be used as source of evidence for V3.1 Experience extraction
v31_changes_needed:
  - V3.1 ExperienceExtractorV31 should use evidence from this store to build StructuredExperience.evidence list
  - No structural changes needed to evidence store itself
potential_regressions:
  - 6 tests in test_evidence_unified.py — must continue passing
```

### 12. `harness/context/v3.py`
```yaml
module: harness.context.v3
location: harness/context/v3.py
status: EXISTS
existing_interfaces:
  - ContextBuilderV3: build_context(task, agent_role, agent_soul, relevant_files, memory_context, evidence_context) -> ContextPackV3
  - ContextRelevanceScorer: score(item, query) -> float
  - ContextBudget: can_add, add, remove, remaining
  - ContextPackV3: add(item), to_dict()
existing_data_structures:
  - ContextItem: content, source, relevance, priority, category, estimated_tokens
  - ContextBudget: max_tokens, reserved_tokens, available_tokens, used_tokens
  - ContextPackV3: agent_role, budget, items[]
current_limitations:
  - ContextRelevanceScorer uses same keyword overlap as RelevanceScorer — no semantic understanding
  - Memory context limited to 20 items, evidence to 10 (hardcoded)
  - No strategy context inclusion
  - No dynamic budget adjustment based on task complexity
  - Priority system is simple (1-10, lower = higher priority)
  - Estimated token counts are rough approximations (len/4)
v31_requirements_met:
  - Context budget management works
  - Priority-based inclusion exists
  - Source-aware ordering exists
  - Relevance scoring exists (can be extended with multi-factor)
v31_changes_needed:
  - V3.1 should include strategy context when building context for agents
  - Memory retrieval from V3.1 MultiFactorRetriever should feed memory_context
potential_regressions:
  - 4 tests in test_context_v3.py — must continue passing
```

### 13. `harness/planner/observability.py`
```yaml
module: harness.planner.observability
location: harness/planner/observability.py
status: EXISTS
existing_interfaces:
  - StructuredLogger: log, info, warning, error, debug, get_recent, flush
  - DecisionTracer: record(decision), get_decisions(trace_id)
  - ExecutionTracer: begin(component, action, input_summary), end(node_id, status, output_summary, error), get_trace, reset
existing_data_structures:
  - LogEntry: level, module, message, context, timestamp, trace_id
  - DecisionRecord: decision_id, decision_type, context, alternatives, selected, rationale, timestamp, trace_id
  - ExecutionTraceNode: node_id, component, action, duration_ms, status, input_summary, output_summary, children, error, timestamp
current_limitations:
  - DecisionTracer stores decisions as individual JSON files (lots of small files)
  - ExecutionTracer is in-memory only (no persistence)
  - No integration with V3.1 decision impact tracking
  - No structured logging integration with learning pipeline
  - Logger stores entries in memory until flush() is called
v31_requirements_met:
  - Decision tracing exists for all decision types
  - Structured logging with module/level/context exists
  - Execution trace tree exists for observability
v31_changes_needed:
  - DecisionTracer should integrate with DecisionImpactTracker for V3.1
  - May need decision log format that includes learning influence flag
potential_regressions:
  - 6 tests in test_observability.py — must continue passing
```

### 14. `harness/benchmark/` (directory)
```yaml
module: harness.benchmark
location: harness/benchmark/
status: EXISTS
existing_interfaces:
  - BenchmarkEngine: register_scenario, load_scenarios, run_scenario, run_all, get_results, get_summary
  - MetricsCollector: record_result, get_metrics, reset
  - BenchmarkReportGenerator: generate(results, summary, baseline, tags), save(report, filename)
existing_data_structures:
  - BenchmarkScenario: id, name, description, task_objective, acceptance_criteria, allowed_changes, verification_required, autonomy_level, expected_capabilities, tags, timeout_seconds
  - BenchmarkResult: scenario_id, scenario_name, status, duration_seconds, execution_status, capabilities_used, agent_assignments, scoring, error, details, timestamp
  - ExecutionMetrics: execution_count, success_count, failure_count, total_duration_seconds, min/max/avg/median/p95, capabilities_used, agent_usage, model_usage, rejection_reasons
current_limitations:
  - BenchmarkEngine runs scenarios sequentially (no parallelism)
  - No built-in generalization pairs (train/eval split)
  - No failure-learning scenarios in default configuration
  - No decision-influence measurement in benchmark results
  - No learning A/B testing (with/without learning comparison)
  - Scenario loading from single JSON file only
v31_requirements_met:
  - Scenario registration and execution framework exists
  - Metrics collection and aggregation works
  - Report generation with baseline comparison exists
  - Capability coverage scoring exists
v31_changes_needed:
  - Add generalization scenarios (train/eval splits)
  - Add failure-learning scenarios
  - Add decision-influence measurement
  - Support A/B testing (learning on/off comparison)
potential_regressions:
  - 6 tests in test_benchmark.py — must continue passing
```

## Summary

### Total Modules Inspected: 14

| # | Module | Status | Changes Needed |
|---|--------|--------|---------------|
| 1 | harness/learning/ | MISSING | Create 5 new files |
| 2 | harness/memory/__init__.py | EXISTS | Bridge/adapter for V3.1 |
| 3 | harness/memory/v3.py | NEEDS_EVOLUTION | Extension + backward compat |
| 4 | harness/routing/performance_v3.py | EXISTS | Minor improvements |
| 5 | harness/planner/model_intelligence.py | NEEDS_EVOLUTION | Adaptive exploration + learning |
| 6 | harness/planner/optimization.py | EXISTS | No changes needed |
| 7 | harness/planner/v3_integration.py | NEEDS_EVOLUTION | DecisionContext, impact tracking, strategy influence |
| 8 | harness/planner/plan_intelligence.py | EXISTS | Integration with Strategy layer |
| 9 | harness/planner/failure_intelligence.py | EXISTS | Richer failure→experience pipeline |
| 10 | harness/capabilities/intelligence.py | EXISTS | No changes needed |
| 11 | harness/evidence/unified.py | EXISTS | Evidence source for V3.1 experiences |
| 12 | harness/context/v3.py | EXISTS | Strategy context inclusion |
| 13 | harness/planner/observability.py | EXISTS | Decision impact integration |
| 14 | harness/benchmark/ | EXISTS | New scenarios + A/B testing |

### Modules That Need Changes: 6

1. **harness/learning/** — Create from scratch (5 files)
2. **harness/memory/v3.py** — Extension for V3.1 compatibility
3. **harness/planner/model_intelligence.py** — Adaptive exploration + learning
4. **harness/planner/v3_integration.py** — Decision context, impact tracking
5. **harness/planner/plan_intelligence.py** — Strategy integration
6. **harness/planner/failure_intelligence.py** — Richer failure pipeline

### Modules That Already Satisfy V3.1 Requirements (partial):
- **harness/memory/v3.py** — Experience storage/retrieval, confidence tracking (needs extension)
- **harness/routing/performance_v3.py** — Statistical aggregation, ranking (minor improvements)
- **harness/planner/model_intelligence.py** — Exploration/exploitation framework (needs adaptive rate)
- **harness/capabilities/intelligence.py** — Gap detection, recommendation (satisfies current needs)
- **harness/evidence/unified.py** — Hash-chain integrity (satisfies current needs)
- **harness/context/v3.py** — Budget management, relevance scoring (satisfies current needs)
- **harness/planner/observability.py** — Decision tracing (satisfies current needs)
- **harness/benchmark/** — Execution framework, metrics, reporting (needs new scenarios)

### Estimated Implementation Effort

| Module | Effort | Rationale |
|--------|--------|-----------|
| harness/learning/experience.py | L | Rich data classes, quality scoring, extraction pipeline, backward compat |
| harness/learning/retrieval.py | L | Multi-factor retrieval, historical tracking, confidence calibration, explainability |
| harness/learning/strategy.py | M | Strategy representation, generation from experiences, validation |
| harness/learning/evaluation.py | M | 5 evaluation metrics + aggregator |
| harness/planner/model_intelligence.py enhancements | S | Add ExplorationResultLearner, adaptive rate |
| harness/planner/v3_integration.py enhancements | M | DecisionContext, DecisionImpact, strategy coupling |
| Tests (6 new test files) | L | ~60+ new tests (unit + integration + release gates) |

### Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Regression in 312 existing tests | LOW | Full test suite run after every change; backward-compatible design |
| Learning bypasses governance | LOW | StrategyValidator enforces policy checks before any learning-based change |
| Memory/experience data treated as untrusted | LOW | PromptInjectionBoundary scans all memory content before retrieval |
| Performance degradation from multi-factor retrieval | MEDIUM | Implement caching and indexing if file-based search becomes bottleneck |
| Dead code in v3_integration.py | LOW | Existing `remove_obsolete_compatibility()` function already identifies candidates |
| Over-engineering retrieval | MEDIUM | Start with keyword-based multi-factor; add semantic only if needed |

### V3.0 Functionality That Satisfies V3.1 Requirements (can reuse):
- ✅ Experience storage and retrieval infrastructure (memory/v3.py)
- ✅ Confidence tracking (basic Bayesian) exists — needs calibration extension
- ✅ Relevance scoring exists — needs multi-factor extension
- ✅ Exploration/exploitation framework exists — needs adaptive rate
- ✅ Free model invariant enforcement — complete and robust
- ✅ Governance/policy framework — complete with immutable rules
- ✅ Benchmark engine and report generation — needs new scenario types
- ✅ Evidence hash-chain integrity — complete
- ✅ Capability gap detection and recommendations — complete
- ✅ Plan scoring and alternative generation — complete
- ✅ Failure taxonomy and analysis framework — complete
- ✅ Context budget management — complete
- ✅ Decision tracing and observability — complete

### Recommendations
1. **Preserve backward compatibility** — V3.1 should not break any V3.0 interface. Add adapters (like `from_v3_experience()`) rather than changing existing signatures.
2. **Learning is advisory** — StrategyGenerator produces suggestions; StrategyValidator enforces policy checks before any learning-based change reaches execution.
3. **Evidence-first quality** — No high-confidence lesson without supporting evidence. Quality = evidence_quality × outcome_quality × extraction_confidence.
4. **Purposeful exploration** — Replace random exploration with information-gain scoring using prior results.
5. **Decision impact tracking** — Every learning-influenced decision must be logged with outcome for retrospective analysis.
