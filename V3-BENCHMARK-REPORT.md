# V3.0 Benchmark Report

## Overview
Full regression benchmark across all 312 tests: V2.1, V2.2, V2.2.1, and V3.0.

## Results

### V3 Unit Tests (71 tests)
| Test File | Tests | Status |
|-----------|-------|--------|
| test_benchmark.py | 11 | ✅ All passed |
| test_evidence_unified.py | 9 | ✅ All passed |
| test_memory_v3.py | 6 | ✅ All passed |
| test_performance_v3.py | 6 | ✅ All passed |
| test_model_intelligence.py | 4 | ✅ All passed |
| test_capability_intelligence.py | 5 | ✅ All passed |
| test_task_intelligence.py | 5 | ✅ All passed |
| test_plan_intelligence.py | 5 | ✅ All passed |
| test_failure_intelligence.py | 6 | ✅ All passed |
| test_context_v3.py | 4 | ✅ All passed |
| test_observability.py | 4 | ✅ All passed |
| test_optimization.py | 6 | ✅ All passed |

### V3 Integration Tests (8 tests)
| Test | Status |
|------|--------|
| test_full_pipeline_with_benchmark_and_evidence | ✅ |
| test_pipeline_with_memory_and_performance | ✅ |
| test_pipeline_with_capability_intelligence | ✅ |
| test_pipeline_with_task_and_plan_intelligence | ✅ |
| test_pipeline_with_failure_recovery | ✅ |
| test_pipeline_with_context_and_policy | ✅ |
| test_pipeline_with_optimization_and_observability | ✅ |
| test_free_model_invariant_throughout_pipeline | ✅ |

### V3 Release Gates (25 gates)
| Gate | Description | Status |
|------|-------------|--------|
| DG3-01 | Benchmark scenario registration | ✅ |
| DG3-02 | Execution statuses (PASSED/FAILED/ERROR) | ✅ |
| DG3-03 | Metrics collector aggregation | ✅ |
| DG3-04 | Evidence hash chain consistency | ✅ |
| DG3-05 | Evidence query by task_id and node_id | ✅ |
| DG3-06 | Performance registry record and aggregate | ✅ |
| DG3-07 | Performance ranking and best_model | ✅ |
| DG3-08 | Free model invariant enforcer | ✅ |
| DG3-09 | Task classifier | ✅ |
| DG3-10 | Task validator removes unknowns | ✅ |
| DG3-11 | Synonym resolver canonicalization | ✅ |
| DG3-12 | Capability gap detector | ✅ |
| DG3-13 | Capability recommender | ✅ |
| DG3-14 | Context budget management | ✅ |
| DG3-15 | Context builder produces valid pack | ✅ |
| DG3-16 | Governance blocks paid models | ✅ |
| DG3-17 | Governance blocks self-modification | ✅ |
| DG3-18 | Governance enforces mandatory gates | ✅ |
| DG3-19 | Security regression suite | ✅ |
| DG3-20 | Prompt injection detection | ✅ |
| DG3-21 | Optimization engine | ✅ |
| DG3-22 | Root cause analysis | ✅ |
| DG3-23 | Replanning optimizer | ✅ |
| DG3-24 | Adaptive model policy | ✅ |
| DG3-25 | Experience store | ✅ |

### V3 E2E Tests (2 tests)
| Test | Status |
|------|--------|
| test_e2e_v3_learning_loop | ✅ |
| test_e2e_v3_adaptive_policy_cycle | ✅ |

### V2.1 / V2.2 / V2.2.1 Regression (~206 tests)
| Test Suite | Status |
|------------|--------|
| V2.1 unit tests | ✅ All passed |
| V2.2 unit tests | ✅ All passed |
| V2.2.1 capability tests | ✅ All passed |
| V2.1 integration tests | ✅ All passed |
| V2.2 integration tests | ✅ All passed |
| V2.2 release gates (DG1-DG10) | ✅ All passed |
| V2.1 E2E tests | ✅ All passed |
| V2.2 E2E tests | ✅ All passed |
| V2.0 release gates (G1-G10) | ✅ All passed |

## Summary
| Metric | Value |
|--------|-------|
| Total tests | 312 |
| Passed | 312 |
| Failed | 0 |
| Pass rate | 100% |
| Execution time | 2.19s |

## Files Changed for V3
- `harness/routing/performance_v3.py` — Empty file handling in `_load()`, fixed key splitting in `rank_models()`
- `harness/context/v3.py` — Fixed `ContextBudget` defaults to respect overridden `max_tokens`
- `tests/unit/test_capability_intelligence.py` — Fixed taxonomy path resolution
- `tests/unit/test_task_intelligence.py` — Fixed taxonomy path resolution (2 locations)
