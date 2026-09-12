# harness/learning/__init__.py — V3.1 Learning Optimization & Decision Intelligence
"""V3.1 Learning Optimization & Decision Intelligence package.

Improves the conversion of accumulated experience into better engineering decisions.
Extends V3.0 foundational learning modules with structured experience extraction,
multi-factor retrieval, learned strategy layer, and decision coupling.
"""

from .context import (
    ExperimentContext,
    ExperimentContextError,
    VALID_MODES,
    LEARNED_MODES,
    GREEDY_NO_EXPLORATION_MODES,
    EXPLORATION_ENABLED_MODES,
)
from .isolation import (
    namespace_path,
    verify_isolation,
    retrieve_filtered,
    check_test_protection,
    get_v32_base,
    find_namespaced_dirs,
)
from .experience import (
    StructuredExperience,
    ExperienceExtractorV31,
    ExperienceQualityScorer,
    ExecutionTrace,
    Evidence,
    ExperiencePipeline,
)
from .retrieval import (
    RetrievalContext,
    RetrievalResult,
    MultiFactorRetriever,
    HistoricalUsefulnessTracker,
    ConfidenceCalibrator,
    RetrievalExplainer,
)
from .strategy import (
    Strategy,
    StrategyGenerator,
    StrategyValidator,
    StrategyStore,
)
from .evaluation import (
    LearningGainCalculator,
    GeneralizationGain,
    FailureAvoidanceMetric,
    DecisionInfluenceMetric,
    LearningEfficiency,
    LearningEvaluator,
    NOT_AVAILABLE,
    NOT_APPLICABLE,
)
from .statistics import (
    SignificanceConfig,
    DEFAULT_SIGNIFICANCE,
    load_significance_config,
    EvaluationObservation,
    ColdBaseline,
    EmpiricalEvaluator,
    build_pairs,
    ComparisonResult,
)
from .exploration import (
    AdaptiveExplorationPolicy,
    PurposefulCandidateSelector,
    ExplorationResultLearner,
)
from .determinism import (
    DeterministicExplorationPolicy,
    ExplorationDecision,
    Candidate,
    derive_exploration_seed,
    context_derived_seed,
    select_greedy,
    sorted_candidates,
    weighted_choice,
    load_exploration_config,
    persist_exploration_decision,
    would_selection_differ,
)
