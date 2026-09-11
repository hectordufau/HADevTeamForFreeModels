# harness/learning/__init__.py — V3.1 Learning Optimization & Decision Intelligence
"""V3.1 Learning Optimization & Decision Intelligence package.

Improves the conversion of accumulated experience into better engineering decisions.
Extends V3.0 foundational learning modules with structured experience extraction,
multi-factor retrieval, learned strategy layer, and decision coupling.
"""

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
)
from .exploration import (
    AdaptiveExplorationPolicy,
    PurposefulCandidateSelector,
    ExplorationResultLearner,
)
