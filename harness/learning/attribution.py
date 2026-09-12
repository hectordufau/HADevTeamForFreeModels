# harness/learning/attribution.py — V3.2 Phase 4: Decision Attribution & Confidence
"""
Deterministic, evidence-based attribution of decisions to learning.

This module answers: *"did learning causally influence this decision, and can
the outcome be attributed to it?"* It is deliberately conservative. It never
claims causality merely because a LEARNED score beats a COLD score; a strong
comparative result is treated as *supporting* evidence, never proof.

Key concepts
------------
attribution_level (CONFIRMED / PROBABLE / POSSIBLE / NONE):
    How confident we are that learning *caused* the decision change AND that
    the outcome is measurable and attributable to that change.

attribution_confidence (0..1):
    A documented, deterministic, explainable confidence score. Every result
    carries a `rationale` explaining why it was classified as it was.

Output semantics (mirrors Phase 3):
    - outcome: IMPROVED | UNCHANGED | WORSENED | UNKNOWN
    - UNKNOWN is never treated as harm and is never silently dropped.

No synthetic evidence is created here. This module only *classifies* the
actual evidence already persisted by the reader / decision pipeline.
"""

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Attribution levels (canonical, must stay in sync with v3_integration.py)
# ---------------------------------------------------------------------------
ATTRIBUTION_LEVELS = ("CONFIRMED", "PROBABLE", "POSSIBLE", "NONE")

# Outcome semantics (Phase 3 canonical)
OUTCOMES = ("IMPROVED", "UNCHANGED", "WORSENED", "UNKNOWN")

# Evidence source names used in the outcome evidence chain
EVIDENCE_SOURCES = ("execution", "evaluation", "verification")

# ---------------------------------------------------------------------------
# Documented, deterministic thresholds
# ---------------------------------------------------------------------------
# These constants are the single source of truth for the classification.
# They are explicit and documented; there are no opaque magic numbers.
CONFIDENCE_HIGH = 0.70          # robust internal confidence (retrieval/strategy)
CONFIDENCE_MODERATE = 0.40      # meaningful but not robust internal confidence
CONFIDENCE_LOW = 0.0            # no usable internal confidence signal
DECISION_CHANGED_WEIGHT = 0.30  # weight given to "decision actually changed"
CONFIDENCE_WEIGHT = 0.70        # weight given to internal confidence signals

# Minimum completeness of the evidence chain for CONFIRMED / PROBABLE.
# "Complete" means all of execution + verification + evaluation are present.
COMPLETE_CHAIN_SOURCES = {"execution", "verification", "evaluation"}
# "Partial" means at least one evidence source is present.
PARTIAL_CHAIN_MIN = 1


class AttributionInputError(ValueError):
    """Raised when attribution inputs are invalid or synthetic."

    The engine is strict because attribution is only as good as its inputs;
    fabricated or malformed evidence must never silently produce a confident
    causal claim.
    """


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def document_thresholds() -> Dict[str, Any]:
    """Return a human- and machine-readable description of every threshold.

    Used for auditability: the semantics of every number in this module are
    defined here, so no constant is opaque.
    """
    return {
        "confident_causal_confidence": CONFIDENCE_HIGH,
        "moderate_causal_confidence": CONFIDENCE_MODERATE,
        "decision_changed_weight": DECISION_CHANGED_WEIGHT,
        "internal_confidence_weight": CONFIDENCE_WEIGHT,
        "complete_chain_sources": sorted(COMPLETE_CHAIN_SOURCES),
        "partial_chain_min": PARTIAL_CHAIN_MIN,
        "naming": (
            "CONFIRMED: learning clearly changed the decision and the outcome "
            "is measurable with a complete evidence chain and robust internal "
            "confidence. PROBABLE: learning influenced the decision with a "
            "measured outcome and partial evidence. POSSIBLE: learning and the "
            "decision correlate but causation is not established (outcome "
            "UNKNOWN, incomplete evidence, or weak confidence). NONE: no "
            "learning influence detected."
        ),
    }


def _evidence_sources(evidence: List[Dict[str, Any]]) -> set:
    """Return the set of distinct evidence source kinds present."""
    if not evidence:
        return set()
    return {e.get("source") for e in evidence if isinstance(e, dict)}


def _has_complete_chain(evidence: List[Dict[str, Any]]) -> bool:
    return COMPLETE_CHAIN_SOURCES.issubset(_evidence_sources(evidence))


def _has_partial_chain(evidence: List[Dict[str, Any]]) -> bool:
    return len(_evidence_sources(evidence)) >= PARTIAL_CHAIN_MIN


def compute_attribution_confidence(
    retrieval_confidence: Optional[float],
    strategy_confidence: Optional[float],
    *,
    decision_changed: bool,
) -> float:
    """Deterministic, explainable 0..1 confidence in the attribution.

    Formula (documented):
        internal = the best (max) of retrieval_confidence / strategy_confidence
                   that is actually present; 0 when neither is given.
        confidence = internal * CONFIDENCE_WEIGHT
                     + (1 if decision_changed else 0) * DECISION_CHANGED_WEIGHT

    This means: a decision that actually changed from baseline contributes a
    fixed 0.30 toward confidence, and the internal learning confidence
    contributes up to 0.70. The score is always in [0, 1], deterministic, and
    monotonic in both inputs. It measures "how much support there is for a
    causal attribution", not the raw score of the decision.
    """
    present = [c for c in (retrieval_confidence, strategy_confidence)
               if c is not None]
    internal = max(present) if present else 0.0
    internal = _clamp01(internal)
    changed_term = DECISION_CHANGED_WEIGHT if decision_changed else 0.0
    confidence = internal * CONFIDENCE_WEIGHT + changed_term
    return round(_clamp01(confidence), 4)


class AttributionResult:
    """Immutable outcome of running the attribution engine on one decision."""

    __slots__ = (
        "decision_id", "attribution_level", "attribution_confidence",
        "rationale", "comparative_used", "causal_evidence_insufficient",
        "input_digest",
    )

    def __init__(
        self,
        decision_id: str,
        attribution_level: str,
        attribution_confidence: float,
        rationale: List[str],
        *,
        comparative_used: bool = False,
        causal_evidence_insufficient: bool = False,
        input_digest: Dict[str, Any],
    ):
        if attribution_level not in ATTRIBUTION_LEVELS:
            raise ValueError(f"Invalid attribution_level '{attribution_level}'")
        self.decision_id = decision_id
        self.attribution_level = attribution_level
        self.attribution_confidence = _clamp01(attribution_confidence)
        self.rationale = list(rationale)
        self.comparative_used = comparative_used
        self.causal_evidence_insufficient = causal_evidence_insufficient
        self.input_digest = input_digest

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "attribution_level": self.attribution_level,
            "attribution_confidence": self.attribution_confidence,
            "rationale": self.rationale,
            "comparative_used": self.comparative_used,
            "causal_evidence_insufficient": self.causal_evidence_insufficient,
            "input_digest": self.input_digest,
        }


def _validate_inputs(kwargs: Dict[str, Any]) -> None:
    """Reject malformed or impossible inputs (garbage-in defense)."""
    influenced = kwargs["learning_influenced"]
    if not isinstance(influenced, bool):
        raise AttributionInputError(
            "learning_influenced must be a bool, got "
            f"{type(influenced).__name__}")

    outcome = kwargs["outcome"]
    if outcome not in OUTCOMES:
        raise AttributionInputError(
            f"outcome '{outcome}' not in {OUTCOMES}")

    for name in ("retrieval_confidence", "strategy_confidence"):
        val = kwargs[name]
        if val is not None and not (0.0 <= float(val) <= 1.0):
            raise AttributionInputError(
                f"{name} must be in [0,1] or None, got {val!r}")

    # A decision marked "influenced" must carry at least one provenance signal
    # pointing at what influenced it (experience_ids or strategy_ids) or an
    # internal confidence signal; otherwise the influence claim is unsupported.
    if influenced and not (kwargs["experience_ids"] or kwargs["strategy_ids"]
                           or kwargs["retrieval_confidence"] is not None
                           or kwargs["strategy_confidence"] is not None):
        # Not an error: it is *unsupported evidence of influence* and must be
        # classified conservatively. We record it in the digest for the rule.
        pass


def classify_attribution(**kwargs: Any) -> AttributionResult:
    """Deterministically classify attribution for one decision.

    Deterministic rules (evaluated in order; first matching rule wins).

    R0. If ``learning_influenced`` is False -> **NONE**.
        (No influence claim, so nothing to attribute.)

    R1. If influenced but the selected decision did NOT differ from the
        baseline -> **NONE**.
        Learning produced no observable change in the choice, so its effect on
        the outcome cannot be attributed. (A bare "learning_influenced=True"
        with an identical decision is a degenerate claim.)

    R2. Causation gate: a causal level above POSSIBLE requires the decision to
        have an outcome that was *measured* (outcome != UNKNOWN) AND non-empty
        decision-change evidence. If the outcome is UNKNOWN -> capped at
        **POSSIBLE** (correlation only), regardless of how high the internal
        confidence is.

    R3. CONFIRMED requires:
        - influenced, decision changed,
        - complete evidence chain (execution + verification + evaluation),
        - measured outcome (!= UNKNOWN),
        - internal confidence >= CONFIDENCE_HIGH (0.70).

    R4. PROBABLE requires:
        - influenced, decision changed,
        - at least partial evidence chain,
        - measured outcome (!= UNKNOWN),
        - internal confidence >= CONFIDENCE_MODERATE (0.40).

    R5. Otherwise (influenced + changed but evidence/outcome/confidence below
        the above) -> **POSSIBLE**. Causation is not established.

    A LEARNED>COLD comparative_outcome never by itself promotes a decision
    beyond the level its evidence chain supports; it is only recorded as
    supporting context (`comparative_used`). If the evidence chain is empty we
    record `causal_evidence_insufficient=True`.
    """
    learning_influenced = bool(kwargs.get("learning_influenced", False))
    outcome = kwargs.pop("outcome", "UNKNOWN")
    experience_ids = list(kwargs.get("experience_ids") or [])
    strategy_ids = list(kwargs.get("strategy_ids") or [])
    retrieval_confidence = kwargs.get("retrieval_confidence")
    strategy_confidence = kwargs.get("strategy_confidence")
    baseline_decision = kwargs.get("baseline_decision", "")
    selected_decision = kwargs.get("selected_decision", "")
    comparative_outcome = kwargs.get("comparative_outcome", "")
    evidence = list(kwargs.get("evidence") or [])
    decision_id = kwargs.get("decision_id", "?")

    _validate_inputs({
        **kwargs,
        "learning_influenced": learning_influenced,
        "outcome": outcome,
        "retrieval_confidence": retrieval_confidence,
        "strategy_confidence": strategy_confidence,
        "experience_ids": experience_ids,
        "strategy_ids": strategy_ids,
    })

    decision_changed = (baseline_decision != selected_decision) and bool(
        selected_decision)
    rationale: List[str] = []
    complete = _has_complete_chain(evidence)
    partial = _has_partial_chain(evidence)
    measured = outcome != "UNKNOWN"
    outcome_above = outcome in ("IMPROVED", "WORSENED")


    cell = {
        "learning_influenced": learning_influenced,
        "outcome": outcome,
        "experience_ids": experience_ids,
        "strategy_ids": strategy_ids,
        "retrieval_confidence": retrieval_confidence,
        "strategy_confidence": strategy_confidence,
        "decision_changed": decision_changed,
        "evidence_complete": complete,
        "evidence_partial": partial,
        "outcome_measured": measured,
        "comparative_outcome": comparative_outcome,
    }

    # R0
    if not learning_influenced:
        return AttributionResult(
            decision_id, "NONE", 0.0,
            ["learning_influenced is false: nothing to attribute."],
            comparative_used=False,
            causal_evidence_insufficient=False,
            input_digest=cell)

    has_internal_conf = (retrieval_confidence is not None
                         or strategy_confidence is not None)
    # Missing-learning-influence evidence (no experience/strategy ids AND no
    # internal confidence): we cannot attribute to any concrete artifact.
    missing_influence_evidence = not (experience_ids or strategy_ids
                                      or has_internal_conf)
    if missing_influence_evidence:
        cell["missing_influence_evidence"] = True

    conf = compute_attribution_confidence(
        retrieval_confidence, strategy_confidence,
        decision_changed=decision_changed)

    comparative_present = bool(comparative_outcome)
    causal_insufficient = False

    # R1: no observable decision change
    if not decision_changed:
        rationale.append(
            "learning_influenced is true but the selected decision equals the "
            "baseline; learning had no observable effect on the choice.")
        return AttributionResult(
            decision_id, "NONE", conf, rationale,
            comparative_used=comparative_present,
            causal_evidence_insufficient=False,
            input_digest=cell)

    # No evidence at all -> causation cannot be established.
    if not partial:
        causal_insufficient = True
        rationale.append(
            "No execution/verification/evaluation evidence present; causal "
            "evidence is insufficient. Classified POSSIBLE (correlation, not "
            "causation).")
        return AttributionResult(
            decision_id, "POSSIBLE", conf, rationale,
            comparative_used=comparative_present,
            causal_evidence_insufficient=True,
            input_digest=cell)

    # Unsupported influence claim (no ids AND no internal confidence).
    if missing_influence_evidence:
        rationale.append(
            "Decision claims learning influence but no experience_ids, "
            "strategy_ids, or confidence signal identify what influenced it. "
            "Classified POSSIBLE (influence unverified).")
        return AttributionResult(
            decision_id, "POSSIBLE", conf, rationale,
            comparative_used=comparative_present,
            causal_evidence_insufficient=True,
            input_digest=cell)

    # R2: causation gate on measured outcome
    if not measured:
        rationale.append(
            "Outcome is UNKNOWN; the effect of learning on the decision cannot "
            "be measured, so causation is not established (POSSIBLE only). "
            "UNKNOWN is not counted as either improvement or harm.")
        return AttributionResult(
            decision_id, "POSSIBLE", conf, rationale,
            comparative_used=comparative_present,
            causal_evidence_insufficient=False,
            input_digest=cell)

    internal_conf = max(
        [c for c in (retrieval_confidence, strategy_confidence)
         if c is not None] or [0.0])

    # R3: CONFIRMED
    if complete and internal_conf >= CONFIDENCE_HIGH:
        rationale.append(
            "Decision changed vs baseline with a complete evidence chain "
            "(execution+verification+evaluation) and a measured outcome; "
            f"internal confidence {internal_conf:.2f} >= {CONFIDENCE_HIGH:.2f} "
            "supports learning clearly causing the change and a measurable "
            "outcome.")
        return AttributionResult(
            decision_id, "CONFIRMED", conf, rationale,
            comparative_used=comparative_present,
            causal_evidence_insufficient=False,
            input_digest=cell)

    # R4: PROBABLE
    if partial and internal_conf >= CONFIDENCE_MODERATE:
        rationale.append(
            "Decision changed vs baseline with partial evidence chain and a "
            "measured outcome; "
            f"internal confidence {internal_conf:.2f} >= "
            f"{CONFIDENCE_MODERATE:.2f}, but the evidence chain is not "
            "complete, so other factors may have contributed (PROBABLE).")
        return AttributionResult(
            decision_id, "PROBABLE", conf, rationale,
            comparative_used=comparative_present,
            causal_evidence_insufficient=False,
            input_digest=cell)

    # R5: POSSIBLE (fall-through)
    if outcome_above:
        rationale.append(
            f"Outcome is {outcome} and the decision changed, but the evidence "
            "gate for a higher causal level was not met (incomplete chain "
            f"and/or internal confidence {internal_conf:.2f} below "
            f"{CONFIDENCE_MODERATE:.2f}); classified POSSIBLE, not causal.")
    else:
        rationale.append(
            f"Outcome {outcome} with changed decision; evidence gate for a "
            "higher causal level not met; classified POSSIBLE.")
    return AttributionResult(
        decision_id, "POSSIBLE", conf, rationale,
        comparative_used=comparative_present,
        causal_evidence_insufficient=False,
        input_digest=cell)


class AttributionMetrics:
    """Aggregates attribution results with explicit, documented semantics.

    Denominator semantics (documented):
        - ``learning_influenced_decisions``: every decision attributed to
          learning (attribution in CONFIRMED/PROBABLE/POSSIBLE, i.e. NONE is
          excluded).
        - ``improved/unchanged/worsened/unknown_decisions``: OUTCOME counts
          over the learning-influenced set. UNKNOWN is counted ONLY against the
          explicitly defined ``unknown_decisions`` counter; it is never placed
          in the improvement or harm numerator/denominator.
        - ``decision_improvement_rate`` = improved / learning_influenced
          (denominator is the influenced set, documented; UNKNOWN decisions
          remain in the denominator but are NOT assumed to be failures).
        - ``learning_harm_rate`` = worsened / learning_influenced.
        - Attribution distribution counted per level.
    """

    def __init__(self, results: List[AttributionResult],
                 outcomes: Optional[Dict[str, str]] = None):
        self.results = list(results)
        # outcomes maps decision_id -> OUTCOME (from persisted evidence).
        self.outcomes = dict(outcomes or {})

    def to_dict(self) -> Dict[str, Any]:
        influenced = [r for r in self.results
                      if r.attribution_level != "NONE"]
        n = len(influenced)
        improved = unchanged = worsened = unknown = 0
        for r in influenced:
            o = self.outcomes.get(r.decision_id, "UNKNOWN")
            if o == "IMPROVED":
                improved += 1
            elif o == "UNCHANGED":
                unchanged += 1
            elif o == "WORSENED":
                worsened += 1
            else:
                unknown += 1

        by_attribution = {lvl: 0 for lvl in ATTRIBUTION_LEVELS}
        for r in self.results:
            by_attribution[r.attribution_level] += 1

        improvement_rate = (round(improved / n, 4) if n else 0.0)
        harm_rate = (round(worsened / n, 4) if n else 0.0)

        return {
            "learning_influenced_decisions": n,
            "improved_decisions": improved,
            "unchanged_decisions": unchanged,
            "worsened_decisions": worsened,
            "unknown_decisions": unknown,
            "decision_improvement_rate": improvement_rate,
            "learning_harm_rate": harm_rate,
            "attribution_confirmed": by_attribution["CONFIRMED"],
            "attribution_probable": by_attribution["PROBABLE"],
            "attribution_possible": by_attribution["POSSIBLE"],
            "attribution_none": by_attribution["NONE"],
            "attribution_confirmed+probable+possible": improved + unchanged
                + worsened + unknown,
            "denominator_semantics": (
                "improvement_rate and harm_rate denominators are the "
                "learning-influenced decision count. UNKNOWN outcomes are "
                "excluded from the numerator (not improvement, not harm) but "
                "remain in the denominator without being assumed to be "
                "failures."),
        }
