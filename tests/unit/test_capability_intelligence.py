"""Tests for Capability Intelligence (Phase G)."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.capabilities.intelligence import (
    SynonymResolver, CapabilityGapDetector, CapabilityGap,
    CapabilityRecommender, CapabilityQualityAnalyzer,
    CapabilityQualityMetrics,
)
from harness.capabilities.registry import CapabilityRegistry
from harness.capabilities.taxonomy import CapabilityTaxonomy


def setup_registry_and_taxonomy():
    reg = CapabilityRegistry()
    reg.register_agent("coder", ["backend_development", "frontend_development"])
    reg.register_agent("tester", ["testing", "unit_testing"])
    reg.register_agent("reviewer", ["code_review", "security_analysis"])

    taxo = CapabilityTaxonomy(taxonomy_path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config", "capability_taxonomy.yaml"
    ))
    return reg, taxo


def test_synonym_resolver():
    assert SynonymResolver.canonical("backend") == "backend_development"
    assert SynonymResolver.canonical("backend_development") == "backend_development"
    assert SynonymResolver.canonical("frontend") == "frontend_development"

    syns = SynonymResolver.resolve("backend")
    assert "backend_development" in syns

    assert SynonymResolver.is_synonym("backend", "backend_development") is True
    assert SynonymResolver.is_synonym("backend", "frontend_development") is False


def test_gap_detector_missing_cap():
    reg, taxo = setup_registry_and_taxonomy()
    detector = CapabilityGapDetector(reg, taxo)

    gaps = detector.detect_gaps(["backend_development", "machine_learning"])
    # backend_development is covered
    backend_gaps = [g for g in gaps if g.capability == "backend_development"]
    assert len(backend_gaps) == 0

    # machine_learning is not covered
    ml_gaps = [g for g in gaps if g.capability == "machine_learning"]
    assert len(ml_gaps) >= 1
    assert ml_gaps[0].gap_type == "missing"


def test_coverage_report():
    reg, taxo = setup_registry_and_taxonomy()
    detector = CapabilityGapDetector(reg, taxo)
    report = detector.get_coverage_report()

    assert report["agents"] == 3
    assert report["covered"] > 0
    assert report["coverage_pct"] > 0


def test_capability_recommender():
    reg, taxo = setup_registry_and_taxonomy()
    recommender = CapabilityRecommender(reg, taxo)

    recs = recommender.recommend(
        "Implement a new backend API endpoint",
        ["Tests must pass", "Security review required"],
    )
    assert len(recs) >= 1
    # Should recommend something backend or API related
    cap_names = [r.capability for r in recs]
    assert any("backend" in c or "api" in c for c in cap_names)


def test_quality_analyzer():
    reg, taxo = setup_registry_and_taxonomy()
    analyzer = CapabilityQualityAnalyzer(reg)
    metrics = analyzer.analyze("backend_development")

    assert isinstance(metrics, CapabilityQualityMetrics)
    assert metrics.capability == "backend_development"
    assert metrics.agent_count >= 1
    assert 0 <= metrics.quality_score <= 1

    # Unknown capability
    unknown = analyzer.analyze("nonexistent_cap")
    assert unknown.agent_count == 0
