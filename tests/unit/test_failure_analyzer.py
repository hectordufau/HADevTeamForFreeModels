# Tests for FailureAnalyzer
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.planner.failure_analyzer import FailureAnalyzer, FailureReport, FailureAnalyzerError


def test_classify_implementation_error():
    analyzer = FailureAnalyzer()
    category = analyzer.classify("TypeError: unsupported operand type(s)")
    assert category == "implementation"


def test_classify_test_error():
    analyzer = FailureAnalyzer()
    category = analyzer.classify("AssertionError: Expected 200, got 404")
    assert category == "test"


def test_classify_architecture_error():
    analyzer = FailureAnalyzer()
    category = analyzer.classify("Circular dependency detected between modules")
    assert category == "architecture"


def test_classify_security_error():
    analyzer = FailureAnalyzer()
    category = analyzer.classify("SQL injection vulnerability found in query")
    assert category == "security"


def test_classify_integration_error():
    analyzer = FailureAnalyzer()
    category = analyzer.classify("Connection refused: API at localhost:8080")
    assert category == "integration"


def test_classify_configuration_error():
    analyzer = FailureAnalyzer()
    category = analyzer.classify("Missing env: DATABASE_URL not set")
    assert category == "configuration"


def test_classify_environment_error():
    analyzer = FailureAnalyzer()
    category = analyzer.classify("Dependency missing: package 'requests' not found")
    assert category == "environment"


def test_classify_unknown_error():
    analyzer = FailureAnalyzer()
    category = analyzer.classify("Something completely unexpected happened")
    assert category == "unknown"


def test_analyze_produces_report():
    analyzer = FailureAnalyzer()
    node = type('obj', (object,), {"id": "test-node", "capability": "testing"})
    result = type('obj', (object,), {"status": "failed", "error": "AssertionError: test failed"})

    report = analyzer.analyze(node, result)
    assert isinstance(report, FailureReport)
    assert report.node_id == "test-node"
    assert report.capability == "testing"
    assert report.category == "test"


def test_analyze_with_evidence():
    analyzer = FailureAnalyzer()
    node = type('obj', (object,), {"id": "sec-node", "capability": "security_analysis"})
    result = type('obj', (object,), {"status": "failed", "error": "Access denied"})
    evidence = {"logs": ["Permission denied for user"]}

    report = analyzer.analyze(node, result, evidence)
    assert report.category == "security"


def test_recommendations_generated():
    analyzer = FailureAnalyzer()
    node = type('obj', (object,), {"id": "n1", "capability": "testing"})
    result = type('obj', (object,), {"status": "failed", "error": "test failure"})

    report = analyzer.analyze(node, result)
    assert len(report.recommendations) > 0
    assert len(report.missing_capabilities) > 0


def test_dict_node_and_result():
    analyzer = FailureAnalyzer()
    node = {"id": "impl1", "capability": "backend_development"}
    result = {"status": "failed", "error": "TypeError: cannot concat str to int"}

    report = analyzer.analyze(node, result)
    assert report.node_id == "impl1"
    assert report.capability == "backend_development"
    assert report.category == "implementation"


def test_no_error_classified_as_unknown():
    analyzer = FailureAnalyzer()
    node = type('obj', (object,), {"id": "n1", "capability": "testing"})
    result = type('obj', (object,), {"status": "failed", "error": ""})

    report = analyzer.analyze(node, result)
    assert report.category == "unknown"


def test_evidence_text_in_report():
    analyzer = FailureAnalyzer()
    node = type('obj', (object,), {"id": "n1", "capability": "testing"})
    result = type('obj', (object,), {"status": "failed", "error": "failure"})
    evidence = type('obj', (object,), {"to_dict": lambda: {"file": "test.py", "line": 42}})

    report = analyzer.analyze(node, result, evidence)
    assert report.evidence_summary is not None


if __name__ == "__main__":
    test_classify_implementation_error()
    test_classify_test_error()
    test_classify_architecture_error()
    test_classify_security_error()
    test_classify_integration_error()
    test_classify_configuration_error()
    test_classify_environment_error()
    test_classify_unknown_error()
    test_analyze_produces_report()
    test_analyze_with_evidence()
    test_recommendations_generated()
    test_dict_node_and_result()
    test_no_error_classified_as_unknown()
    test_evidence_text_in_report()
    print("All failure_analyzer tests passed!")
