# Tests for CapabilityGraph
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from harness.capabilities.graph import CapabilityGraph, CapabilityNode, CapabilityGraphError


def test_add_node():
    graph = CapabilityGraph()
    node = CapabilityNode(id="node1", capability="testing", dependencies=[])
    graph.add_node(node)
    assert graph.get_node_count() == 1
    assert graph.get_node("node1") is node


def test_duplicate_node_raises():
    graph = CapabilityGraph()
    graph.add_node(CapabilityNode(id="node1", capability="testing"))
    try:
        graph.add_node(CapabilityNode(id="node1", capability="other"))
        assert False, "Should have raised"
    except CapabilityGraphError:
        pass


def test_has_cycle_no_cycle():
    graph = CapabilityGraph()
    graph.add_node(CapabilityNode(id="a", capability="arch", dependencies=[]))
    graph.add_node(CapabilityNode(id="b", capability="impl", dependencies=["a"]))
    graph.add_node(CapabilityNode(id="c", capability="test", dependencies=["b"]))
    assert graph.has_cycle() is False


def test_has_cycle_detected():
    graph = CapabilityGraph()
    graph.add_node(CapabilityNode(id="a", capability="arch", dependencies=["c"]))
    graph.add_node(CapabilityNode(id="b", capability="impl", dependencies=["a"]))
    graph.add_node(CapabilityNode(id="c", capability="test", dependencies=["b"]))
    assert graph.has_cycle() is True


def test_topological_sort():
    graph = CapabilityGraph()
    graph.add_node(CapabilityNode(id="a", capability="arch", dependencies=[]))
    graph.add_node(CapabilityNode(id="b", capability="impl", dependencies=["a"]))
    graph.add_node(CapabilityNode(id="c", capability="test", dependencies=["b"]))
    order = graph.topological_sort()
    ids = [n.id for n in order]
    assert ids == ["a", "b", "c"]


def test_topological_sort_with_cycle_raises():
    graph = CapabilityGraph()
    graph.add_node(CapabilityNode(id="a", capability="arch", dependencies=["c"]))
    graph.add_node(CapabilityNode(id="b", capability="impl", dependencies=["a"]))
    graph.add_node(CapabilityNode(id="c", capability="test", dependencies=["b"]))
    try:
        graph.topological_sort()
        assert False, "Should have raised"
    except CapabilityGraphError:
        pass


def test_get_ready_nodes():
    graph = CapabilityGraph()
    graph.add_node(CapabilityNode(id="a", capability="arch", dependencies=[]))
    graph.add_node(CapabilityNode(id="b", capability="impl", dependencies=["a"]))
    graph.add_node(CapabilityNode(id="c", capability="test", dependencies=["b"]))

    ready = graph.get_ready_nodes()
    assert len(ready) == 1
    assert ready[0].id == "a"

    graph.mark_completed("a")
    ready = graph.get_ready_nodes()
    assert len(ready) == 1
    assert ready[0].id == "b"

    graph.mark_completed("b")
    ready = graph.get_ready_nodes()
    assert len(ready) == 1
    assert ready[0].id == "c"


def test_mark_completed():
    graph = CapabilityGraph()
    node = CapabilityNode(id="n1", capability="t")
    graph.add_node(node)
    graph.mark_completed("n1", {"result": "ok"})
    assert graph.get_node("n1").status == "completed"
    assert graph.get_node("n1").result == {"result": "ok"}


def test_mark_failed():
    graph = CapabilityGraph()
    graph.add_node(CapabilityNode(id="n1", capability="t"))
    graph.mark_failed("n1")
    assert graph.get_node("n1").status == "failed"


def test_mark_skipped():
    graph = CapabilityGraph()
    graph.add_node(CapabilityNode(id="n1", capability="t"))
    graph.mark_skipped("n1")
    assert graph.get_node("n1").status == "skipped"


def test_mark_nonexistent_raises():
    graph = CapabilityGraph()
    try:
        graph.mark_completed("nonexistent")
        assert False, "Should have raised"
    except CapabilityGraphError:
        pass


def test_serialization_roundtrip():
    graph = CapabilityGraph()
    graph.add_node(CapabilityNode(id="a", capability="arch", dependencies=[]))
    graph.add_node(CapabilityNode(id="b", capability="impl", dependencies=["a"]))
    graph.mark_completed("a", {"done": True})

    data = graph.to_dict()
    restored = CapabilityGraph.from_dict(data)

    assert restored.get_node_count() == 2
    assert restored.get_node("a").status == "completed"
    assert restored.get_node("a").result == {"done": True}
    assert restored.get_node("b").dependencies == ["a"]


def test_get_all_nodes():
    graph = CapabilityGraph()
    graph.add_node(CapabilityNode(id="a", capability="x"))
    graph.add_node(CapabilityNode(id="b", capability="y"))
    nodes = graph.get_all_nodes()
    assert len(nodes) == 2
    assert {n.id for n in nodes} == {"a", "b"}


if __name__ == "__main__":
    test_add_node()
    test_duplicate_node_raises()
    test_has_cycle_no_cycle()
    test_has_cycle_detected()
    test_topological_sort()
    test_topological_sort_with_cycle_raises()
    test_get_ready_nodes()
    test_mark_completed()
    test_mark_failed()
    test_mark_skipped()
    test_mark_nonexistent_raises()
    test_serialization_roundtrip()
    test_get_all_nodes()
    print("All capability_graph tests passed!")
