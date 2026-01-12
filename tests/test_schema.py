# Copyright (C) 2026 CoReason
# Licensed under the Prosperity Public License 3.0.0

import pytest
from pydantic import ValidationError

from coreason_inference.schema import (
    CausalGraph,
    CausalNode,
    InterventionResult,
    LoopType,
)


def test_loop_type_enum() -> None:
    assert LoopType.POSITIVE_FEEDBACK.value == "POSITIVE"
    assert LoopType.NEGATIVE_FEEDBACK.value == "NEGATIVE"
    assert LoopType.NONE.value == "ACYCLIC"


def test_causal_node_creation() -> None:
    node = CausalNode(id="node_1", codex_concept_id=123, is_latent=False)
    assert node.id == "node_1"
    assert node.codex_concept_id == 123
    assert not node.is_latent


def test_causal_graph_creation_valid() -> None:
    node1 = CausalNode(id="A", codex_concept_id=1, is_latent=False)
    node2 = CausalNode(id="B", codex_concept_id=2, is_latent=False)
    graph = CausalGraph(
        nodes=[node1, node2],
        edges=[("A", "B")],
        loop_dynamics=[],
        stability_score=0.9,
    )
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.edges[0] == ("A", "B")


def test_causal_graph_invalid_edge_source() -> None:
    node1 = CausalNode(id="A", codex_concept_id=1, is_latent=False)
    with pytest.raises(ValidationError) as excinfo:
        CausalGraph(
            nodes=[node1],
            edges=[("B", "A")],  # B does not exist
            loop_dynamics=[],
            stability_score=0.5,
        )
    assert "Edge source node 'B' not found" in str(excinfo.value)


def test_causal_graph_invalid_edge_target() -> None:
    node1 = CausalNode(id="A", codex_concept_id=1, is_latent=False)
    with pytest.raises(ValidationError) as excinfo:
        CausalGraph(
            nodes=[node1],
            edges=[("A", "B")],  # B does not exist
            loop_dynamics=[],
            stability_score=0.5,
        )
    assert "Edge target node 'B' not found" in str(excinfo.value)


def test_intervention_result_creation() -> None:
    result = InterventionResult(
        patient_id="patient_001",
        intervention="do(X=1)",
        counterfactual_outcome=0.8,
        confidence_interval=(0.7, 0.9),
        refutation_status="PASSED",
    )
    assert result.patient_id == "patient_001"
    assert result.counterfactual_outcome == 0.8
    assert result.confidence_interval == (0.7, 0.9)
    assert result.refutation_status == "PASSED"


# --- New Tests for Edge Cases and Complex Scenarios ---


def test_duplicate_node_ids() -> None:
    node1 = CausalNode(id="A", codex_concept_id=1, is_latent=False)
    node2 = CausalNode(id="A", codex_concept_id=2, is_latent=False)  # Duplicate ID
    with pytest.raises(ValidationError) as excinfo:
        CausalGraph(
            nodes=[node1, node2],
            edges=[],
            loop_dynamics=[],
            stability_score=0.5,
        )
    assert "Duplicate node ID found: 'A'" in str(excinfo.value)


def test_self_loop() -> None:
    node = CausalNode(id="A", codex_concept_id=1, is_latent=False)
    graph = CausalGraph(
        nodes=[node],
        edges=[("A", "A")],
        loop_dynamics=[{"path": ["A", "A"], "type": "POSITIVE"}],
        stability_score=0.1,
    )
    assert graph.edges == [("A", "A")]
    assert len(graph.loop_dynamics) == 1


def test_complex_cyclic_graph() -> None:
    # A -> B -> C -> A
    nodes = [
        CausalNode(id="A", codex_concept_id=1, is_latent=False),
        CausalNode(id="B", codex_concept_id=2, is_latent=False),
        CausalNode(id="C", codex_concept_id=3, is_latent=False),
    ]
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    loop_dynamics = [{"path": ["A", "B", "C", "A"], "type": LoopType.NEGATIVE_FEEDBACK.value}]

    graph = CausalGraph(
        nodes=nodes,
        edges=edges,
        loop_dynamics=loop_dynamics,
        stability_score=0.8,
    )
    assert len(graph.edges) == 3
    assert graph.loop_dynamics[0]["path"] == ["A", "B", "C", "A"]


def test_loop_dynamics_invalid_edge() -> None:
    nodes = [
        CausalNode(id="A", codex_concept_id=1, is_latent=False),
        CausalNode(id="B", codex_concept_id=2, is_latent=False),
    ]
    edges = [("A", "B")]
    # Path implies B->A exists, but it doesn't
    loop_dynamics = [{"path": ["A", "B", "A"], "type": "NEGATIVE"}]

    with pytest.raises(ValidationError) as excinfo:
        CausalGraph(
            nodes=nodes,
            edges=edges,
            loop_dynamics=loop_dynamics,
            stability_score=0.5,
        )
    assert "Loop path edge ('B', 'A') does not exist" in str(excinfo.value)


def test_loop_dynamics_invalid_node_in_path() -> None:
    nodes = [
        CausalNode(id="A", codex_concept_id=1, is_latent=False),
    ]
    edges: list[tuple[str, str]] = []
    # Node B does not exist
    loop_dynamics = [{"path": ["A", "B", "A"], "type": "NEGATIVE"}]

    with pytest.raises(ValidationError) as excinfo:
        CausalGraph(
            nodes=nodes,
            edges=edges,
            loop_dynamics=loop_dynamics,
            stability_score=0.5,
        )
    # This might fail on "Loop path node 'B' not found" OR "Loop path edge ('A', 'B') does not exist"
    # The order of checks matters. In my implementation, edge check comes first.
    # ('A', 'B') is not in edges, so it should fail there.
    assert "Loop path edge ('A', 'B') does not exist" in str(excinfo.value)


def test_loop_dynamics_malformed_path() -> None:
    nodes = [CausalNode(id="A", codex_concept_id=1, is_latent=False)]
    with pytest.raises(ValidationError) as excinfo:
        CausalGraph(
            nodes=nodes,
            edges=[],
            loop_dynamics=[{"path": "invalid_path_string", "type": "NEGATIVE"}],
            stability_score=0.5,
        )
    assert "Loop dynamics must contain a 'path' list" in str(excinfo.value)


def test_empty_graph() -> None:
    graph = CausalGraph(nodes=[], edges=[], loop_dynamics=[], stability_score=1.0)
    assert len(graph.nodes) == 0
