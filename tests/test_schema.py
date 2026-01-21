# Copyright (C) 2026 CoReason
# Licensed under the Prosperity Public License 3.0.0

import pytest
from coreason_inference.schema import (
    CausalGraph,
    CausalNode,
    InterventionResult,
    LoopDynamics,
    LoopType,
    RefutationStatus,
)
from pydantic import ValidationError


def test_loop_type_enum() -> None:
    assert LoopType.POSITIVE_FEEDBACK.value == "POSITIVE"
    assert LoopType.NEGATIVE_FEEDBACK.value == "NEGATIVE"
    assert LoopType.NONE.value == "ACYCLIC"


def test_refutation_status_enum() -> None:
    assert RefutationStatus.PASSED.value == "PASSED"
    assert RefutationStatus.FAILED.value == "FAILED"


def test_causal_node_creation() -> None:
    node = CausalNode(id="node_1", codex_concept_id=123, is_latent=False)
    assert node.id == "node_1"
    assert node.codex_concept_id == 123
    assert not node.is_latent


def test_loop_dynamics_creation() -> None:
    loop = LoopDynamics(path=["A", "B", "A"], type=LoopType.NEGATIVE_FEEDBACK)
    assert loop.path == ["A", "B", "A"]
    assert loop.type == LoopType.NEGATIVE_FEEDBACK


def test_loop_dynamics_invalid_path_length() -> None:
    with pytest.raises(ValidationError) as excinfo:
        LoopDynamics(path=["A"], type=LoopType.NEGATIVE_FEEDBACK)
    assert "List should have at least 2 items" in str(excinfo.value)


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
        refutation_status=RefutationStatus.PASSED,
    )
    assert result.patient_id == "patient_001"
    assert result.counterfactual_outcome == 0.8
    assert result.confidence_interval == (0.7, 0.9)
    assert result.refutation_status == RefutationStatus.PASSED


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
    loop = LoopDynamics(path=["A", "A"], type=LoopType.POSITIVE_FEEDBACK)
    graph = CausalGraph(
        nodes=[node],
        edges=[("A", "A")],
        loop_dynamics=[loop],
        stability_score=0.1,
    )
    assert graph.edges == [("A", "A")]
    assert len(graph.loop_dynamics) == 1
    assert graph.loop_dynamics[0] == loop


def test_complex_cyclic_graph() -> None:
    # A -> B -> C -> A
    nodes = [
        CausalNode(id="A", codex_concept_id=1, is_latent=False),
        CausalNode(id="B", codex_concept_id=2, is_latent=False),
        CausalNode(id="C", codex_concept_id=3, is_latent=False),
    ]
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    loop = LoopDynamics(path=["A", "B", "C", "A"], type=LoopType.NEGATIVE_FEEDBACK)

    graph = CausalGraph(
        nodes=nodes,
        edges=edges,
        loop_dynamics=[loop],
        stability_score=0.8,
    )
    assert len(graph.edges) == 3
    assert graph.loop_dynamics[0].path == ["A", "B", "C", "A"]


def test_loop_dynamics_invalid_edge() -> None:
    nodes = [
        CausalNode(id="A", codex_concept_id=1, is_latent=False),
        CausalNode(id="B", codex_concept_id=2, is_latent=False),
    ]
    edges = [("A", "B")]
    # Path implies B->A exists, but it doesn't
    loop = LoopDynamics(path=["A", "B", "A"], type=LoopType.NEGATIVE_FEEDBACK)

    with pytest.raises(ValidationError) as excinfo:
        CausalGraph(
            nodes=nodes,
            edges=edges,
            loop_dynamics=[loop],
            stability_score=0.5,
        )
    assert "Loop path edge ('B', 'A') does not exist" in str(excinfo.value)


def test_loop_dynamics_invalid_node_in_path() -> None:
    nodes = [
        CausalNode(id="A", codex_concept_id=1, is_latent=False),
    ]
    edges: list[tuple[str, str]] = []
    # Node B does not exist
    # LoopDynamics will succeed in creation because it doesn't know about the graph nodes
    # Validation happens in CausalGraph
    loop = LoopDynamics(path=["A", "B", "A"], type=LoopType.NEGATIVE_FEEDBACK)

    with pytest.raises(ValidationError) as excinfo:
        CausalGraph(
            nodes=nodes,
            edges=edges,
            loop_dynamics=[loop],
            stability_score=0.5,
        )
    # The error should be about the edge ('A', 'B') not existing
    assert "Loop path edge ('A', 'B') does not exist" in str(excinfo.value)


def test_empty_graph() -> None:
    graph = CausalGraph(nodes=[], edges=[], loop_dynamics=[], stability_score=1.0)
    assert len(graph.nodes) == 0
