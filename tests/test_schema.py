# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

import pytest
from pydantic import ValidationError

from coreason_inference.schema import (
    CausalGraph,
    CausalNode,
    InterventionResult,
    LoopDynamics,
    LoopType,
    RefutationStatus,
)


def test_loop_type_enum() -> None:
    assert LoopType.POSITIVE_FEEDBACK.value == "POSITIVE"
    assert LoopType.NEGATIVE_FEEDBACK.value == "NEGATIVE"
    assert LoopType.NONE.value == "ACYCLIC"


def test_valid_causal_graph() -> None:
    node_a = CausalNode(id="A", codex_concept_id=1)
    node_b = CausalNode(id="B", codex_concept_id=2)

    graph = CausalGraph(
        nodes=[node_a, node_b],
        edges=[("A", "B")],
        loop_dynamics=[],
        stability_score=0.9,
    )
    assert len(graph.nodes) == 2
    assert graph.edges == [("A", "B")]


def test_causal_graph_invalid_edge_source() -> None:
    node_b = CausalNode(id="B", codex_concept_id=2)
    with pytest.raises(ValidationError) as exc:
        CausalGraph(
            nodes=[node_b],
            edges=[("A", "B")],
            stability_score=0.5,  # A missing
        )
    assert "Edge source 'A' not found in nodes" in str(exc.value)


def test_causal_graph_invalid_edge_target() -> None:
    node_a = CausalNode(id="A", codex_concept_id=1)
    with pytest.raises(ValidationError) as exc:
        CausalGraph(
            nodes=[node_a],
            edges=[("A", "B")],
            stability_score=0.5,  # B missing
        )
    assert "Edge target 'B' not found in nodes" in str(exc.value)


def test_causal_graph_duplicate_nodes() -> None:
    node_a1 = CausalNode(id="A", codex_concept_id=1)
    node_a2 = CausalNode(id="A", codex_concept_id=2)
    with pytest.raises(ValidationError) as exc:
        CausalGraph(nodes=[node_a1, node_a2], edges=[], stability_score=0.5)
    assert "Node IDs must be unique" in str(exc.value)


def test_causal_graph_loop_dynamics() -> None:
    node_a = CausalNode(id="A", codex_concept_id=1)
    node_b = CausalNode(id="B", codex_concept_id=2)
    loop = LoopDynamics(path=["A", "B", "A"], type=LoopType.NEGATIVE_FEEDBACK)

    graph = CausalGraph(
        nodes=[node_a, node_b],
        edges=[("A", "B"), ("B", "A")],
        loop_dynamics=[loop],
        stability_score=0.8,
    )
    assert len(graph.loop_dynamics) == 1
    assert graph.loop_dynamics[0].type == LoopType.NEGATIVE_FEEDBACK


def test_causal_graph_invalid_loop_node() -> None:
    node_a = CausalNode(id="A", codex_concept_id=1)
    loop = LoopDynamics(path=["A", "B"], type=LoopType.NEGATIVE_FEEDBACK)  # B missing

    with pytest.raises(ValidationError) as exc:
        CausalGraph(nodes=[node_a], edges=[], loop_dynamics=[loop], stability_score=0.5)
    assert "Loop node 'B' not found in nodes" in str(exc.value)


def test_intervention_result() -> None:
    res = InterventionResult(
        patient_id="123",
        intervention="do(X=1)",
        counterfactual_outcome=0.5,
        confidence_interval=(0.4, 0.6),
        refutation_status=RefutationStatus.PASSED,
    )
    assert res.refutation_status == RefutationStatus.PASSED
    assert res.patient_id == "123"


def test_empty_graph() -> None:
    graph = CausalGraph(nodes=[], edges=[], loop_dynamics=[], stability_score=1.0)
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0


def test_self_loop_graph() -> None:
    node_a = CausalNode(id="A", codex_concept_id=1)
    graph = CausalGraph(
        nodes=[node_a],
        edges=[("A", "A")],
        stability_score=0.5,
    )
    assert ("A", "A") in graph.edges


def test_complex_cyclic_graph() -> None:
    # A -> B -> C -> A (Negative Loop)
    # Z (Latent) -> B
    # Z (Latent) -> C

    node_a = CausalNode(id="A", codex_concept_id=101)
    node_b = CausalNode(id="B", codex_concept_id=102)
    node_c = CausalNode(id="C", codex_concept_id=103)
    node_z = CausalNode(id="Z", codex_concept_id=999, is_latent=True)

    loop = LoopDynamics(path=["A", "B", "C", "A"], type=LoopType.NEGATIVE_FEEDBACK)

    graph = CausalGraph(
        nodes=[node_a, node_b, node_c, node_z],
        edges=[("A", "B"), ("B", "C"), ("C", "A"), ("Z", "B"), ("Z", "C")],
        loop_dynamics=[loop],
        stability_score=0.75,
    )

    assert len(graph.nodes) == 4
    assert len(graph.edges) == 5
    assert len(graph.loop_dynamics) == 1
    assert graph.nodes[3].is_latent is True


def test_type_validation_error() -> None:
    # codex_concept_id expects int, pass invalid string
    with pytest.raises(ValidationError):
        CausalNode(id="A", codex_concept_id="not-an-int")


def test_type_coercion() -> None:
    # codex_concept_id expects int, pass valid string "123"
    node = CausalNode(id="A", codex_concept_id="123")
    assert node.codex_concept_id == 123
    assert isinstance(node.codex_concept_id, int)


def test_loop_dynamics_subscriptable() -> None:
    loop = LoopDynamics(path=["A", "B"], type=LoopType.NEGATIVE_FEEDBACK)
    assert loop["path"] == ["A", "B"]
    assert loop["type"] == LoopType.NEGATIVE_FEEDBACK
