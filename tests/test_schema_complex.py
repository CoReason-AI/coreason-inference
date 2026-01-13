# Copyright (C) 2026 CoReason
# Licensed under the Prosperity Public License 3.0.0

import math

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


def test_disconnected_graph() -> None:
    """
    Verify that a graph with two completely disconnected components is valid.
    Component 1: A -> B
    Component 2: C -> D
    """
    nodes = [
        CausalNode(id="A", codex_concept_id=1, is_latent=False),
        CausalNode(id="B", codex_concept_id=2, is_latent=False),
        CausalNode(id="C", codex_concept_id=3, is_latent=False),
        CausalNode(id="D", codex_concept_id=4, is_latent=False),
    ]
    edges = [("A", "B"), ("C", "D")]

    graph = CausalGraph(
        nodes=nodes,
        edges=edges,
        loop_dynamics=[],
        stability_score=1.0,
    )

    assert len(graph.nodes) == 4
    assert len(graph.edges) == 2


def test_orphan_nodes() -> None:
    """
    Verify that a graph can contain orphan nodes (nodes with no edges).
    """
    nodes = [
        CausalNode(id="A", codex_concept_id=1, is_latent=False),
        CausalNode(id="Orphan", codex_concept_id=99, is_latent=False),
    ]
    edges = [("A", "A")]
    loops = [LoopDynamics(path=["A", "A"], type=LoopType.NEGATIVE_FEEDBACK)]

    graph = CausalGraph(
        nodes=nodes,
        edges=edges,
        loop_dynamics=loops,
        stability_score=1.0,
    )

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.nodes[1].id == "Orphan"


def test_figure_eight_topology() -> None:
    """
    Verify a complex topology where two loops share a central hub node.
    Loop 1: Hub -> Left -> Hub (Negative)
    Loop 2: Hub -> Right -> Hub (Positive)
    """
    nodes = [
        CausalNode(id="Hub", codex_concept_id=1, is_latent=False),
        CausalNode(id="Left", codex_concept_id=2, is_latent=False),
        CausalNode(id="Right", codex_concept_id=3, is_latent=False),
    ]
    edges = [
        ("Hub", "Left"),
        ("Left", "Hub"),
        ("Hub", "Right"),
        ("Right", "Hub"),
    ]

    loop1 = LoopDynamics(path=["Hub", "Left", "Hub"], type=LoopType.NEGATIVE_FEEDBACK)
    loop2 = LoopDynamics(path=["Hub", "Right", "Hub"], type=LoopType.POSITIVE_FEEDBACK)

    graph = CausalGraph(
        nodes=nodes,
        edges=edges,
        loop_dynamics=[loop1, loop2],
        stability_score=0.5,
    )

    assert len(graph.loop_dynamics) == 2
    types = {loop.type for loop in graph.loop_dynamics}
    assert LoopType.NEGATIVE_FEEDBACK in types
    assert LoopType.POSITIVE_FEEDBACK in types


def test_mixed_feedback_types() -> None:
    """
    Verify a graph containing both Positive and Negative feedback loops.
    """
    nodes = [
        CausalNode(id="A", codex_concept_id=1, is_latent=False),
        CausalNode(id="B", codex_concept_id=2, is_latent=False),
    ]
    # A is negative self-loop, B is positive self-loop
    edges = [("A", "A"), ("B", "B")]
    loops = [
        LoopDynamics(path=["A", "A"], type=LoopType.NEGATIVE_FEEDBACK),
        LoopDynamics(path=["B", "B"], type=LoopType.POSITIVE_FEEDBACK),
    ]

    graph = CausalGraph(
        nodes=nodes,
        edges=edges,
        loop_dynamics=loops,
        stability_score=0.0,
    )

    assert len(graph.loop_dynamics) == 2


def test_loop_path_directionality_strictness() -> None:
    """
    Verify that loop validation strictly respects edge direction.
    If Edge A->B exists, a loop path ["B", "A"] should FAIL
    unless Edge B->A also exists.
    """
    nodes = [
        CausalNode(id="A", codex_concept_id=1, is_latent=False),
        CausalNode(id="B", codex_concept_id=2, is_latent=False),
    ]
    edges = [("A", "B")]  # Only A->B exists

    # Try to define a loop path B->A->B (Implies B->A exists)
    loop = LoopDynamics(path=["B", "A", "B"], type=LoopType.NEGATIVE_FEEDBACK)

    with pytest.raises(ValidationError) as excinfo:
        CausalGraph(
            nodes=nodes,
            edges=edges,
            loop_dynamics=[loop],
            stability_score=0.0,
        )

    # Error should catch the missing B->A edge
    assert "Loop path edge ('B', 'A') does not exist" in str(excinfo.value)


def test_intervention_result_extreme_values() -> None:
    """
    Test how InterventionResult handles Infinity and NaN values for outcomes.
    Pydantic generally allows float('inf') and float('nan').
    """
    result = InterventionResult(
        patient_id="test_pat",
        intervention="do(X)",
        counterfactual_outcome=float("inf"),
        confidence_interval=(float("-inf"), float("nan")),
        refutation_status=RefutationStatus.PASSED,
    )

    assert result.counterfactual_outcome == float("inf")
    assert result.confidence_interval[0] == float("-inf")
    assert math.isnan(result.confidence_interval[1])
