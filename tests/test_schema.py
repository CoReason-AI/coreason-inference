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
