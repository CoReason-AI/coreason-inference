# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import Any

import pytest
from pydantic import ValidationError

from coreason_inference.schema import CausalGraph, CausalNode, InterventionResult, LoopType


def test_loop_type_enum() -> None:
    assert LoopType.POSITIVE_FEEDBACK.value == "POSITIVE"
    assert LoopType.NEGATIVE_FEEDBACK.value == "NEGATIVE"
    assert LoopType.NONE.value == "ACYCLIC"


def test_causal_node_creation() -> None:
    node = CausalNode(id="gene_a", codex_concept_id=101, is_latent=False)
    assert node.id == "gene_a"
    assert node.codex_concept_id == 101
    assert node.is_latent is False


def test_causal_node_validation_error() -> None:
    # Use Any to bypass mypy static checking so we can test runtime validation
    invalid_id: Any = "invalid"
    with pytest.raises(ValidationError):
        # We deliberately pass an invalid type to trigger validation error
        CausalNode(id="gene_a", codex_concept_id=invalid_id, is_latent=False)


def test_causal_graph_creation_valid() -> None:
    nodes = [
        CausalNode(id="A", codex_concept_id=1, is_latent=False),
        CausalNode(id="B", codex_concept_id=2, is_latent=False),
    ]
    edges = [("A", "B")]
    loop_dynamics = [{"path": ["A", "B"], "type": "ACYCLIC"}]

    graph = CausalGraph(nodes=nodes, edges=edges, loop_dynamics=loop_dynamics, stability_score=0.95)

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.stability_score == 0.95


def test_causal_graph_edge_validation_missing_source() -> None:
    nodes = [CausalNode(id="B", codex_concept_id=2, is_latent=False)]
    edges = [("A", "B")]  # 'A' is not in nodes

    with pytest.raises(ValidationError) as exc_info:
        CausalGraph(nodes=nodes, edges=edges, loop_dynamics=[], stability_score=0.9)
    assert "Edge source 'A' not found in nodes" in str(exc_info.value)


def test_causal_graph_edge_validation_missing_target() -> None:
    nodes = [CausalNode(id="A", codex_concept_id=1, is_latent=False)]
    edges = [("A", "B")]  # 'B' is not in nodes

    with pytest.raises(ValidationError) as exc_info:
        CausalGraph(nodes=nodes, edges=edges, loop_dynamics=[], stability_score=0.9)
    assert "Edge target 'B' not found in nodes" in str(exc_info.value)


def test_intervention_result_creation() -> None:
    result = InterventionResult(
        patient_id="p123",
        intervention="do(Dose=50)",
        counterfactual_outcome=0.85,
        confidence_interval=(0.80, 0.90),
        refutation_status="PASSED",
    )
    assert result.patient_id == "p123"
    assert result.confidence_interval == (0.80, 0.90)


def test_intervention_result_validation_error() -> None:
    # Use Any to bypass mypy static checking so we can test runtime validation
    invalid_outcome: Any = "not_a_float"

    with pytest.raises(ValidationError):
        InterventionResult(
            patient_id="p123",
            intervention="do(Dose=50)",
            counterfactual_outcome=invalid_outcome,
            confidence_interval=(0.80, 0.90),
            refutation_status="PASSED",
        )
