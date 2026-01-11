# Copyright (c) 2025 CoReason, Inc.
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
    LoopType,
)


def test_loop_type_enum() -> None:
    """Test LoopType Enum values."""
    assert LoopType.POSITIVE_FEEDBACK.value == "POSITIVE"
    assert LoopType.NEGATIVE_FEEDBACK.value == "NEGATIVE"
    assert LoopType.NONE.value == "ACYCLIC"


def test_causal_node_validation() -> None:
    """Test CausalNode Pydantic model validation."""
    # Valid node
    node = CausalNode(id="node_1", codex_concept_id=101, is_latent=False)
    assert node.id == "node_1"
    assert node.codex_concept_id == 101
    assert node.is_latent is False

    # Invalid type for codex_concept_id
    with pytest.raises(ValidationError):
        CausalNode(id="node_2", codex_concept_id="not_an_int", is_latent=True)  # type: ignore

    # Missing field
    with pytest.raises(ValidationError):
        CausalNode(id="node_3", codex_concept_id=102)  # type: ignore


def test_causal_graph_validation() -> None:
    """Test CausalGraph Pydantic model validation."""
    node1 = CausalNode(id="A", codex_concept_id=1, is_latent=False)
    node2 = CausalNode(id="B", codex_concept_id=2, is_latent=False)

    # Valid graph
    graph = CausalGraph(
        nodes=[node1, node2],
        edges=[("A", "B"), ("B", "A")],
        loop_dynamics=[{"path": ["A", "B", "A"], "type": "NEGATIVE"}],
        stability_score=0.85,
    )
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 2
    assert graph.loop_dynamics[0]["type"] == "NEGATIVE"
    assert graph.stability_score == 0.85

    # Invalid stability_score type
    with pytest.raises(ValidationError):
        CausalGraph(
            nodes=[node1],
            edges=[],
            loop_dynamics=[],
            stability_score="high",  # type: ignore
        )


def test_intervention_result_validation() -> None:
    """Test InterventionResult Pydantic model validation."""
    # Valid result
    result = InterventionResult(
        patient_id="p_001",
        intervention="do(Dose=50)",
        counterfactual_outcome=120.5,
        confidence_interval=(115.0, 125.0),
        refutation_status="PASSED",
    )
    assert result.patient_id == "p_001"
    assert result.counterfactual_outcome == 120.5
    assert result.confidence_interval == (115.0, 125.0)

    # Invalid confidence interval (tuple of length 2 expected)
    with pytest.raises(ValidationError):
        InterventionResult(
            patient_id="p_002",
            intervention="do(X=1)",
            counterfactual_outcome=10.0,
            confidence_interval=(10.0,),  # type: ignore
            refutation_status="FAILED",
        )


def test_serialization() -> None:
    """Test JSON serialization of models."""
    node = CausalNode(id="A", codex_concept_id=1, is_latent=False)
    json_str = node.model_dump_json()
    assert '"id":"A"' in json_str
    assert '"codex_concept_id":1' in json_str

    node_loaded = CausalNode.model_validate_json(json_str)
    assert node_loaded == node


def test_graph_integrity_duplicates() -> None:
    """Test that graph rejects duplicate node IDs."""
    node1 = CausalNode(id="A", codex_concept_id=1, is_latent=False)
    node2 = CausalNode(id="A", codex_concept_id=2, is_latent=True)  # Duplicate ID

    with pytest.raises(ValidationError) as excinfo:
        CausalGraph(
            nodes=[node1, node2],
            edges=[],
            loop_dynamics=[],
            stability_score=1.0,
        )
    assert "Duplicate node IDs found in graph" in str(excinfo.value)


def test_graph_integrity_dangling_edges() -> None:
    """Test that graph rejects edges pointing to non-existent nodes."""
    node1 = CausalNode(id="A", codex_concept_id=1, is_latent=False)

    # Edge A->B where B does not exist
    with pytest.raises(ValidationError) as excinfo:
        CausalGraph(
            nodes=[node1],
            edges=[("A", "B")],
            loop_dynamics=[],
            stability_score=1.0,
        )
    assert "Edge target 'B' not found" in str(excinfo.value)

    # Edge C->A where C does not exist
    with pytest.raises(ValidationError) as excinfo:
        CausalGraph(
            nodes=[node1],
            edges=[("C", "A")],
            loop_dynamics=[],
            stability_score=1.0,
        )
    assert "Edge source 'C' not found" in str(excinfo.value)


def test_graph_integrity_invalid_loops() -> None:
    """Test that graph rejects loop paths referencing non-existent nodes."""
    node1 = CausalNode(id="A", codex_concept_id=1, is_latent=False)

    # Loop path involves 'B' which does not exist
    with pytest.raises(ValidationError) as excinfo:
        CausalGraph(
            nodes=[node1],
            edges=[],
            loop_dynamics=[{"path": ["A", "B"], "type": "NEGATIVE"}],
            stability_score=1.0,
        )
    assert "Loop path node 'B' not found" in str(excinfo.value)

    # Loop path is not a list
    with pytest.raises(ValidationError) as excinfo:
        CausalGraph(
            nodes=[node1],
            edges=[],
            loop_dynamics=[{"path": "A->B", "type": "NEGATIVE"}],
            stability_score=1.0,
        )
    assert "Loop path must be a list" in str(excinfo.value)
