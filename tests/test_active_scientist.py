# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from coreason_inference.analysis.active_scientist import ActiveScientist
from coreason_inference.schema import ExperimentProposal


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """
    Generates synthetic data for A - B - C.
    PC algorithm usually returns undirected edges for this chain if v-structure check fails
    or if it's markov equivalent.
    """
    np.random.seed(42)
    n = 1000
    a = np.random.randn(n)
    b = 0.5 * a + np.random.randn(n) * 0.1
    c = 0.5 * b + np.random.randn(n) * 0.1
    df = pd.DataFrame({"A": a, "B": b, "C": c})
    return df


def test_active_scientist_fit(synthetic_data: pd.DataFrame) -> None:
    """Test that ActiveScientist fits without error."""
    scientist = ActiveScientist()
    scientist.fit(synthetic_data)
    assert scientist.cpdag is not None
    assert scientist.cpdag.shape == (3, 3)


def test_active_scientist_proposals_heuristic(synthetic_data: pd.DataFrame) -> None:
    """
    Test the intelligent experiment selection.
    In a chain A - B - C, B is connected to A and C.
    Degree of B is 2. Degree of A is 1. Degree of C is 1.
    The heuristic should pick B.
    """
    scientist = ActiveScientist()
    scientist.fit(synthetic_data)
    proposals = scientist.propose_experiments()

    assert isinstance(proposals, list)
    assert len(proposals) == 1  # Should select the ONE best
    proposal = proposals[0]

    assert isinstance(proposal, ExperimentProposal)

    # We expect B to be the target as it is the central node
    assert proposal.target == "B"
    assert "Degree: 2" in proposal.rationale


def test_empty_data_error() -> None:
    scientist = ActiveScientist()
    with pytest.raises(ValueError, match="Input data is empty"):
        scientist.fit(pd.DataFrame())


def test_propose_without_fit() -> None:
    scientist = ActiveScientist()
    with pytest.raises(ValueError, match="Model not fitted"):
        scientist.propose_experiments()


def test_pc_algorithm_failure(synthetic_data: pd.DataFrame) -> None:
    """Test exception handling when PC algorithm fails."""
    scientist = ActiveScientist()
    with patch("coreason_inference.analysis.active_scientist.pc", side_effect=Exception("PC Error")):
        with pytest.raises(Exception, match="PC Error"):
            scientist.fit(synthetic_data)


def test_no_undirected_edges() -> None:
    """Test case where graph is fully directed or empty, should return empty list."""
    scientist = ActiveScientist()

    # Mock a fully directed graph (DAG)
    # 3 nodes: A->B->C
    # Matrix:
    # 0 -> 1: -1, 1 (Directed)
    # 1 -> 2: -1, 1 (Directed)
    # 0 -> 2: 0, 0
    adj = np.zeros((3, 3))
    adj[0, 1] = -1
    adj[1, 0] = 1  # A -> B
    adj[1, 2] = -1
    adj[2, 1] = 1  # B -> C

    scientist.cpdag = adj
    scientist.labels = ["A", "B", "C"]

    proposals = scientist.propose_experiments()
    assert proposals == []


def test_star_graph_heuristic() -> None:
    """
    Test a Star graph configuration:
         A
         |
    C -- Hub -- D
         |
         B

    Hub should be selected.
    """
    scientist = ActiveScientist()

    labels = ["A", "B", "C", "D", "Hub"]
    n = len(labels)
    # Hub is index 4
    hub_idx = 4

    adj = np.zeros((n, n))

    # Create undirected edges between Hub and everyone else
    for i in range(4):
        # -1 at both ends
        adj[i, hub_idx] = -1
        adj[hub_idx, i] = -1

    scientist.cpdag = adj
    scientist.labels = labels

    proposals = scientist.propose_experiments()
    assert len(proposals) == 1
    assert proposals[0].target == "Hub"
    # Expected degree is 4
    assert "Degree: 4" in proposals[0].rationale
