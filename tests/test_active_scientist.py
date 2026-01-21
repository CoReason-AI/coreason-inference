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
    Generates synthetic data for A -> B <- C
    This pattern (v-structure) is identifiable.

    However, A - D - B is not always identifiable.
    Let's create a chain A -> B -> C.
    In PC, A -> B -> C is Markov Equivalent to A <- B <- C and A <- B -> C.
    So the edges should be undirected in the CPDAG: A - B - C.
    We expect proposals for A-B and B-C.
    """
    np.random.seed(42)
    n = 1000
    a = np.random.randn(n)
    b = 0.5 * a + np.random.randn(n) * 0.1  # Strong correlation
    c = 0.5 * b + np.random.randn(n) * 0.1

    df = pd.DataFrame({"A": a, "B": b, "C": c})
    return df


def test_active_scientist_fit(synthetic_data: pd.DataFrame) -> None:
    """Test that ActiveScientist fits without error."""
    scientist = ActiveScientist()
    scientist.fit(synthetic_data)
    assert scientist.cpdag is not None
    assert scientist.cpdag.shape == (3, 3)


def test_active_scientist_proposals(synthetic_data: pd.DataFrame) -> None:
    """
    Test that ActiveScientist proposes experiments for ambiguous edges.
    For A -> B -> C, the skeleton is A-B, B-C.
    V-structures: None (because B is a mediator).
    So PC should return A - B - C (undirected).
    We expect proposals for A-B and B-C.
    """
    scientist = ActiveScientist()
    scientist.fit(synthetic_data)
    proposals = scientist.propose_experiments()

    assert isinstance(proposals, list)
    assert len(proposals) > 0
    assert isinstance(proposals[0], ExperimentProposal)

    # Check content
    targets = [p.target for p in proposals]
    # We expect intervention on A or B (for A-B edge) and B or C (for B-C edge).
    # Since we iterate, we likely see 'A' (for A-B) and 'B' (for B-C).

    # Verify we have found ambiguities
    assert "A" in targets or "B" in targets


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
