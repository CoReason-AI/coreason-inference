# Copyright (c) 2025 CoReason, Inc.
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
import torch

from coreason_inference.analysis.dynamics import DynamicsEngine
from coreason_inference.schema import LoopType


@pytest.fixture
def time_series_data() -> pd.DataFrame:
    """
    Generates a simple harmonic oscillator time series (A <-> B).
    """
    t = np.linspace(0, 10, 100)
    # A = sin(t), B = cos(t) -> dA/dt = B, dB/dt = -A
    a = np.sin(t)
    b = np.cos(t)
    df = pd.DataFrame({"time": t, "A": a, "B": b})
    return df


def test_dynamics_fit_and_discover(time_series_data: pd.DataFrame) -> None:
    """Test that DynamicsEngine fits and discovers the feedback loop."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Increased epochs and reduced LR for better convergence
    engine = DynamicsEngine(epochs=300, learning_rate=0.01, l1_lambda=0.0, jacobian_lambda=0.0)
    engine.fit(time_series_data, "time", ["A", "B"])

    assert engine.model is not None

    graph = engine.discover_loops(threshold=0.05)

    assert len(graph.nodes) == 2
    # Should detect loop A-B-A
    # Since dA/dt = B (positive), dB/dt = -A (negative).
    # A->B weight (d(dB)/dA) is negative.
    # B->A weight (d(dA)/dB) is positive.
    # Product is negative -> Negative Feedback.

    # We expect at least one loop
    assert len(graph.loop_dynamics) > 0
    loop = graph.loop_dynamics[0]
    assert loop.type == LoopType.NEGATIVE_FEEDBACK


def test_dynamics_regularization(time_series_data: pd.DataFrame) -> None:
    """Test that L1 and Jacobian regularization run without error."""
    # Run with heavy regularization to ensure terms are computed
    engine = DynamicsEngine(epochs=10, l1_lambda=0.1, jacobian_lambda=0.1)
    engine.fit(time_series_data, "time", ["A", "B"])
    assert engine.model is not None


def test_validation_errors(time_series_data: pd.DataFrame) -> None:
    """Test input validation error handling."""

    # Invalid constructor params
    with pytest.raises(ValueError, match="l1_lambda must be non-negative"):
        DynamicsEngine(l1_lambda=-1.0)

    with pytest.raises(ValueError, match="jacobian_lambda must be non-negative"):
        DynamicsEngine(jacobian_lambda=-1.0)

    engine = DynamicsEngine()

    # Empty data
    with pytest.raises(ValueError, match="Input data is empty"):
        engine.fit(pd.DataFrame(), "time", ["A"])

    # NaNs in data
    df_nan = time_series_data.copy()
    df_nan.loc[0, "A"] = np.nan
    with pytest.raises(ValueError, match="Input data contains NaN values"):
        engine.fit(df_nan, "time", ["A", "B"])

    # Insufficient data
    df_short = time_series_data.iloc[:1]
    with pytest.raises(ValueError, match="Insufficient data points"):
        engine.fit(df_short, "time", ["A", "B"])


def test_discover_loops_before_fit() -> None:
    """Test that discover_loops raises error if called before fit."""
    engine = DynamicsEngine()
    with pytest.raises(ValueError, match="Model has not been fitted yet"):
        engine.discover_loops()


def test_stability_score_calculation(time_series_data: pd.DataFrame) -> None:
    """Test that stability score is calculated."""
    engine = DynamicsEngine(epochs=10)
    engine.fit(time_series_data, "time", ["A", "B"])
    graph = engine.discover_loops()
    assert isinstance(graph.stability_score, float)


def test_stability_score_error_handling(time_series_data: pd.DataFrame) -> None:
    """Test fallback when eigenvalue calculation fails."""
    engine = DynamicsEngine(epochs=5)
    engine.fit(time_series_data, "time", ["A", "B"])

    with patch("numpy.linalg.eigvals", side_effect=ValueError("Math error")):
        graph = engine.discover_loops()
        assert graph.stability_score == 0.0


def test_l1_nonzero_check(time_series_data: pd.DataFrame) -> None:
    """Explicitly test l1_lambda > 0 branch."""
    engine = DynamicsEngine(epochs=5, l1_lambda=0.01)
    engine.fit(time_series_data, "time", ["A", "B"])
    # If it runs, the branch was executed.


def test_jacobian_nonzero_check(time_series_data: pd.DataFrame) -> None:
    """Explicitly test jacobian_lambda > 0 branch."""
    engine = DynamicsEngine(epochs=5, jacobian_lambda=0.01)
    engine.fit(time_series_data, "time", ["A", "B"])
    # If it runs, the branch was executed.
