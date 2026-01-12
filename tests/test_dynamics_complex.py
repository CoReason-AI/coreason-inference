# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

import numpy as np
import pandas as pd
import pytest

from coreason_inference.analysis.dynamics import DynamicsEngine
from coreason_inference.schema import LoopType


def test_positive_feedback() -> None:
    """
    Test a system with positive feedback (Runaway growth).
    dy/dt = 0.5 * y
    """
    t = np.linspace(0, 2, 20)
    y0 = 1.0
    y = y0 * np.exp(0.5 * t)
    data = pd.DataFrame({"time": t, "y": y})

    # Use rk4 for stability
    engine = DynamicsEngine(learning_rate=0.05, epochs=500, method="rk4")
    engine.fit(data, time_col="time", variable_cols=["y"])

    graph = engine.discover_loops(threshold=0.1)

    assert graph.nodes[0].id == "y"
    # Should detect POSITIVE feedback
    assert len(graph.loop_dynamics) == 1
    assert graph.loop_dynamics[0]["type"] == LoopType.POSITIVE_FEEDBACK.value
    # Stability score should be positive
    assert graph.stability_score > 0.0


def test_scale_invariance() -> None:
    """
    Test two coupled variables with vastly different scales.
    y1 range [0, 1000]
    y2 range [0, 0.01]
    System:
    dy1/dt = 1000 * y2  (Effect of small y2 on large y1)
    dy2/dt = -0.000001 * y1 (Effect of large y1 on small y2)
    This is effectively a harmonic oscillator but scaled.
    """
    # Normalized equivalents: u1 = y1/1000, u2 = y2/0.01
    # We simulate the scaled version and map back to test the scaler logic.
    # Actually, let's just create a damped oscillator data and scale it manually.
    t = np.linspace(0, 5, 50)
    y1_base = np.sin(t)
    y2_base = np.cos(t)

    # Scale them
    y1_scaled = y1_base * 1000.0
    y2_scaled = y2_base * 0.01

    data = pd.DataFrame({"time": t, "y1": y1_scaled, "y2": y2_scaled})

    engine = DynamicsEngine(learning_rate=0.05, epochs=600, method="rk4")
    engine.fit(data, time_col="time", variable_cols=["y1", "y2"])

    # The learned weights will be on the SCALED space (normalized).
    # So the engine should still discover the loop structure regardless of input scale.
    graph = engine.discover_loops(threshold=0.1)

    # Check for feedback loop
    has_loop = False
    for loop in graph.loop_dynamics:
        path = loop["path"]
        if "y1" in path and "y2" in path and len(path) == 3:
            has_loop = True

    assert has_loop, "Failed to detect loop in scaled data."


def test_edge_case_empty_data() -> None:
    engine = DynamicsEngine()
    empty_df = pd.DataFrame({"time": [], "y": []})
    with pytest.raises(ValueError, match="Input data is empty"):
        engine.fit(empty_df, "time", ["y"])


def test_edge_case_nans() -> None:
    engine = DynamicsEngine()
    df = pd.DataFrame({"time": [0, 1, 2], "y": [1.0, np.nan, 2.0]})
    with pytest.raises(ValueError, match="Input data contains NaN values"):
        engine.fit(df, "time", ["y"])


def test_edge_case_insufficient_points() -> None:
    engine = DynamicsEngine()
    df = pd.DataFrame({"time": [0], "y": [1.0]})
    with pytest.raises(ValueError, match="Insufficient data points"):
        engine.fit(df, "time", ["y"])


def test_noisy_data_robustness() -> None:
    """
    Test that the model can still find the loop with moderate noise.
    Decay: y' = -0.5 y
    """
    t = np.linspace(0, 5, 50)
    y = 10.0 * np.exp(-0.5 * t)
    # Add noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.5, size=y.shape)
    y_noisy = y + noise

    data = pd.DataFrame({"time": t, "y": y_noisy})

    engine = DynamicsEngine(learning_rate=0.05, epochs=500, method="rk4")
    engine.fit(data, time_col="time", variable_cols=["y"])

    graph = engine.discover_loops(threshold=0.1)

    # Should still find the negative self loop
    assert len(graph.loop_dynamics) == 1
    assert graph.loop_dynamics[0]["type"] == LoopType.NEGATIVE_FEEDBACK.value


def test_irregular_time_steps() -> None:
    """
    Test fitting with irregularly sampled time points.
    """
    t = np.array([0.0, 0.1, 0.5, 0.7, 1.2, 2.0, 3.5, 5.0])
    y = 10.0 * np.exp(-0.5 * t)
    data = pd.DataFrame({"time": t, "y": y})

    engine = DynamicsEngine(learning_rate=0.05, epochs=500, method="rk4")
    engine.fit(data, time_col="time", variable_cols=["y"])

    graph = engine.discover_loops(threshold=0.1)
    assert len(graph.loop_dynamics) == 1
    assert graph.loop_dynamics[0]["type"] == LoopType.NEGATIVE_FEEDBACK.value
