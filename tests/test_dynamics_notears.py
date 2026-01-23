# Copyright (c) 2026 CoReason, Inc.
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
import torch

from coreason_inference.analysis.dynamics import DynamicsEngine
from coreason_inference.schema import LoopType


@pytest.fixture
def cyclic_data() -> pd.DataFrame:
    """
    Generates a simple harmonic oscillator time series (A <-> B).
    A = sin(t), B = cos(t) -> dA/dt = B, dB/dt = -A
    """
    t = np.linspace(0, 10, 100)
    a = np.sin(t)
    b = np.cos(t)
    df = pd.DataFrame({"time": t, "A": a, "B": b})
    return df


def test_acyclicity_parameter_validation() -> None:
    """Test that negative acyclicity_lambda raises ValueError."""
    with pytest.raises(ValueError, match="acyclicity_lambda must be non-negative"):
        DynamicsEngine(acyclicity_lambda=-1.0)


def test_acyclicity_constraint_suppresses_loops(cyclic_data: pd.DataFrame) -> None:
    """
    Test that a high acyclicity penalty suppresses feedback loops,
    effectively forcing the model to find a DAG approximation (or fail to fit well, but structured as DAG).
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # High penalty for cycles
    engine = DynamicsEngine(epochs=300, learning_rate=0.01, acyclicity_lambda=10.0)
    engine.fit(cyclic_data, "time", ["A", "B"])

    # Check the acyclicity constraint value directly
    # h(W) should be close to 0
    with torch.no_grad():
        h_val = engine._compute_acyclicity_constraint(engine.model.W).item()

    # h(W) is non-negative.
    # For a 2-node graph, if W has both edges, h > 0. If one edge, h = 0.
    # We expect h to be very small.
    assert h_val < 1e-3, f"Acyclicity constraint failed: h(W) = {h_val}"

    # Verify via graph discovery
    graph = engine.discover_loops(threshold=0.05)

    # Should NOT have loops (or at least not the feedback loop)
    # If A->B and B->A are both present, it's a loop.
    # NOTEARS should kill one direction.

    # We check if there are any length-2 loops
    loops = [l for l in graph.loop_dynamics if len(l.path) == 3] # A-B-A
    assert len(loops) == 0, f"Found loops despite high acyclicity penalty: {loops}"


def test_acyclicity_constraint_allows_loops_when_zero(cyclic_data: pd.DataFrame) -> None:
    """
    Test that default acyclicity_lambda=0 allows loops (Baseline).
    """
    torch.manual_seed(42)
    np.random.seed(42)

    engine = DynamicsEngine(epochs=300, learning_rate=0.01, acyclicity_lambda=0.0)
    engine.fit(cyclic_data, "time", ["A", "B"])

    # Check h(W) might be > 0
    with torch.no_grad():
        h_val = engine._compute_acyclicity_constraint(engine.model.W).item()

    # We expect it to find the loop, so h > 0 (unless it found a diagonal only, but A<->B implies cycles)
    # Actually A<->B is a cycle.
    # Note: h(W) logic uses W*W. If W[i,j] and W[j,i] are non-zero, exp(W*W) trace grows.

    graph = engine.discover_loops(threshold=0.05)
    loops = [l for l in graph.loop_dynamics if len(l.path) == 3]

    # Should find the loop
    assert len(loops) > 0, "Failed to find loop with zero penalty"
    assert h_val > 1e-4, f"h(W) unexpectedly small for cyclic graph: {h_val}"
