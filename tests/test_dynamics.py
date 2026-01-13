# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

import math

import pandas as pd
import pytest
import torch
from coreason_inference.analysis.dynamics import DynamicsEngine
from coreason_inference.schema import CausalGraph, CausalNode, LoopDynamics, LoopType
from pydantic import ValidationError
from torchdiffeq import odeint as torch_odeint


@pytest.fixture
def acyclic_data() -> pd.DataFrame:
    """
    Generates data for a simple decay system: dy/dt = -0.5 * y
    """
    t = torch.linspace(0, 5, 50)
    y0 = torch.tensor([10.0])

    class DecayDynamics(torch.nn.Module):  # type: ignore[misc]
        def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return -0.5 * y

    with torch.no_grad():
        y = torch_odeint(DecayDynamics(), y0, t, method="dopri5")

    return pd.DataFrame({"time": t.numpy(), "variable_a": y.squeeze().numpy()})


@pytest.fixture
def cyclic_data() -> pd.DataFrame:
    """
    Generates data for a PURE harmonic oscillator (marginally stable).
    dy1/dt = y2
    dy2/dt = -y1
    Jacobian: [[0, 1], [-1, 0]]
    Period: 2*pi approx 6.28
    Time: 0 to 6.28 (One full cycle).
    This ensures mean is approx 0, making StandardScaler consistent.
    """
    period = 2 * math.pi
    # Reduced points to 50 for faster training
    t = torch.linspace(0, period, 50)
    y0 = torch.tensor([0.0, 1.0])

    class HarmonicDynamics(torch.nn.Module):  # type: ignore[misc]
        def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # y is [y1, y2]
            dydt = torch.zeros_like(y)
            dydt[0] = y[1]
            dydt[1] = -y[0]
            return dydt

    with torch.no_grad():
        y_true = torch_odeint(HarmonicDynamics(), y0, t, method="dopri5")

    y1_vals = y_true[:, 0].numpy()
    y2_vals = y_true[:, 1].numpy()
    t_vals = t.numpy()

    return pd.DataFrame({"time": t_vals, "y1": y1_vals, "y2": y2_vals})


def test_dynamics_fit_acyclic(acyclic_data: pd.DataFrame) -> None:
    """
    Test fitting on a simple acyclic decay system.
    Expect: Self-loop with negative weight (decay).
    """
    # Use rk4 for fixed step training
    engine = DynamicsEngine(learning_rate=0.05, epochs=800, method="rk4")
    engine.fit(acyclic_data, time_col="time", variable_cols=["variable_a"])

    # Reduced threshold slightly to be safe against normalization effects
    graph = engine.discover_loops(threshold=0.05)

    assert isinstance(graph, CausalGraph)
    assert len(graph.nodes) == 1
    assert graph.nodes[0].id == "variable_a"

    # Should detect negative self-loop (decay)
    assert len(graph.loop_dynamics) >= 1
    loop = graph.loop_dynamics[0]
    # Check using object attributes, not dictionary keys
    assert loop.path == ["variable_a", "variable_a"]
    assert loop.type == LoopType.NEGATIVE_FEEDBACK


def test_dynamics_fit_cyclic(cyclic_data: pd.DataFrame) -> None:
    """
    Test fitting on a cyclic system (Harmonic Oscillator).
    Expect: Feedback loop between y1 and y2.
    """
    # Use rk4 for stable training
    # Optimized epochs/LR for speed and convergence
    engine = DynamicsEngine(learning_rate=0.02, epochs=2000, method="rk4")
    engine.fit(cyclic_data, time_col="time", variable_cols=["y1", "y2"])

    # Threshold 0.05 to ensure off-diagonal interactions are detected
    graph = engine.discover_loops(threshold=0.05)

    # Check nodes
    node_ids = {node.id for node in graph.nodes}
    assert "y1" in node_ids
    assert "y2" in node_ids

    # Check for feedback loop (y1 <-> y2)
    has_loop = False
    for loop in graph.loop_dynamics:
        path = loop.path
        # Look for cycle y1->y2->y1
        if "y1" in path and "y2" in path and len(path) == 3:
            has_loop = True
            assert loop.type == LoopType.NEGATIVE_FEEDBACK

    # Debug info if failed
    if not has_loop:
        print(f"Detected Loops: {graph.loop_dynamics}")

    assert has_loop, f"Failed to detect feedback loop. Found loops: {graph.loop_dynamics}"

    # Stability score should be around 0.
    # We check < 0.5 which covers 0.
    assert graph.stability_score < 0.5


def test_dynamics_not_fitted() -> None:
    """
    Test error when calling discover_loops before fit.
    """
    engine = DynamicsEngine()
    with pytest.raises(ValueError, match="Model has not been fitted yet"):
        engine.discover_loops()


def test_graph_validation() -> None:
    """
    Test CausalGraph validation logic (coverage for schema.py).
    """
    node_a = CausalNode(id="A", codex_concept_id=1, is_latent=False)

    # 1. Duplicate IDs
    # Updated match string to match actual error
    with pytest.raises(ValueError, match="Duplicate node ID found: 'A'"):
        CausalGraph(nodes=[node_a, node_a], edges=[], loop_dynamics=[], stability_score=0.0)

    # 2. Edge referencing unknown node
    with pytest.raises(ValueError, match="Edge source node 'B' not found"):
        CausalGraph(nodes=[node_a], edges=[("B", "A")], loop_dynamics=[], stability_score=0.0)

    # 3. Loop path integrity
    # LoopDynamics must be valid (path length >= 2), but edge must fail
    loop = LoopDynamics(path=["A", "B", "A"], type=LoopType.NEGATIVE_FEEDBACK)
    with pytest.raises(ValueError, match=r"Loop path edge \('A', 'B'\) does not exist"):
        CausalGraph(nodes=[node_a], edges=[], loop_dynamics=[loop], stability_score=0.0)

    # 4. Loop path format is now handled by LoopDynamics type check
    # But we can test LoopDynamics validation independently or via graph
    # If we pass a dict to loop_dynamics in constructor, Pydantic should raise a validation error
    with pytest.raises(ValidationError):  # Catch Pydantic validation error explicitly
        CausalGraph(
            nodes=[node_a],
            edges=[],
            loop_dynamics=[{"path": "NOT_A_LIST", "type": "NEGATIVE"}],  # type: ignore
            stability_score=0.0,
        )
