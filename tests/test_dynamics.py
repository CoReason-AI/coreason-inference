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
from coreason_inference.schema import CausalGraph, CausalNode, LoopType


@pytest.fixture
def acyclic_data() -> pd.DataFrame:
    """
    Generates data for a simple decay system: dy/dt = -0.5 * y
    """
    t = np.linspace(0, 5, 50)
    y0 = 10.0
    # Analytical solution: y(t) = y0 * exp(-0.5 * t)
    y = y0 * np.exp(-0.5 * t)
    return pd.DataFrame({"time": t, "variable_a": y})


@pytest.fixture
def cyclic_data() -> pd.DataFrame:
    """
    Generates data for a DAMPED harmonic oscillator (stable negative feedback).
    dy1/dt = y2
    dy2/dt = -y1 - 0.5*y2
    Jacobian: [[0, 1], [-1, -0.5]]
    """
    t = np.linspace(0, 10, 200)  # More points
    y1_vals = []
    y2_vals = []

    # Simulate manually to ensure correct data
    dt = t[1] - t[0]
    y1 = 0.0
    y2 = 1.0

    for _ in t:
        y1_vals.append(y1)
        y2_vals.append(y2)

        dy1 = y2
        dy2 = -y1 - 0.5 * y2

        y1 += dy1 * dt
        y2 += dy2 * dt

    return pd.DataFrame({"time": t, "y1": y1_vals, "y2": y2_vals})


def test_dynamics_fit_acyclic(acyclic_data: pd.DataFrame) -> None:
    """
    Test fitting on a simple acyclic decay system.
    Expect: Self-loop with negative weight (decay).
    """
    # Use rk4 for fixed step training (faster/stable for simple gradients)
    engine = DynamicsEngine(learning_rate=0.05, epochs=500, method="rk4")
    engine.fit(acyclic_data, time_col="time", variable_cols=["variable_a"])

    graph = engine.discover_loops(threshold=0.1)

    assert isinstance(graph, CausalGraph)
    assert len(graph.nodes) == 1
    assert graph.nodes[0].id == "variable_a"

    # Should detect negative self-loop (decay)
    assert len(graph.loop_dynamics) >= 1
    loop = graph.loop_dynamics[0]
    assert loop["path"] == ["variable_a", "variable_a"]
    assert loop["type"] == LoopType.NEGATIVE_FEEDBACK.value


def test_dynamics_fit_cyclic(cyclic_data: pd.DataFrame) -> None:
    """
    Test fitting on a cyclic system (Damped Oscillator).
    Expect: Feedback loop between y1 and y2.
    """
    # Use rk4 for stable training
    engine = DynamicsEngine(learning_rate=0.05, epochs=1000, method="rk4")
    engine.fit(cyclic_data, time_col="time", variable_cols=["y1", "y2"])

    # Threshold 0.2
    graph = engine.discover_loops(threshold=0.2)

    # Check nodes
    node_ids = {node.id for node in graph.nodes}
    assert "y1" in node_ids
    assert "y2" in node_ids

    # Check for feedback loop (y1 <-> y2)
    has_loop = False
    for loop in graph.loop_dynamics:
        path = loop["path"]
        # Look for cycle y1->y2->y1
        if "y1" in path and "y2" in path and len(path) == 3:
            has_loop = True
            assert loop["type"] == LoopType.NEGATIVE_FEEDBACK.value

    assert has_loop, f"Failed to detect feedback loop. Found loops: {graph.loop_dynamics}"

    # Stability score should be < 0 for damped oscillator
    # Eigenvalues of [[0, 1], [-1, -0.5]] are approx -0.25 +/- 0.97i
    # Real part is -0.25
    assert graph.stability_score < 0.0


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
    with pytest.raises(ValueError, match="Duplicate node IDs"):
        CausalGraph(nodes=[node_a, node_a], edges=[], loop_dynamics=[], stability_score=0.0)

    # 2. Edge referencing unknown node
    with pytest.raises(ValueError, match="Edge source 'B' not found"):
        CausalGraph(nodes=[node_a], edges=[("B", "A")], loop_dynamics=[], stability_score=0.0)

    # 3. Loop path integrity
    with pytest.raises(ValueError, match="Loop path node 'B' not found"):
        CausalGraph(
            nodes=[node_a], edges=[], loop_dynamics=[{"path": ["A", "B", "A"], "type": "NEGATIVE"}], stability_score=0.0
        )

    # 4. Loop path format
    with pytest.raises(ValueError, match="Loop path must be a list"):
        CausalGraph(
            nodes=[node_a],
            edges=[],
            loop_dynamics=[{"path": "NOT_A_LIST", "type": "NEGATIVE"}],
            stability_score=0.0,
        )
