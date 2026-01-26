from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from coreason_inference.analysis.virtual_simulator import VirtualSimulator
from coreason_inference.schema import CausalGraph, CausalNode
from coreason_inference.server import app


def test_analyze_causal_dynamics() -> None:
    # Create dummy data
    data = [{"X": float(np.sin(t)), "Y": float(np.cos(t)), "time": float(t)} for t in np.linspace(0, 10, 20)]

    # Mock return value for discover_loops
    mock_graph = CausalGraph(
        nodes=[
            CausalNode(id="X", codex_concept_id=1, is_latent=False),
            CausalNode(id="Y", codex_concept_id=2, is_latent=False),
        ],
        edges=[("X", "Y")],
        loop_dynamics=[],
        stability_score=0.95,
    )

    with patch("coreason_inference.server.DynamicsEngine") as MockEngine:
        instance = MockEngine.return_value
        instance.discover_loops.return_value = mock_graph

        with TestClient(app) as client:
            response = client.post(
                "/analyze/causal", json={"dataset": data, "variables": ["X", "Y"], "method": "dynamics"}
            )
            assert response.status_code == 200
            json_resp = response.json()
            assert "graph" in json_resp
            assert "metrics" in json_resp
            assert "stability_score" in json_resp["metrics"]

            # Verify fit was called
            instance.fit.assert_called()


def test_analyze_causal_empty_dataset() -> None:
    with TestClient(app) as client:
        response = client.post("/analyze/causal", json={"dataset": [], "variables": ["X", "Y"], "method": "dynamics"})
        assert response.status_code == 400
        assert "Dataset is empty" in response.json()["detail"]


def test_analyze_causal_missing_variable_dynamics() -> None:
    data = [{"X": 1.0, "time": 0.0}]
    with TestClient(app) as client:
        response = client.post("/analyze/causal", json={"dataset": data, "variables": ["X", "Y"], "method": "dynamics"})
        assert response.status_code == 400
        assert "Variable 'Y' not found" in response.json()["detail"]


def test_analyze_causal_missing_time_col_dynamics() -> None:
    # Test that time col is added if missing
    data = [{"X": 1.0, "Y": 1.0}]  # No time
    mock_graph = CausalGraph(nodes=[], edges=[], loop_dynamics=[], stability_score=0.0)

    with patch("coreason_inference.server.DynamicsEngine") as MockEngine:
        instance = MockEngine.return_value
        instance.discover_loops.return_value = mock_graph

        with TestClient(app) as client:
            response = client.post(
                "/analyze/causal", json={"dataset": data, "variables": ["X", "Y"], "method": "dynamics"}
            )
            assert response.status_code == 200
            # Ensure fit was called with generated time col (index)
            call_args = instance.fit.call_args
            assert call_args is not None
            df_arg = call_args[0][0]
            assert "time" in df_arg.columns


def test_analyze_causal_dynamics_exception() -> None:
    data = [{"X": 1.0, "time": 0.0}]
    with patch("coreason_inference.server.DynamicsEngine") as MockEngine:
        instance = MockEngine.return_value
        instance.fit.side_effect = Exception("Fit failed")

        with TestClient(app) as client:
            response = client.post("/analyze/causal", json={"dataset": data, "variables": ["X"], "method": "dynamics"})
            assert response.status_code == 500
            assert "Analysis failed" in response.json()["detail"]


def test_analyze_causal_pc() -> None:
    # Create dummy data with more samples for PC
    t = np.linspace(0, 10, 100)
    data = [{"X": float(np.sin(x)), "Y": float(np.cos(x))} for x in t]

    # Mock ActiveScientist to avoid heavy PC algorithm dependency/time
    with patch("coreason_inference.server.ActiveScientist") as MockScientist:
        instance = MockScientist.return_value
        # Mock CPDAG: X -> Y
        # M[i, j] is endpoint at j.
        # X is index 0, Y is index 1.
        # X -> Y: M[0, 1] = 1 (Arrow), M[1, 0] = -1 (Tail)
        mock_cpdag = np.zeros((2, 2))
        mock_cpdag[0, 1] = 1
        mock_cpdag[1, 0] = -1

        instance.cpdag = mock_cpdag
        instance.labels = ["X", "Y"]

        with TestClient(app) as client:
            response = client.post("/analyze/causal", json={"dataset": data, "variables": ["X", "Y"], "method": "pc"})
            assert response.status_code == 200
            json_resp = response.json()
            assert "graph" in json_resp
            assert "nodes" in json_resp["graph"]

            # Check edge X->Y found
            edges = json_resp["graph"]["edges"]
            assert ["X", "Y"] in edges or [["X", "Y"]] in edges or ("X", "Y") in edges


def test_analyze_causal_missing_variable_pc() -> None:
    data = [{"X": 1.0}]
    with TestClient(app) as client:
        response = client.post("/analyze/causal", json={"dataset": data, "variables": ["X", "Y"], "method": "pc"})
        assert response.status_code == 400
        assert "Variable 'Y' not found" in response.json()["detail"]


def test_analyze_causal_pc_exception() -> None:
    data = [{"X": 1.0}]
    with patch("coreason_inference.server.ActiveScientist") as MockScientist:
        instance = MockScientist.return_value
        instance.fit.side_effect = Exception("PC failed")

        with TestClient(app) as client:
            response = client.post("/analyze/causal", json={"dataset": data, "variables": ["X"], "method": "pc"})
            assert response.status_code == 500
            assert "Analysis failed" in response.json()["detail"]


def test_analyze_causal_unknown_method() -> None:
    data = [{"X": 1.0}]
    with TestClient(app) as client:
        response = client.post("/analyze/causal", json={"dataset": data, "variables": ["X"], "method": "magic"})
        assert response.status_code == 400
        assert "Unknown method" in response.json()["detail"]


def test_simulate_virtual() -> None:
    # This test relies on the lifespan handler which loads a real (but small) model
    with TestClient(app) as client:
        initial_state = {"X": 0.0, "Y": 1.0}
        response = client.post(
            "/simulate/virtual", json={"initial_state": initial_state, "steps": 5, "intervention": {"X": 0.5}}
        )
        if response.status_code != 200:
            print(response.json())
        assert response.status_code == 200
        json_resp = response.json()
        assert "trajectory" in json_resp
        traj = json_resp["trajectory"]
        assert len(traj) == 6  # steps + 1

        # Verify intervention
        assert abs(traj[0]["X"] - 0.5) < 1e-4
        assert abs(traj[-1]["X"] - 0.5) < 1e-4


def test_simulate_virtual_no_model() -> None:
    # Force models to be empty
    from coreason_inference.server import models

    with TestClient(app) as client:
        # Clear models AFTER client startup (lifespan has run)
        models.clear()

        response = client.post("/simulate/virtual", json={"initial_state": {"X": 0.0}, "steps": 5})
        assert response.status_code == 503
        assert "Simulation model not initialized" in response.json()["detail"]


def test_simulate_virtual_value_error() -> None:
    # To trigger ValueError, we can mock VirtualSimulator to raise it
    with patch("coreason_inference.server.VirtualSimulator") as MockSim:
        instance = MockSim.return_value
        instance.simulate_trajectory.side_effect = ValueError("Invalid input")

        # Restore model
        from coreason_inference.server import models

        models["default_dynamics"] = MagicMock()

        with TestClient(app) as client:
            response = client.post("/simulate/virtual", json={"initial_state": {"X": 0.0}, "steps": 5})
            assert response.status_code == 400
            assert "Invalid input" in response.json()["detail"]


def test_simulate_virtual_exception() -> None:
    # To trigger general Exception
    with patch("coreason_inference.server.VirtualSimulator") as MockSim:
        instance = MockSim.return_value
        instance.simulate_trajectory.side_effect = Exception("Unexpected error")

        from coreason_inference.server import models

        models["default_dynamics"] = MagicMock()

        with TestClient(app) as client:
            response = client.post("/simulate/virtual", json={"initial_state": {"X": 0.0}, "steps": 5})
            assert response.status_code == 500
            assert "Unexpected error" in response.json()["detail"]


def test_lifespan_exception() -> None:
    # Test that lifespan handles exceptions during startup
    with patch("coreason_inference.server.DynamicsEngine") as MockEngine:
        MockEngine.side_effect = Exception("Startup failed")

        # We need to manually invoke lifespan since TestClient handles it differently usually
        import asyncio

        from coreason_inference.server import lifespan

        async def run_lifespan() -> None:
            async with lifespan(app):
                pass

        # Should catch exception and log it, not raise
        asyncio.run(run_lifespan())
        # We can verified log call if we mocked logger, but execution without raise confirms safety


# --- VirtualSimulator Unit Tests for Coverage ---


def test_simulate_trajectory_no_model() -> None:
    sim = VirtualSimulator()
    with pytest.raises(ValueError, match="No model provided"):
        sim.simulate_trajectory(initial_state={}, steps=10, model=None)


def test_simulate_trajectory_model_no_varnames() -> None:
    sim = VirtualSimulator()
    model = MagicMock()
    model.variable_names = []
    with pytest.raises(ValueError, match="Model has no variable names"):
        sim.simulate_trajectory(initial_state={}, steps=10, model=model)


def test_simulate_trajectory_model_no_odefunc() -> None:
    sim = VirtualSimulator()
    model = MagicMock()
    model.variable_names = ["X"]
    model.model = None
    with pytest.raises(ValueError, match="Model has no internal ODEFunc"):
        sim.simulate_trajectory(initial_state={}, steps=10, model=model)


def test_simulate_trajectory_missing_initial() -> None:
    sim = VirtualSimulator()
    model = MagicMock()
    model.variable_names = ["X", "Y"]
    model.model = MagicMock()
    model.scaler = None

    with pytest.raises(ValueError, match="Missing initial value"):
        sim.simulate_trajectory(initial_state={"X": 0.0}, steps=10, model=model)


def test_simulate_trajectory_no_scaler() -> None:
    # Test path where model.scaler is None
    sim = VirtualSimulator()
    model = MagicMock()
    model.variable_names = ["X"]
    model.scaler = None
    model.method = "dopri5"

    # Mock ODE func
    def odefunc(t: Any, y: Any) -> Any:
        return y * 0.0

    model.model = odefunc

    traj = sim.simulate_trajectory(initial_state={"X": 1.0}, steps=1, model=model)
    assert len(traj) == 2
    assert traj[0]["X"] == 1.0
