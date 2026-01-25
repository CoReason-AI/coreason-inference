import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from coreason_inference.server import app
from coreason_inference.schema import CausalGraph, CausalNode

def test_analyze_causal_dynamics():
    # Create dummy data
    data = [{"X": float(np.sin(t)), "Y": float(np.cos(t)), "time": float(t)} for t in np.linspace(0, 10, 20)]

    # Mock return value for discover_loops
    mock_graph = CausalGraph(
        nodes=[
            CausalNode(id="X", codex_concept_id=1, is_latent=False),
            CausalNode(id="Y", codex_concept_id=2, is_latent=False)
        ],
        edges=[("X", "Y")],
        loop_dynamics=[],
        stability_score=0.95
    )

    with patch("coreason_inference.server.DynamicsEngine") as MockEngine:
        instance = MockEngine.return_value
        instance.discover_loops.return_value = mock_graph

        with TestClient(app) as client:
            response = client.post(
                "/analyze/causal",
                json={
                    "dataset": data,
                    "variables": ["X", "Y"],
                    "method": "dynamics"
                }
            )
            assert response.status_code == 200
            json_resp = response.json()
            assert "graph" in json_resp
            assert "metrics" in json_resp
            assert "stability_score" in json_resp["metrics"]

            # Verify fit was called
            instance.fit.assert_called()

def test_simulate_virtual():
    # This test relies on the lifespan handler which loads a real (but small) model
    with TestClient(app) as client:
        initial_state = {"X": 0.0, "Y": 1.0}
        response = client.post(
            "/simulate/virtual",
            json={
                "initial_state": initial_state,
                "steps": 5,
                "intervention": {"X": 0.5}
            }
        )
        if response.status_code != 200:
            print(response.json())
        assert response.status_code == 200
        json_resp = response.json()
        assert "trajectory" in json_resp
        traj = json_resp["trajectory"]
        assert len(traj) == 6 # steps + 1

        # Verify intervention
        assert abs(traj[0]["X"] - 0.5) < 1e-4
        assert abs(traj[-1]["X"] - 0.5) < 1e-4

def test_analyze_causal_pc():
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
            response = client.post(
                "/analyze/causal",
                json={
                    "dataset": data,
                    "variables": ["X", "Y"],
                    "method": "pc"
                }
            )
            assert response.status_code == 200
            json_resp = response.json()
            assert "graph" in json_resp
            assert "nodes" in json_resp["graph"]

            # Check edge X->Y found
            edges = json_resp["graph"]["edges"]
            assert ["X", "Y"] in edges or [["X", "Y"]] in edges or ("X", "Y") in edges
