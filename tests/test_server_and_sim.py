import numpy as np
import pytest
from fastapi.testclient import TestClient
from coreason_inference.server import app

def test_analyze_causal_dynamics():
    # Create dummy data
    # Convert numpy float to python float for json serialization
    data = [{"X": float(np.sin(t)), "Y": float(np.cos(t)), "time": float(t)} for t in np.linspace(0, 10, 20)]

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

def test_simulate_virtual():
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
