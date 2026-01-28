from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from coreason_inference.analysis.virtual_simulator import VirtualSimulator
from coreason_inference.schema import CausalGraph, CausalNode
from coreason_inference.engine import InferenceResult
from coreason_inference.server import app


@pytest.fixture
def mock_inference_result():
    graph = CausalGraph(
        nodes=[
            CausalNode(id="X", codex_concept_id=1, is_latent=False),
            CausalNode(id="Y", codex_concept_id=2, is_latent=False)
        ],
        edges=[("X", "Y")],
        loop_dynamics=[],
        stability_score=0.95
    )
    return InferenceResult(
        graph=graph,
        latents=pd.DataFrame(),
        proposals=[],
        augmented_data=pd.DataFrame()
    )


def test_analyze_causal_success(mock_inference_result) -> None:
    # Create dummy data
    data = [{"X": 1.0, "Y": 1.0, "time": 0.0}]

    with patch("coreason_inference.server.InferenceEngineAsync") as MockEngineCls:
        mock_engine = MockEngineCls.return_value
        # Setup async context manager
        mock_engine.__aenter__.return_value = mock_engine
        mock_engine.__aexit__ = AsyncMock(return_value=None)

        mock_engine.analyze = AsyncMock(return_value=mock_inference_result)

        with TestClient(app) as client:
            payload = {"dataset": data, "variables": ["X", "Y"], "method": "dynamics"}
            headers = {"X-User-Sub": "test", "X-User-Email": "t@e.com"}
            response = client.post(
                "/analyze/causal", json=payload, headers=headers
            )
            assert response.status_code == 200
            json_resp = response.json()
            assert "graph" in json_resp
            assert json_resp["metrics"]["stability_score"] == 0.95

            mock_engine.analyze.assert_awaited_once()


def test_analyze_causal_empty_dataset() -> None:
    with TestClient(app) as client:
        payload = {"dataset": [], "variables": ["X", "Y"], "method": "dynamics"}
        headers = {"X-User-Sub": "test", "X-User-Email": "t@e.com"}
        response = client.post("/analyze/causal", json=payload, headers=headers)
        assert response.status_code == 400
        assert "Dataset is empty" in response.json()["detail"]


def test_analyze_causal_missing_variable() -> None:
    data = [{"X": 1.0, "time": 0.0}]
    with TestClient(app) as client:
        payload = {"dataset": data, "variables": ["X", "Y"], "method": "dynamics"}
        headers = {"X-User-Sub": "test", "X-User-Email": "t@e.com"}
        response = client.post("/analyze/causal", json=payload, headers=headers)
        assert response.status_code == 400
        assert "Variable 'Y' not found" in response.json()["detail"]


def test_analyze_causal_exception() -> None:
    data = [{"X": 1.0, "time": 0.0}]
    with patch("coreason_inference.server.InferenceEngineAsync") as MockEngineCls:
        mock_engine = MockEngineCls.return_value
        mock_engine.__aenter__.return_value = mock_engine
        mock_engine.__aexit__ = AsyncMock(return_value=None)

        mock_engine.analyze.side_effect = Exception("Pipeline failed")

        with TestClient(app) as client:
            payload = {"dataset": data, "variables": ["X"], "method": "dynamics"}
            headers = {"X-User-Sub": "test", "X-User-Email": "t@e.com"}
            response = client.post("/analyze/causal", json=payload, headers=headers)
            assert response.status_code == 500
            assert "Analysis failed" in response.json()["detail"]


def test_simulate_virtual() -> None:
    # This test relies on the lifespan handler which loads a real (but small) model
    with TestClient(app) as client:
        initial_state = {"X": 0.0, "Y": 1.0}
        payload = {"initial_state": initial_state, "steps": 5, "intervention": {"X": 0.5}}
        headers = {"X-User-Sub": "test", "X-User-Email": "t@e.com"}
        response = client.post(
            "/simulate/virtual", json=payload, headers=headers
        )
        if response.status_code != 200:
            print(response.json())
        assert response.status_code == 200
        json_resp = response.json()
        assert "trajectory" in json_resp
        traj = json_resp["trajectory"]
        assert len(traj) == 6


def test_simulate_virtual_no_model() -> None:
    # Force models to be empty
    from coreason_inference.server import models

    with TestClient(app) as client:
        # Clear models AFTER client startup (lifespan has run)
        models.clear()

        payload = {"initial_state": {"X": 0.0}, "steps": 5}
        headers = {"X-User-Sub": "test", "X-User-Email": "t@e.com"}
        response = client.post("/simulate/virtual", json=payload, headers=headers)
        assert response.status_code == 503
        assert "Simulation model not initialized" in response.json()["detail"]


def test_simulate_virtual_exception() -> None:
    # To trigger general Exception
    with patch("coreason_inference.server.VirtualSimulator") as MockSim:
        instance = MockSim.return_value
        instance.simulate_trajectory.side_effect = Exception("Unexpected error")

        from coreason_inference.server import models
        models["default_dynamics"] = MagicMock()

        with TestClient(app) as client:
            payload = {"initial_state": {"X": 0.0}, "steps": 5}
            headers = {"X-User-Sub": "test", "X-User-Email": "t@e.com"}
            response = client.post("/simulate/virtual", json=payload, headers=headers)
            assert response.status_code == 500
            assert "Unexpected error" in response.json()["detail"]


def test_lifespan_exception() -> None:
    # Test that lifespan handles exceptions during startup
    with patch("coreason_inference.server.DynamicsEngine") as MockEngine:
        MockEngine.side_effect = Exception("Startup failed")

        import asyncio
        from coreason_inference.server import lifespan

        async def run_lifespan() -> None:
            async with lifespan(app):
                pass

        asyncio.run(run_lifespan())


# --- VirtualSimulator Unit Tests for Coverage ---
# Note: These test VirtualSimulator directly, not via server.
# `simulate_trajectory` was NOT updated to require context.
# `simulate_trial` WAS updated.

def test_simulate_trajectory_no_model() -> None:
    sim = VirtualSimulator()
    with pytest.raises(ValueError, match="No model provided"):
        sim.simulate_trajectory(initial_state={}, steps=10, model=None)
