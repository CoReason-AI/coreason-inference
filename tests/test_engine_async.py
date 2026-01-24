from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from coreason_inference.engine import InferenceEngineAsync, InferenceResult
from coreason_inference.schema import (
    CausalGraph,
    CausalNode,
    InterventionResult,
    RefutationStatus,
)


@pytest.fixture
async def async_engine() -> InferenceEngineAsync:
    engine = InferenceEngineAsync()
    engine.augmented_data = pd.DataFrame(
        {"X1": [1, 2, 3, 4], "X2": [0.1, 0.2, 0.3, 0.4], "T": [0, 1, 0, 1], "Y": [1, 2, 1, 3]}
    )
    return engine


@pytest.mark.asyncio
async def test_analyze_heterogeneity_async(async_engine: InferenceEngineAsync) -> None:
    mock_result = InterventionResult(
        patient_id="POPULATION_ATE",
        intervention="do(T)",
        counterfactual_outcome=0.5,
        confidence_interval=(0.4, 0.6),
        refutation_status=RefutationStatus.PASSED,
        cate_estimates=[0.1, 0.9, 0.2, 0.8],
    )

    with patch("coreason_inference.engine.CausalEstimator") as MockEstimator:
        instance = MockEstimator.return_value
        instance.estimate_effect.return_value = mock_result

        result = await async_engine.analyze_heterogeneity("T", "Y", ["X1", "X2"])

        assert result == mock_result
        assert async_engine.cate_estimates is not None
        assert len(async_engine.cate_estimates) == 4


@pytest.mark.asyncio
async def test_context_manager() -> None:
    async with InferenceEngineAsync() as engine:
        assert engine._client is not None
        assert not engine._client.is_closed
    assert engine._client.is_closed


@pytest.mark.asyncio
async def test_analyze_pipeline_mocked() -> None:
    """Test the full analyze pipeline with mocked sub-engines to avoid heavy computation."""
    # Setup mocks
    dynamics_engine = MagicMock()
    latent_miner = MagicMock()
    active_scientist = MagicMock()

    # DynamicsEngine mocks
    dynamics_engine.discover_loops.return_value = CausalGraph(
        nodes=[
            CausalNode(id="A", codex_concept_id=1, is_latent=False),
            CausalNode(id="B", codex_concept_id=2, is_latent=False),
        ],
        edges=[("A", "B")],
        loop_dynamics=[],
        stability_score=0.9,
    )
    dynamics_engine.fit.return_value = None

    # LatentMiner mocks
    latent_miner.fit.return_value = None
    latent_miner.discover_latents.return_value = pd.DataFrame({"Z": [0.1, 0.2]})

    # ActiveScientist mocks
    active_scientist.fit.return_value = None
    active_scientist.propose_experiments.return_value = []

    # Initialize engine with mocks
    engine = InferenceEngineAsync(
        dynamics_engine=dynamics_engine, latent_miner=latent_miner, active_scientist=active_scientist
    )

    data = pd.DataFrame({"time": [0, 1], "A": [1, 2], "B": [3, 4]})

    # Run analyze
    # We also test the optional estimation part
    with patch("coreason_inference.engine.CausalEstimator") as MockEstimator:
        instance = MockEstimator.return_value
        instance.estimate_effect.return_value = InterventionResult(
            patient_id="POPULATION_ATE",
            intervention="do(A)",
            counterfactual_outcome=0.5,
            confidence_interval=(0.4, 0.6),
            refutation_status=RefutationStatus.PASSED,
        )

        result = await engine.analyze(
            data=data, time_col="time", variable_cols=["A", "B"], estimate_effect_for=("A", "B")
        )

        # Assertions
        assert isinstance(result, InferenceResult)
        assert len(result.graph.edges) == 1
        assert "Z" in result.latents.columns
        assert "Z" in result.augmented_data.columns

        # Verify calls (ensure run_sync called the underlying methods)
        dynamics_engine.fit.assert_called_once()
        dynamics_engine.discover_loops.assert_called_once()
        latent_miner.fit.assert_called_once()
        latent_miner.discover_latents.assert_called_once()
        active_scientist.fit.assert_called_once()
        active_scientist.propose_experiments.assert_called_once()

        # Verify estimator call
        instance.estimate_effect.assert_called_once()
