from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest

from coreason_inference.engine import InferenceEngine, InferenceEngineAsync, InferenceResult
from coreason_inference.schema import (
    CausalGraph,
    CausalNode,
    InterventionResult,
    OptimizationOutput,
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


@pytest.mark.asyncio
async def test_error_handling_in_async_analyze() -> None:
    """Test error handling blocks in analyze and run_virtual_trial."""
    engine = InferenceEngineAsync()
    data = pd.DataFrame({"time": [0, 1], "A": [1, 2]})

    # 1. Analyze Estimation Error
    # To hit line 202 (logger.error(f"Estimation failed during pipeline: {e}"))
    # We mock dynamics/latent/active to pass quickly
    engine.dynamics_engine = MagicMock()
    engine.dynamics_engine.discover_loops.return_value = CausalGraph(
        nodes=[], edges=[], loop_dynamics=[], stability_score=0
    )
    engine.latent_miner = MagicMock()
    engine.latent_miner.discover_latents.return_value = pd.DataFrame()
    engine.active_scientist = MagicMock()

    # We need to simulate estimator failure
    # Use a concrete mock class to ensure side_effect works across threads in run_sync
    class BrokenEstimator:
        def __init__(self, df: pd.DataFrame) -> None:
            pass

        def estimate_effect(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("Boom")

    with patch("coreason_inference.engine.CausalEstimator", side_effect=BrokenEstimator):
        # This should catch exception and log error, not raise
        await engine.analyze(data, "time", ["A"], estimate_effect_for=("A", "A"))


@pytest.mark.asyncio
async def test_analyze_missing_estimator() -> None:
    """Test coverage for line 193: raise ValueError('Estimator not initialized')."""
    engine = InferenceEngineAsync()
    engine.dynamics_engine = MagicMock()
    engine.dynamics_engine.discover_loops.return_value = CausalGraph(
        nodes=[], edges=[], loop_dynamics=[], stability_score=0
    )
    engine.latent_miner = MagicMock()
    engine.latent_miner.discover_latents.return_value = pd.DataFrame()
    engine.active_scientist = MagicMock()

    data = pd.DataFrame({"time": [0, 1], "A": [1, 2]})

    # We mock _estimator property to return None
    # Since _estimator is a property, we patch it on the class or instance.
    # Note: engine.analyze sets self.estimator = self._estimator.
    # If self._estimator returns None, self.estimator is None.
    # But self._estimator type hint says CausalEstimator. We force it for test.
    with patch.object(InferenceEngineAsync, "_estimator", new_callable=PropertyMock) as mock_prop:
        mock_prop.return_value = None

        # Should log error about "Estimator not initialized"
        await engine.analyze(data, "time", ["A"], estimate_effect_for=("A", "A"))


@pytest.mark.asyncio
async def test_error_handling_in_virtual_trial() -> None:
    """Test error handling in run_virtual_trial."""
    engine = InferenceEngineAsync()
    engine.latent_miner.model = MagicMock()
    engine.graph = MagicMock()

    # To hit line 546 (logger.error(f"Virtual trial simulation failed: {e}"))
    # Use patch.object to avoid Mypy "Cannot assign to a method" errors
    with (
        patch.object(engine.virtual_simulator, "generate_synthetic_cohort", return_value=pd.DataFrame({"A": [1]})),
        patch.object(engine.virtual_simulator, "scan_safety", return_value=[]),
        patch.object(engine.virtual_simulator, "simulate_trial", side_effect=Exception("Sim Boom")),
    ):
        result = await engine.run_virtual_trial(
            OptimizationOutput(new_criteria=[], original_pos=0, optimized_pos=0), "T", "Y", []
        )

        assert result.simulation_result is None
        assert result.cohort_size == 1


def test_sync_facade_setters() -> None:
    """Test setters in Sync Facade to cover lines 459, 462, 470, 486, 514, 518."""
    engine = InferenceEngine()

    # 459: dynamics_engine setter
    engine.dynamics_engine = MagicMock()
    # 462: latent_miner setter (missed?)
    # 470: active_scientist setter (missed?)
    # 486: augmented_data setter (missed?)
    # 514: _last_analysis_meta setter
    engine._last_analysis_meta = {"key": "value"}
    assert engine._last_analysis_meta == {"key": "value"}

    # 518: _latent_features setter
    engine._latent_features = ["F1"]
    assert engine._latent_features == ["F1"]

    # Cover others explicitly just in case
    engine.latent_miner = MagicMock()
    engine.active_scientist = MagicMock()
    engine.virtual_simulator = MagicMock()
    engine.rule_inductor = MagicMock()
    engine.graph = None
    engine.latents = None
    engine.augmented_data = None
    engine.cate_estimates = None
