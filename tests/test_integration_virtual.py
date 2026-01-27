# Copyright (C) 2026 CoReason
# Licensed under the Prosperity Public License 3.0.0

from typing import cast
from unittest.mock import MagicMock

import pandas as pd
import pytest
from coreason_identity.models import UserContext

from coreason_inference.engine import InferenceEngine
from coreason_inference.schema import (
    CausalGraph,
    InterventionResult,
    OptimizationOutput,
    ProtocolRule,
    RefutationStatus,
    VirtualTrialResult,
)


@pytest.fixture
def mock_engine() -> InferenceEngine:
    engine = InferenceEngine()
    engine.latent_miner = MagicMock()
    engine.virtual_simulator = MagicMock()
    engine.graph = MagicMock(spec=CausalGraph)
    # Mock latent miner behavior to look "fitted"
    engine.latent_miner.model = MagicMock()
    return engine


def test_run_virtual_trial_success(mock_engine: InferenceEngine) -> None:
    """Test full flow of run_virtual_trial."""

    # Setup inputs
    optimization_result = OptimizationOutput(
        new_criteria=[ProtocolRule(feature="X", operator=">", value=0.5, rationale="Test")],
        original_pos=0.3,
        optimized_pos=0.8,
        safety_flags=[],
    )

    # Setup mocks
    mock_cohort = pd.DataFrame({"X": [1.0, 0.8], "Y": [0, 1], "T": [0, 1]})

    # Cast to MagicMock for MyPy
    mock_generate = cast(MagicMock, mock_engine.virtual_simulator.generate_synthetic_cohort)
    mock_scan = cast(MagicMock, mock_engine.virtual_simulator.scan_safety)
    mock_simulate = cast(MagicMock, mock_engine.virtual_simulator.simulate_trial)

    mock_generate.return_value = mock_cohort
    mock_scan.return_value = ["Risk A"]

    mock_sim_result = InterventionResult(
        patient_id="POPULATION_ATE",
        intervention="do(T)",
        counterfactual_outcome=0.9,
        confidence_interval=(0.8, 1.0),
        refutation_status=RefutationStatus.PASSED,
    )
    mock_simulate.return_value = mock_sim_result

    # Execute
    user = UserContext(user_id="test_user", email="test@example.com", claims={"tenant_id": "test_tenant"})
    result = mock_engine.run_virtual_trial(
        optimization_result=optimization_result,
        treatment="T",
        outcome="Y",
        confounders=["X"],
        n_samples=100,
        adverse_outcomes=["Adverse1"],
        user_context=user,
    )

    # Assertions
    mock_generate.assert_called_once_with(
        miner=mock_engine.latent_miner, n_samples=100, rules=optimization_result.new_criteria
    )

    mock_scan.assert_called_once_with(graph=mock_engine.graph, treatment="T", adverse_outcomes=["Adverse1"])

    mock_simulate.assert_called_once_with(cohort=mock_cohort, treatment="T", outcome="Y", confounders=["X"])

    assert isinstance(result, VirtualTrialResult)
    assert result.cohort_size == 2
    assert result.safety_scan == ["Risk A"]
    assert result.simulation_result == mock_sim_result


def test_run_virtual_trial_not_fitted(mock_engine: InferenceEngine) -> None:
    """Test that it raises error if miner not fitted."""
    mock_engine.latent_miner.model = None

    user = UserContext(user_id="test_user", email="test@example.com", claims={"tenant_id": "test_tenant"})
    with pytest.raises(ValueError, match="Model not fitted"):
        mock_engine.run_virtual_trial(
            optimization_result=MagicMock(),
            treatment="T",
            outcome="Y",
            confounders=[],
            user_context=user,
        )


def test_run_virtual_trial_empty_cohort(mock_engine: InferenceEngine) -> None:
    """Test handling of empty cohort generation."""
    mock_generate = cast(MagicMock, mock_engine.virtual_simulator.generate_synthetic_cohort)
    mock_simulate = cast(MagicMock, mock_engine.virtual_simulator.simulate_trial)

    mock_generate.return_value = pd.DataFrame()

    user = UserContext(user_id="test_user", email="test@example.com", claims={"tenant_id": "test_tenant"})
    result = mock_engine.run_virtual_trial(
        optimization_result=MagicMock(),
        treatment="T",
        outcome="Y",
        confounders=[],
        user_context=user,
    )

    assert isinstance(result, VirtualTrialResult)
    assert result.cohort_size == 0
    assert result.simulation_result is None
    # Should not attempt simulation
    mock_simulate.assert_not_called()


def test_run_virtual_trial_simulation_failure(mock_engine: InferenceEngine) -> None:
    """Test handling of simulation failure within engine."""
    mock_generate = cast(MagicMock, mock_engine.virtual_simulator.generate_synthetic_cohort)
    mock_simulate = cast(MagicMock, mock_engine.virtual_simulator.simulate_trial)

    mock_generate.return_value = pd.DataFrame({"A": [1]})
    mock_simulate.side_effect = Exception("Sim Error")

    user = UserContext(user_id="test_user", email="test@example.com", claims={"tenant_id": "test_tenant"})
    result = mock_engine.run_virtual_trial(
        optimization_result=MagicMock(),
        treatment="T",
        outcome="Y",
        confounders=[],
        user_context=user,
    )

    assert isinstance(result, VirtualTrialResult)
    assert result.simulation_result is None
