# Copyright (c) 2026 CoReason, Inc.
# Licensed under the Prosperity Public License 3.0.0

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

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
def mock_engine_with_data() -> InferenceEngine:
    engine = InferenceEngine()
    engine.augmented_data = pd.DataFrame(
        {"X1": [1, 2, 3, 4], "X2": [0.1, 0.2, 0.3, 0.4], "T": [0, 1, 0, 1], "Y": [1, 2, 1, 3]}
    )
    return engine


@pytest.fixture
def mock_causal_graph() -> CausalGraph:
    return CausalGraph(nodes=[], edges=[], loop_dynamics=[], stability_score=0.9)


@pytest.fixture
def mock_intervention_result() -> InterventionResult:
    return InterventionResult(
        patient_id="POPULATION_ATE",
        intervention="do(T)",
        counterfactual_outcome=0.5,
        confidence_interval=(0.4, 0.6),
        refutation_status=RefutationStatus.PASSED,
        cate_estimates=[0.1, 0.9, 0.2, 0.8],
    )


@pytest.fixture
def mock_optimization_output() -> OptimizationOutput:
    return OptimizationOutput(
        new_criteria=[ProtocolRule(feature="X1", operator=">", value=2.0, rationale="Test")],
        original_pos=0.5,
        optimized_pos=0.8,
        safety_flags=[],
    )


def test_analyze_heterogeneity_success(
    mock_engine_with_data: InferenceEngine, mock_intervention_result: InterventionResult
) -> None:
    """Test successful heterogeneity analysis and state update."""
    with patch("coreason_inference.engine.CausalEstimator") as MockEstimator:
        instance = MockEstimator.return_value
        instance.estimate_effect.return_value = mock_intervention_result

        result = mock_engine_with_data.analyze_heterogeneity("T", "Y", ["X1", "X2"])

        # Verify method called correctly
        instance.estimate_effect.assert_called_once_with(
            treatment="T", outcome="Y", confounders=["X1", "X2"], method="forest"
        )

        # Verify result
        assert result == mock_intervention_result

        # Verify state update
        assert mock_engine_with_data.cate_estimates is not None
        assert len(mock_engine_with_data.cate_estimates) == 4
        pd.testing.assert_series_equal(
            mock_engine_with_data.cate_estimates, pd.Series([0.1, 0.9, 0.2, 0.8], name="CATE_T_Y"), check_index=True
        )

        # Verify metadata update (New requirement)
        assert mock_engine_with_data._last_analysis_meta == {"treatment": "T", "outcome": "Y"}


def test_analyze_heterogeneity_no_data() -> None:
    """Test error when data is missing."""
    engine = InferenceEngine()
    # augmented_data is None

    with pytest.raises(ValueError, match="Data not available"):
        engine.analyze_heterogeneity("T", "Y", ["X"])


def test_analyze_heterogeneity_no_cate_returned(mock_engine_with_data: InferenceEngine) -> None:
    """Test handling when CausalEstimator returns no CATE estimates."""

    # Mock result with None cate_estimates
    mock_result = InterventionResult(
        patient_id="POPULATION_ATE",
        intervention="do(T)",
        counterfactual_outcome=0.5,
        confidence_interval=(0.4, 0.6),
        refutation_status=RefutationStatus.PASSED,
        cate_estimates=None,
    )

    with patch("coreason_inference.engine.CausalEstimator") as MockEstimator:
        instance = MockEstimator.return_value
        instance.estimate_effect.return_value = mock_result

        result = mock_engine_with_data.analyze_heterogeneity("T", "Y", ["X1"])

        assert result == mock_result
        assert mock_engine_with_data.cate_estimates is None


def test_induce_rules_success(
    mock_engine_with_data: InferenceEngine, mock_optimization_output: OptimizationOutput
) -> None:
    """Test successful rule induction."""

    # Setup state
    mock_engine_with_data.cate_estimates = pd.Series([0.1, 0.9, 0.2, 0.8], name="CATE_T_Y")

    # We patch the instance on the engine
    mock_engine_with_data.rule_inductor = MagicMock()
    mock_engine_with_data.rule_inductor.induce_rules_with_data.return_value = mock_optimization_output

    result = mock_engine_with_data.induce_rules(feature_cols=["X1"])

    # Verify calls
    # Features passed should be just X1
    args, _ = mock_engine_with_data.rule_inductor.fit.call_args
    features_arg = args[0]
    assert list(features_arg.columns) == ["X1"]

    assert result == mock_optimization_output


def test_induce_rules_auto_features(mock_engine_with_data: InferenceEngine) -> None:
    """Test rule induction with automatic feature selection."""

    mock_engine_with_data.cate_estimates = pd.Series([0.1, 0.9, 0.2, 0.8])
    mock_engine_with_data.rule_inductor = MagicMock()
    mock_engine_with_data.rule_inductor.induce_rules_with_data.return_value = OptimizationOutput(
        new_criteria=[], original_pos=0, optimized_pos=0, safety_flags=[]
    )

    mock_engine_with_data.induce_rules(feature_cols=None)

    # Should use all numeric columns (X1, X2, T, Y) since meta is empty
    args, _ = mock_engine_with_data.rule_inductor.fit.call_args
    features_arg = args[0]
    assert set(features_arg.columns) == {"X1", "X2", "T", "Y"}


def test_induce_rules_excludes_leakage(mock_engine_with_data: InferenceEngine) -> None:
    """Test that treatment and outcome are excluded from features during rule induction."""

    # Setup state
    mock_engine_with_data.cate_estimates = pd.Series([0.1, 0.9, 0.2, 0.8], name="CATE_T_Y")
    mock_engine_with_data._last_analysis_meta = {"treatment": "T", "outcome": "Y"}

    mock_engine_with_data.rule_inductor = MagicMock()
    mock_engine_with_data.rule_inductor.induce_rules_with_data.return_value = OptimizationOutput(
        new_criteria=[], original_pos=0, optimized_pos=0, safety_flags=[]
    )

    mock_engine_with_data.induce_rules(feature_cols=None)

    # Verify T and Y are excluded
    args, _ = mock_engine_with_data.rule_inductor.fit.call_args
    features_arg = args[0]
    assert "T" not in features_arg.columns
    assert "Y" not in features_arg.columns
    assert set(features_arg.columns) == {"X1", "X2"}


def test_induce_rules_no_cate() -> None:
    """Test error when CATEs are missing."""
    engine = InferenceEngine()
    engine.augmented_data = pd.DataFrame({"A": [1]})

    with pytest.raises(ValueError, match="No CATE estimates"):
        engine.induce_rules()


def test_induce_rules_data_missing_integrity() -> None:
    """Test error when augmented_data is missing but cate_estimates exists."""
    engine = InferenceEngine()
    # Manually inject CATE to bypass first check
    engine.cate_estimates = pd.Series([1.0, 2.0])
    # augmented_data is None

    with pytest.raises(ValueError, match="Data not available"):
        engine.induce_rules()


def test_run_virtual_trial_success(
    mock_engine_with_data: InferenceEngine,
    mock_causal_graph: CausalGraph,
    mock_optimization_output: OptimizationOutput,
    mock_intervention_result: InterventionResult,
) -> None:
    """Test successful virtual trial execution."""
    mock_engine_with_data.latent_miner.model = MagicMock()
    mock_engine_with_data.graph = mock_causal_graph

    mock_cohort = pd.DataFrame({"X1": [1, 2], "X2": [0.1, 0.2]})

    # Mock generate_synthetic_cohort (used via self.virtual_simulator instance on engine)
    mock_engine_with_data.virtual_simulator = MagicMock()
    mock_engine_with_data.virtual_simulator.generate_synthetic_cohort.return_value = mock_cohort
    mock_engine_with_data.virtual_simulator.scan_safety.return_value = ["Safety Warning"]
    mock_engine_with_data.virtual_simulator.simulate_trial.return_value = mock_intervention_result

    result = mock_engine_with_data.run_virtual_trial(
        optimization_result=mock_optimization_output,
        treatment="T",
        outcome="Y",
        confounders=["X1"],
        n_samples=50,
        adverse_outcomes=["Death"],
    )

    assert isinstance(result, VirtualTrialResult)
    assert result.cohort_size == 2
    assert result.safety_scan == ["Safety Warning"]
    assert result.simulation_result == mock_intervention_result

    # Verify calls
    mock_engine_with_data.virtual_simulator.generate_synthetic_cohort.assert_called_once_with(
        miner=mock_engine_with_data.latent_miner,
        n_samples=50,
        rules=mock_optimization_output.new_criteria,
    )
    mock_engine_with_data.virtual_simulator.scan_safety.assert_called_once_with(
        graph=mock_engine_with_data.graph,
        treatment="T",
        adverse_outcomes=["Death"],
    )
    mock_engine_with_data.virtual_simulator.simulate_trial.assert_called_once_with(
        cohort=mock_cohort,
        treatment="T",
        outcome="Y",
        confounders=["X1"],
    )


def test_run_virtual_trial_not_fitted() -> None:
    """Test error when model is not fitted."""
    engine = InferenceEngine()
    # latent_miner.model is None by default

    with pytest.raises(ValueError, match="Model not fitted"):
        engine.run_virtual_trial(optimization_result=MagicMock(), treatment="T", outcome="Y", confounders=[])


def test_run_virtual_trial_empty_cohort(mock_engine_with_data: InferenceEngine, mock_causal_graph: CausalGraph) -> None:
    """Test handling of empty cohort generation."""
    mock_engine_with_data.latent_miner.model = MagicMock()
    mock_engine_with_data.graph = mock_causal_graph

    mock_engine_with_data.virtual_simulator = MagicMock()
    mock_engine_with_data.virtual_simulator.generate_synthetic_cohort.return_value = pd.DataFrame()

    result = mock_engine_with_data.run_virtual_trial(
        optimization_result=MagicMock(), treatment="T", outcome="Y", confounders=[]
    )

    assert result.cohort_size == 0
    assert result.simulation_result is None


def test_run_virtual_trial_simulation_failure(
    mock_engine_with_data: InferenceEngine, mock_causal_graph: CausalGraph
) -> None:
    """Test graceful handling of simulation failure."""
    mock_engine_with_data.latent_miner.model = MagicMock()
    mock_engine_with_data.graph = mock_causal_graph

    mock_engine_with_data.virtual_simulator = MagicMock()
    mock_engine_with_data.virtual_simulator.generate_synthetic_cohort.return_value = pd.DataFrame({"A": [1]})
    mock_engine_with_data.virtual_simulator.simulate_trial.side_effect = Exception("Sim Error")

    result = mock_engine_with_data.run_virtual_trial(
        optimization_result=MagicMock(), treatment="T", outcome="Y", confounders=[]
    )

    assert result.cohort_size == 1
    assert result.simulation_result is None


def test_analyze_latent_index_mismatch(monkeypatch: pytest.MonkeyPatch, mock_causal_graph: CausalGraph) -> None:
    """
    Test edge case where LatentMiner returns data with mismatched index,
    checking if pd.concat produces NaNs (augmented_data integrity).
    """
    engine = InferenceEngine()
    data = pd.DataFrame({"X": [1, 2]}, index=[0, 1])

    # Mock components
    monkeypatch.setattr(engine.dynamics_engine, "fit", MagicMock())
    # Use real CausalGraph
    monkeypatch.setattr(engine.dynamics_engine, "discover_loops", MagicMock(return_value=mock_causal_graph))

    monkeypatch.setattr(engine.latent_miner, "fit", MagicMock())

    # Mock discover_latents to return MISMATCHED index
    mismatched_latents = pd.DataFrame({"Z": [0.1, 0.2]}, index=[2, 3])  # Different index than data
    monkeypatch.setattr(engine.latent_miner, "discover_latents", MagicMock(return_value=mismatched_latents))

    monkeypatch.setattr(engine.active_scientist, "fit", MagicMock())
    monkeypatch.setattr(engine.active_scientist, "propose_experiments", MagicMock(return_value=[]))

    # Run analyze
    # This should succeed but augmented_data will have NaNs due to concat
    result = engine.analyze(data, "time", ["X"])

    # Verify augmented_data
    # data has indices 0,1. latents has 2,3.
    # concat(axis=1) will result in indices 0,1,2,3 with NaNs.
    assert len(result.augmented_data) == 4
    assert result.augmented_data["X"].isna().sum() == 2  # Indices 2,3 will have NaN X
    assert result.augmented_data["Z"].isna().sum() == 2  # Indices 0,1 will have NaN Z


def test_analyze_heterogeneity_empty_confounders(mock_engine_with_data: InferenceEngine) -> None:
    """Test error when confounders list is empty for forest method."""
    with patch("coreason_inference.engine.CausalEstimator") as MockEstimator:
        instance = MockEstimator.return_value
        # EconML/CausalEstimator raises ValueError if X is empty
        instance.estimate_effect.side_effect = ValueError("Effect modifiers cannot be empty")

        with pytest.raises(ValueError, match="Effect modifiers cannot be empty"):
            mock_engine_with_data.analyze_heterogeneity("T", "Y", [])


def test_full_workflow_state_consistency(monkeypatch: pytest.MonkeyPatch, mock_causal_graph: CausalGraph) -> None:
    """
    Complex Scenario: Execute the full pipeline sequentially and verify state transitions.
    analyze -> analyze_heterogeneity -> induce_rules -> run_virtual_trial
    """
    engine = InferenceEngine()
    data = pd.DataFrame({"X": [1, 2, 3], "T": [0, 1, 0], "Y": [1, 2, 1], "time": [0, 1, 2]}, index=[0, 1, 2])

    # 1. Mock Analysis Components
    monkeypatch.setattr(engine.dynamics_engine, "fit", MagicMock())
    monkeypatch.setattr(engine.dynamics_engine, "discover_loops", MagicMock(return_value=mock_causal_graph))

    monkeypatch.setattr(engine.latent_miner, "fit", MagicMock())
    mock_latents = pd.DataFrame({"Z": [0.1, 0.2, 0.3]}, index=[0, 1, 2])
    monkeypatch.setattr(engine.latent_miner, "discover_latents", MagicMock(return_value=mock_latents))
    # Needed for virtual trial check
    engine.latent_miner.model = MagicMock()

    monkeypatch.setattr(engine.active_scientist, "fit", MagicMock())
    monkeypatch.setattr(engine.active_scientist, "propose_experiments", MagicMock(return_value=[]))

    # 2. Run Analyze
    engine.analyze(data, "time", ["X"])

    # Verify State after Analyze
    assert engine.graph == mock_causal_graph
    assert engine.latents is not None
    assert engine.augmented_data is not None
    assert "Z" in engine.augmented_data.columns
    assert len(engine.augmented_data) == 3

    # 3. Run Heterogeneity Analysis
    mock_cate_result = InterventionResult(
        patient_id="POPULATION_ATE",
        intervention="do(T)",
        counterfactual_outcome=0.5,
        confidence_interval=(0.4, 0.6),
        refutation_status=RefutationStatus.PASSED,
        cate_estimates=[0.1, 0.5, 0.9],
    )

    with patch("coreason_inference.engine.CausalEstimator") as MockEstimator:
        instance = MockEstimator.return_value
        instance.estimate_effect.return_value = mock_cate_result

        engine.analyze_heterogeneity("T", "Y", ["X", "Z"])

    # Verify State after Heterogeneity
    assert engine.cate_estimates is not None
    assert len(engine.cate_estimates) == 3
    assert engine._last_analysis_meta == {"treatment": "T", "outcome": "Y"}

    # 4. Run Rule Induction
    mock_opt_output = OptimizationOutput(
        new_criteria=[ProtocolRule(feature="X", operator=">", value=1.5, rationale="Test")],
        original_pos=0.3,
        optimized_pos=0.8,
        safety_flags=[],
    )
    # Patch rule inductor instance
    engine.rule_inductor = MagicMock()
    engine.rule_inductor.induce_rules_with_data.return_value = mock_opt_output

    opt_result = engine.induce_rules(feature_cols=["X", "Z"])

    assert opt_result == mock_opt_output

    # 5. Run Virtual Trial
    engine.virtual_simulator = MagicMock()
    engine.virtual_simulator.generate_synthetic_cohort.return_value = pd.DataFrame({"X": [2], "Z": [0.2]})
    engine.virtual_simulator.scan_safety.return_value = []
    mock_vt_sim_result = InterventionResult(
        patient_id="POPULATION_ATE",
        intervention="do(T)",
        counterfactual_outcome=0.8,
        confidence_interval=(0.7, 0.9),
        refutation_status=RefutationStatus.PASSED,
    )
    engine.virtual_simulator.simulate_trial.return_value = mock_vt_sim_result

    vt_result = engine.run_virtual_trial(
        optimization_result=opt_result,
        treatment="T",
        outcome="Y",
        confounders=["X", "Z"],
        n_samples=10,
    )

    assert vt_result.cohort_size == 1
    assert vt_result.simulation_result == mock_vt_sim_result
