# Copyright (c) 2026 CoReason, Inc.
# Licensed under the Prosperity Public License 3.0.0

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from coreason_inference.engine import InferenceEngine
from coreason_inference.schema import InterventionResult, OptimizationOutput, ProtocolRule, RefutationStatus


@pytest.fixture
def mock_engine_with_data() -> InferenceEngine:
    engine = InferenceEngine()
    engine.augmented_data = pd.DataFrame(
        {"X1": [1, 2, 3, 4], "X2": [0.1, 0.2, 0.3, 0.4], "T": [0, 1, 0, 1], "Y": [1, 2, 1, 3]}
    )
    return engine


def test_analyze_heterogeneity_success(mock_engine_with_data: InferenceEngine) -> None:
    """Test successful heterogeneity analysis and state update."""

    # Mock CausalEstimator to return CATEs
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

        result = mock_engine_with_data.analyze_heterogeneity("T", "Y", ["X1", "X2"])

        # Verify method called correctly
        instance.estimate_effect.assert_called_once_with(
            treatment="T", outcome="Y", confounders=["X1", "X2"], method="forest"
        )

        # Verify result
        assert result == mock_result

        # Verify state update
        assert mock_engine_with_data.cate_estimates is not None
        assert len(mock_engine_with_data.cate_estimates) == 4
        pd.testing.assert_series_equal(
            mock_engine_with_data.cate_estimates, pd.Series([0.1, 0.9, 0.2, 0.8], name="CATE_T_Y"), check_index=True
        )


def test_analyze_heterogeneity_no_data() -> None:
    """Test error when data is missing."""
    engine = InferenceEngine()
    # augmented_data is None

    with pytest.raises(ValueError, match="Data not available"):
        engine.analyze_heterogeneity("T", "Y", ["X"])


def test_induce_rules_success(mock_engine_with_data: InferenceEngine) -> None:
    """Test successful rule induction."""

    # Setup state
    mock_engine_with_data.cate_estimates = pd.Series([0.1, 0.9, 0.2, 0.8], name="CATE_T_Y")

    # Mock RuleInductor
    mock_optimization_output = OptimizationOutput(
        new_criteria=[ProtocolRule(feature="X1", operator=">", value=2.0, rationale="Test")],
        original_pos=0.5,
        optimized_pos=0.8,
        safety_flags=[],
    )

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

    # Should use all numeric columns (X1, X2, T, Y)
    args, _ = mock_engine_with_data.rule_inductor.fit.call_args
    features_arg = args[0]
    assert set(features_arg.columns) == {"X1", "X2", "T", "Y"}


def test_induce_rules_no_cate() -> None:
    """Test error when CATEs are missing."""
    engine = InferenceEngine()
    engine.augmented_data = pd.DataFrame({"A": [1]})

    with pytest.raises(ValueError, match="No CATE estimates"):
        engine.induce_rules()
