# Copyright (c) 2026 CoReason, Inc.
# Licensed under the Prosperity Public License 3.0.0

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from coreason_inference.engine import InferenceEngine
from coreason_inference.schema import InterventionResult, OptimizationOutput, RefutationStatus


@pytest.fixture
def mock_engine_complex() -> InferenceEngine:
    engine = InferenceEngine()
    # Create data with some edge case potential
    # X_cat is categorical (string)
    # X_nan has missing values
    engine.augmented_data = pd.DataFrame(
        {
            "X_num": [1.0, 2.0, 3.0, 4.0, 5.0],
            "X_cat": ["A", "B", "A", "B", "A"],
            "X_nan": [1.0, np.nan, 3.0, 4.0, 5.0],
            "T": [0, 1, 0, 1, 0],
            "Y": [1, 2, 1, 2, 1],
        }
    )
    return engine


def test_analyze_heterogeneity_empty_confounders(mock_engine_complex: InferenceEngine, mock_user_context) -> None:
    """Test behavior when confounders list is empty."""
    # CausalForest usually requires features to split on.
    # We mock CausalEstimator to simulate the failure we expect from the library.

    with patch("coreason_inference.engine.CausalEstimator") as MockEstimator:
        instance = MockEstimator.return_value
        # Simulate validation error from underlying estimator
        instance.estimate_effect.side_effect = ValueError("Need at least one confounder")

        with pytest.raises(ValueError, match="Need at least one confounder"):
            mock_engine_complex.analyze_heterogeneity("T", "Y", [], context=mock_user_context)


def test_induce_rules_constant_cate(mock_engine_complex: InferenceEngine, mock_user_context) -> None:
    """Test rule induction when CATE is constant (no heterogeneity)."""
    # All CATEs are identical
    mock_engine_complex.cate_estimates = pd.Series([0.5] * 5, name="CATE_T_Y")

    # We use the REAL RuleInductor here to verify its logic with constant target.
    # The engine initializes rule_inductor in __init__.

    # With constant target, the tree shouldn't find any splits that reduce impurity significantly.
    output = mock_engine_complex.induce_rules(feature_cols=["X_num"], context=mock_user_context)

    assert isinstance(output, OptimizationOutput)
    # Expect no new criteria
    assert len(output.new_criteria) == 0


def test_induce_rules_non_numeric_user_features(mock_engine_complex: InferenceEngine, mock_user_context) -> None:
    """Test error when user explicitly passes a non-numeric column."""
    mock_engine_complex.cate_estimates = pd.Series([0.1, 0.9, 0.2, 0.8, 0.5])

    # Passing "X_cat" which is strings
    # sklearn DecisionTree will raise ValueError
    with pytest.raises(ValueError, match="could not convert string to float"):
        mock_engine_complex.induce_rules(feature_cols=["X_cat"], context=mock_user_context)


def test_induce_rules_nan_features(mock_engine_complex: InferenceEngine, mock_user_context) -> None:
    """Test behavior with NaN values in features."""
    mock_engine_complex.cate_estimates = pd.Series([0.1, 0.9, 0.2, 0.8, 0.5])

    # Sklearn 1.6+ supports missing values in Decision Trees.
    # It should not raise, but produce an output.
    output = mock_engine_complex.induce_rules(feature_cols=["X_nan"], context=mock_user_context)

    assert isinstance(output, OptimizationOutput)
    # We don't check rules content, just that it ran successfully.


def test_optimization_state_overwrite(mock_engine_complex: InferenceEngine, mock_user_context) -> None:
    """Test that subsequent calls overwrite the stored CATE estimates."""

    # First Analysis
    mock_result_1 = InterventionResult(
        patient_id="POP",
        intervention="do(T)",
        counterfactual_outcome=0.5,
        confidence_interval=(0, 1),
        refutation_status=RefutationStatus.PASSED,
        cate_estimates=[0.1, 0.1, 0.1, 0.1, 0.1],
    )

    with patch("coreason_inference.engine.CausalEstimator") as MockEstimator:
        instance = MockEstimator.return_value
        instance.estimate_effect.return_value = mock_result_1

        mock_engine_complex.analyze_heterogeneity("T", "Y", ["X_num"], context=mock_user_context)
        assert mock_engine_complex.cate_estimates is not None
        assert mock_engine_complex.cate_estimates.name == "CATE_T_Y"
        assert mock_engine_complex.cate_estimates.mean() == 0.1

        # Second Analysis (different Outcome)
        mock_result_2 = InterventionResult(
            patient_id="POP",
            intervention="do(T)",
            counterfactual_outcome=0.8,
            confidence_interval=(0, 1),
            refutation_status=RefutationStatus.PASSED,
            cate_estimates=[0.9, 0.9, 0.9, 0.9, 0.9],
        )
        instance.estimate_effect.return_value = mock_result_2

        mock_engine_complex.analyze_heterogeneity("T", "Y2", ["X_num"], context=mock_user_context)
        assert mock_engine_complex.cate_estimates is not None
        assert mock_engine_complex.cate_estimates.name == "CATE_T_Y2"
        assert mock_engine_complex.cate_estimates.mean() == 0.9
