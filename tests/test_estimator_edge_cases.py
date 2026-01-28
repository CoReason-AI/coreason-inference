# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from coreason_inference.analysis.estimator import METHOD_FOREST, CausalEstimator
from coreason_inference.schema import RefutationStatus


@pytest.fixture
def sample_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": [f"P{i}" for i in range(40)],
            "T": [0, 1, 0, 1] * 10,
            "Y": [1, 2, 1, 3] * 10,
            "X": [0.1, 0.2, 0.3, 0.4] * 10,
        }
    )


def test_estimator_failure_handling_none_value(sample_data: pd.DataFrame, mock_user_context) -> None:
    """
    Test that CausalEstimator gracefully handles cases where DoWhy returns None
    for the estimated effect (e.g., due to missing data/confounders).
    """
    estimator = CausalEstimator(sample_data)

    with patch("coreason_inference.analysis.estimator.CausalModel") as mock_model_cls:
        mock_model = mock_model_cls.return_value
        mock_model.identify_effect.return_value = MagicMock()

        # Mock estimate_effect returning None value
        mock_estimate = MagicMock()
        mock_estimate.value = None
        mock_model.estimate_effect.return_value = mock_estimate

        result = estimator.estimate_effect(treatment="T", outcome="Y", confounders=["X"], context=mock_user_context)

        assert result.refutation_status == RefutationStatus.FAILED
        assert result.counterfactual_outcome is None
        assert result.patient_id == "ERROR"


def test_cate_extraction_failure_fallback(sample_data: pd.DataFrame, mock_user_context) -> None:
    """
    Test edge case: User requests personalized inference (target_patient_id),
    but CATE extraction fails (e.g. underlying forest fails).
    Should fallback to Population ATE gracefully.
    """
    estimator = CausalEstimator(sample_data)

    with patch("coreason_inference.analysis.estimator.CausalModel") as mock_model_cls:
        mock_model = mock_model_cls.return_value
        mock_model.identify_effect.return_value = MagicMock()

        mock_estimate = MagicMock()
        mock_estimate.value = 5.0
        # Mocking the estimator inside estimate object to raise Exception when calling .effect()
        mock_estimate.estimator.effect.side_effect = Exception("Forest failed")
        mock_model.estimate_effect.return_value = mock_estimate

        # Mock Refutation Success
        mock_refutation = MagicMock()
        mock_refutation.refutation_result = {"is_statistically_significant": False, "p_value": 0.5}
        mock_model.refute_estimate.return_value = mock_refutation

        # Request personalized inference
        result = estimator.estimate_effect(
            treatment="T", outcome="Y", confounders=["X"], method=METHOD_FOREST, target_patient_id="P0"
        , context=mock_user_context)

        # Should fall back to ATE
        assert result.patient_id == "POPULATION_ATE"
        assert result.counterfactual_outcome == 5.0
        assert result.cate_estimates is None
        assert result.refutation_status == RefutationStatus.PASSED


def test_refutation_exception_handling(sample_data: pd.DataFrame, mock_user_context) -> None:
    """
    Test edge case: Refutation raises an unhandled exception.
    The updated code handles this by logging warning and setting status to FAILED.
    """
    estimator = CausalEstimator(sample_data)

    with patch("coreason_inference.analysis.estimator.CausalModel") as mock_model_cls:
        mock_model = mock_model_cls.return_value
        mock_model.identify_effect.return_value = MagicMock()

        mock_estimate = MagicMock()
        mock_estimate.value = 5.0
        mock_model.estimate_effect.return_value = mock_estimate

        # Mock Refutation raising Exception
        mock_model.refute_estimate.side_effect = ValueError("Refuter crashed")

        result = estimator.estimate_effect(treatment="T", outcome="Y", confounders=["X"], context=mock_user_context)

        # Should NOT crash, but return FAILED
        assert result.refutation_status == RefutationStatus.FAILED
        assert result.counterfactual_outcome is None


def test_estimation_exception_propagation(sample_data: pd.DataFrame, mock_user_context) -> None:
    """
    Test that if estimation (model fitting) fails, it raises the exception.
    """
    estimator = CausalEstimator(sample_data)

    with patch("coreason_inference.analysis.estimator.CausalModel") as mock_model_cls:
        mock_model = mock_model_cls.return_value
        mock_model.identify_effect.return_value = MagicMock()

        # Mock estimate_effect raising Exception
        mock_model.estimate_effect.side_effect = RuntimeError("Optimization failed")

        with pytest.raises(RuntimeError, match="Optimization failed"):
            estimator.estimate_effect(treatment="T", outcome="Y", confounders=["X"], context=mock_user_context)


def test_confidence_interval_fallback(sample_data: pd.DataFrame, mock_user_context) -> None:
    """
    Test that if confidence interval extraction fails, defaults are returned.
    """
    estimator = CausalEstimator(sample_data)

    with patch("coreason_inference.analysis.estimator.CausalModel") as mock_model_cls:
        mock_model = mock_model_cls.return_value

        mock_estimate = MagicMock()
        mock_estimate.value = 5.0
        # Mock CI raising exception
        mock_estimate.get_confidence_intervals.side_effect = Exception("CI failed")
        mock_model.estimate_effect.return_value = mock_estimate

        mock_refutation = MagicMock()
        mock_refutation.refutation_result = {"is_statistically_significant": False, "p_value": 0.5}
        mock_model.refute_estimate.return_value = mock_refutation

        result = estimator.estimate_effect(treatment="T", outcome="Y", confounders=["X"], context=mock_user_context)

        # Default value matches effect value (or 0.0? Implementation uses default_value=effect_value)
        # Check implementation: _extract_confidence_intervals(estimate, effect_value) -> returns default_value
        assert result.confidence_interval == (5.0, 5.0)


def test_empty_confounders_forest_error(sample_data: pd.DataFrame, mock_user_context) -> None:
    """
    Test that calling forest method with empty confounders/effect modifiers
    might cause issues in EconML, handled via exception or propagated.
    """
    estimator = CausalEstimator(sample_data)

    with patch("coreason_inference.analysis.estimator.CausalModel") as mock_model_cls:
        mock_model = mock_model_cls.return_value
        mock_model.estimate_effect.side_effect = ValueError("EconML Error: X is empty")

        with pytest.raises(ValueError, match="EconML Error"):
            estimator.estimate_effect(treatment="T", outcome="Y", confounders=[], method=METHOD_FOREST, context=mock_user_context)
