from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from coreason_inference.analysis.estimator import CausalEstimator


def test_estimator_cate_extraction_failure(mock_user_context) -> None:
    """
    Test that cate_estimates are None if extraction fails.
    """
    data = pd.DataFrame({"T": [0, 1] * 50, "Y": [0, 1] * 50, "X": [0] * 100})
    estimator = CausalEstimator(data)

    # Mock the estimator in the estimate object to raise exception during effect()
    with patch("dowhy.CausalModel.estimate_effect") as mock_estimate:
        mock_result = MagicMock()
        mock_result.value = 1.0
        # Mock estimator
        mock_estimator = MagicMock()
        mock_estimator.effect.side_effect = Exception("Mock extraction failure")
        mock_result.estimator = mock_estimator
        mock_result.get_confidence_intervals.return_value = (0.9, 1.1)

        mock_estimate.return_value = mock_result

        # We also need to mock identify_effect and refute_estimate to avoid real calls
        with patch("dowhy.CausalModel.identify_effect"), patch("dowhy.CausalModel.refute_estimate") as mock_refute:
            mock_refute.return_value.refutation_result = {"is_statistically_significant": False, "p_value": 0.5}

            result = estimator.estimate_effect("T", "Y", ["X"], method="forest", context=mock_user_context)

            assert result.cate_estimates is None
            assert result.counterfactual_outcome == 1.0


def test_estimator_confidence_interval_array_handling(mock_user_context) -> None:
    """
    Test handling of array-like confidence intervals (taking mean).
    """
    data = pd.DataFrame({"T": [0, 1] * 50, "Y": [0, 1] * 50, "X": [0] * 100})
    estimator = CausalEstimator(data)

    with patch("dowhy.CausalModel.estimate_effect") as mock_estimate:
        mock_result = MagicMock()
        mock_result.value = 1.0
        # return array for CI
        mock_result.get_confidence_intervals.return_value = (np.array([0.8, 0.9]), np.array([1.1, 1.2]))

        mock_estimate.return_value = mock_result

        with patch("dowhy.CausalModel.identify_effect"), patch("dowhy.CausalModel.refute_estimate") as mock_refute:
            mock_refute.return_value.refutation_result = {"is_statistically_significant": False, "p_value": 0.5}

            result = estimator.estimate_effect("T", "Y", ["X"], context=mock_user_context)

            # Should be mean: low=(0.8+0.9)/2=0.85, high=(1.1+1.2)/2=1.15
            assert pytest.approx(result.confidence_interval[0]) == 0.85
            assert pytest.approx(result.confidence_interval[1]) == 1.15


def test_estimator_confidence_interval_exception(mock_user_context) -> None:
    """
    Test fallback when CI extraction raises exception.
    """
    data = pd.DataFrame({"T": [0, 1] * 50, "Y": [0, 1] * 50, "X": [0] * 100})
    estimator = CausalEstimator(data)

    with patch("dowhy.CausalModel.estimate_effect") as mock_estimate:
        mock_result = MagicMock()
        mock_result.value = 1.0

        # Mocking numpy.mean to fail to simulate exception in CI processing block
        mock_result.get_confidence_intervals.return_value = ([0.9], [1.1])
        mock_estimate.return_value = mock_result

        with (
            patch("dowhy.CausalModel.identify_effect"),
            patch("dowhy.CausalModel.refute_estimate") as mock_refute,
            patch("numpy.mean", side_effect=Exception("Mock math error")),
        ):
            mock_refute.return_value.refutation_result = {"is_statistically_significant": False, "p_value": 0.5}

            result = estimator.estimate_effect("T", "Y", ["X"], context=mock_user_context)

            # Should fallback to (effect, effect)
            assert result.confidence_interval == (1.0, 1.0)
