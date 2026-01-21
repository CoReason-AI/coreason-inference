# Copyright (c) 2026 CoReason, Inc.
# Licensed under the Prosperity Public License 3.0.0

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from coreason_inference.analysis.estimator import METHOD_FOREST, CausalEstimator


class TestEstimatorAdvanced:
    @pytest.fixture
    def basic_data(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "T": [0, 1] * 5,
                "Y": np.random.randn(10),
                "X": np.random.randn(10),
                "patient_id": [str(i) for i in range(10)],
            }
        )

    def test_personalized_inference_missing_id_col(self, basic_data: pd.DataFrame) -> None:
        """Test ValueError when patient_id_col is missing from data but requested."""
        # Drop ID col
        df = basic_data.drop(columns=["patient_id"])
        estimator = CausalEstimator(df)

        # We need to simulate a successful Forest estimation that returns CATEs
        with patch("coreason_inference.analysis.estimator.CausalModel") as MockModel:
            mock_instance = MockModel.return_value
            mock_estimate = MagicMock()
            mock_estimate.value = 1.0
            # Mock the estimator.effect() call to return CATEs
            # Input data has 10 rows
            mock_estimate.estimator.effect.return_value = np.zeros((10, 1))
            mock_instance.estimate_effect.return_value = mock_estimate

            # Mock refutation to pass
            mock_instance.refute_estimate.return_value.refutation_result = {
                "is_statistically_significant": False,
                "p_value": 0.5,
            }

            # We request personalized inference
            with pytest.raises(ValueError, match="Patient ID column 'patient_id' not found"):
                estimator.estimate_effect(
                    "T", "Y", ["X"], method=METHOD_FOREST, target_patient_id="0", patient_id_col="patient_id"
                )

    def test_personalized_inference_id_not_found(self, basic_data: pd.DataFrame) -> None:
        """Test ValueError when specific patient ID is not found."""
        estimator = CausalEstimator(basic_data)

        with patch("coreason_inference.analysis.estimator.CausalModel") as MockModel:
            mock_instance = MockModel.return_value
            mock_estimate = MagicMock()
            mock_estimate.estimator.effect.return_value = np.zeros((10, 1))
            mock_instance.estimate_effect.return_value = mock_estimate

            mock_instance.refute_estimate.return_value.refutation_result = {
                "is_statistically_significant": False,
                "p_value": 0.5,
            }

            with pytest.raises(ValueError, match="Patient ID 'MISSING_ID' not found"):
                estimator.estimate_effect("T", "Y", ["X"], method=METHOD_FOREST, target_patient_id="MISSING_ID")

    def test_cate_extraction_failure(self, basic_data: pd.DataFrame) -> None:
        """Test that failure in CATE extraction returns None for cate_estimates."""
        estimator = CausalEstimator(basic_data)

        with patch("coreason_inference.analysis.estimator.CausalModel") as MockModel:
            mock_instance = MockModel.return_value
            mock_estimate = MagicMock()
            mock_estimate.value = 5.0
            mock_instance.estimate_effect.return_value = mock_estimate

            # Force exception in effect() call
            mock_estimate.estimator.effect.side_effect = Exception("EconML Error")

            # Mock refuter
            mock_instance.refute_estimate.return_value.refutation_result = {
                "is_statistically_significant": False,
                "p_value": 0.5,
            }

            result = estimator.estimate_effect("T", "Y", ["X"], method=METHOD_FOREST)

            assert result.cate_estimates is None
            # Main effect should still be present
            assert result.counterfactual_outcome == 5.0

    def test_confidence_interval_failure(self, basic_data: pd.DataFrame) -> None:
        """Test that failure in CI extraction returns defaults."""
        estimator = CausalEstimator(basic_data)

        with patch("coreason_inference.analysis.estimator.CausalModel") as MockModel:
            mock_instance = MockModel.return_value
            mock_estimate = MagicMock()
            mock_estimate.value = 1.0
            mock_instance.estimate_effect.return_value = mock_estimate

            # Force exception in get_confidence_intervals
            mock_estimate.get_confidence_intervals.side_effect = Exception("CI Error")

            mock_instance.refute_estimate.return_value.refutation_result = {
                "is_statistically_significant": False,
                "p_value": 0.5,
            }

            result = estimator.estimate_effect("T", "Y", ["X"])

            # Should default to (value, value)
            assert result.confidence_interval == (1.0, 1.0)

    def test_causal_forest_integration_mocked(self, basic_data: pd.DataFrame) -> None:
        """
        Verify that correct parameters are passed to CausalForestDML (via dowhy).
        """
        estimator = CausalEstimator(basic_data)

        with patch("coreason_inference.analysis.estimator.CausalModel") as MockModel:
            mock_instance = MockModel.return_value

            # Mock refutation to pass
            mock_instance.refute_estimate.return_value.refutation_result = {
                "is_statistically_significant": False,
                "p_value": 0.5,
            }

            estimator.estimate_effect("T", "Y", ["X"], method=METHOD_FOREST)

            # Verify estimate_effect called with correct method_params
            mock_instance.estimate_effect.assert_called_once()
            call_kwargs = mock_instance.estimate_effect.call_args[1]

            assert call_kwargs["method_name"] == "backdoor.econml.dml.CausalForestDML"
            assert call_kwargs["method_params"]["init_params"]["n_estimators"] == 100
