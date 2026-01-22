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


def test_estimator_failure_handling_none_value(sample_data: pd.DataFrame) -> None:
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

        result = estimator.estimate_effect(treatment="T", outcome="Y", confounders=["X"])

        assert result.refutation_status == RefutationStatus.FAILED
        assert result.counterfactual_outcome is None
        assert result.patient_id == "ERROR"


def test_cate_extraction_failure_fallback(sample_data: pd.DataFrame) -> None:
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
        )

        # Should fall back to ATE
        assert result.patient_id == "POPULATION_ATE"
        assert result.counterfactual_outcome == 5.0
        assert result.cate_estimates is None
        assert result.refutation_status == RefutationStatus.PASSED


def test_refutation_exception_propagation(sample_data: pd.DataFrame) -> None:
    """
    Test edge case: Refutation raises an unhandled exception.
    It should propagate up to the caller (or be handled if we decide to wrap it).
    Currently, we expect it to raise.
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

        with pytest.raises(ValueError, match="Refuter crashed"):
            estimator.estimate_effect(treatment="T", outcome="Y", confounders=["X"])


def test_empty_confounders_forest_error(sample_data: pd.DataFrame) -> None:
    """
    Test that calling forest method with empty confounders/effect modifiers
    might cause issues in EconML, handled via exception or propagated.
    Actually, CausalForestDML requires at least one feature for splitting.
    """
    estimator = CausalEstimator(sample_data)

    # EconML CausalForestDML usually requires X != None.
    # But DoWhy handles the mapping.
    # If we pass empty list, DoWhy might pass None or empty DF.

    # We expect a crash or specific error from backend if we don't mock it.
    # But since we don't want to rely on external lib behavior in unit test,
    # we can check if our code allows it.

    # Our code: `effect_modifiers = confounders if method == METHOD_FOREST else []`
    # If confounders is [], effect_modifiers is [].
    # CausalModel init with effect_modifiers=[].

    # Let's verify that validation passes until the library call.
    # We won't mock here to see integration behavior if possible,
    # OR mock to verify inputs.

    with patch("coreason_inference.analysis.estimator.CausalModel") as mock_model_cls:
        mock_model = mock_model_cls.return_value
        mock_model.estimate_effect.side_effect = ValueError("EconML Error: X is empty")

        with pytest.raises(ValueError, match="EconML Error"):
            estimator.estimate_effect(
                treatment="T",
                outcome="Y",
                confounders=[],  # Empty
                method=METHOD_FOREST,
            )
