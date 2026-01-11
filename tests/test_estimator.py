# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from coreason_inference.analysis.estimator import CausalEstimator


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """
    Generates synthetic data with a known causal effect.
    Model: Y = 2 * T + 0.5 * C + Noise
    True Effect of T on Y is 2.
    """
    np.random.seed(42)
    n = 500
    # Confounder
    C = np.random.normal(0, 1, n)
    # Treatment (depends on C)
    T = np.random.normal(C, 1, n)
    # Outcome (depends on T and C)
    Y = 2 * T + 0.5 * C + np.random.normal(0, 0.1, n)

    return pd.DataFrame({"treatment": T, "outcome": Y, "confounder": C})


def test_estimate_effect_recovery(synthetic_data: pd.DataFrame) -> None:
    """
    Test that the estimator recovers the true causal effect (approx 2.0).
    """
    estimator = CausalEstimator(synthetic_data)
    result = estimator.estimate_effect(treatment="treatment", outcome="outcome", confounders=["confounder"])

    # Check if effect is close to 2.0
    # Allow some tolerance due to noise and finite sample size
    assert result.counterfactual_outcome == pytest.approx(2.0, abs=0.2)
    assert result.patient_id == "POPULATION_ATE"
    assert "do(treatment)" in result.intervention


def test_refutation_passed(synthetic_data: pd.DataFrame) -> None:
    """
    Test that the placebo refuter passes for valid data.
    """
    estimator = CausalEstimator(synthetic_data)
    result = estimator.estimate_effect(treatment="treatment", outcome="outcome", confounders=["confounder"])

    # Placebo test should usually pass (i.e., find no effect for placebo)
    # Status "PASSED" means the placebo effect was insignificant.
    assert result.refutation_status == "PASSED"


def test_missing_columns(synthetic_data: pd.DataFrame) -> None:
    """
    Test that the estimator raises an error if columns are missing.
    """
    estimator = CausalEstimator(synthetic_data)
    # dowhy (via pandas) raises KeyError when columns are missing.
    with pytest.raises(KeyError):
        estimator.estimate_effect(treatment="non_existent", outcome="outcome", confounders=["confounder"])


def test_confidence_interval_fallback(synthetic_data: pd.DataFrame) -> None:
    """
    Test fallback when confidence intervals are not available.
    """
    estimator = CausalEstimator(synthetic_data)

    # Mock the internal CausalModel and its estimate_effect return value
    with patch("coreason_inference.analysis.estimator.CausalModel") as MockModel:
        mock_instance = MockModel.return_value
        # Mock identify_effect
        mock_instance.identify_effect.return_value = MagicMock()

        # Mock estimate_effect return value
        mock_estimate = MagicMock()
        mock_estimate.value = 5.0
        mock_estimate.get_confidence_intervals.return_value = None  # SIMULATE MISSING CI
        mock_instance.estimate_effect.return_value = mock_estimate

        # Mock refute_estimate return value
        mock_refutation = MagicMock()
        mock_refutation.refutation_result = {
            "is_statistically_significant": False,
            "p_value": 0.8,
        }
        mock_instance.refute_estimate.return_value = mock_refutation

        result = estimator.estimate_effect(treatment="treatment", outcome="outcome", confounders=["confounder"])

        assert result.counterfactual_outcome == 5.0
        assert result.confidence_interval == (5.0, 5.0)  # Fallback: same as value


def test_confidence_interval_exists(synthetic_data: pd.DataFrame) -> None:
    """
    Test logic when confidence intervals ARE available.
    """
    estimator = CausalEstimator(synthetic_data)

    # Mock the internal CausalModel and its estimate_effect return value
    with patch("coreason_inference.analysis.estimator.CausalModel") as MockModel:
        mock_instance = MockModel.return_value
        mock_instance.identify_effect.return_value = MagicMock()

        mock_estimate = MagicMock()
        mock_estimate.value = 5.0
        mock_estimate.get_confidence_intervals.return_value = (4.0, 6.0)  # SIMULATE EXISTING CI
        mock_instance.estimate_effect.return_value = mock_estimate

        mock_refutation = MagicMock()
        mock_refutation.refutation_result = {
            "is_statistically_significant": False,
            "p_value": 0.8,
        }
        mock_instance.refute_estimate.return_value = mock_refutation

        result = estimator.estimate_effect(treatment="treatment", outcome="outcome", confounders=["confounder"])

        assert result.counterfactual_outcome == 5.0
        assert result.confidence_interval == (4.0, 6.0)
