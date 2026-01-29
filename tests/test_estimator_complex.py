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


def test_complex_binary_treatment(mock_user_context) -> None:
    """
    Complex Case 1: Binary Treatment.
    We generate data where T is binary (0/1).
    Y = 2*T + 0.5*C + Noise.
    Effect should be approx 2.0.
    """
    np.random.seed(42)
    n = 1000
    C = np.random.normal(0, 1, n)
    # Propensity score for T=1
    ps = 1 / (1 + np.exp(-(0.5 * C)))
    T = np.random.binomial(1, ps)
    Y = 2 * T + 0.5 * C + np.random.normal(0, 0.1, n)

    df = pd.DataFrame({"treatment": T, "outcome": Y, "confounder": C})

    estimator = CausalEstimator(df)
    result = estimator.estimate_effect(
        treatment="treatment",
        outcome="outcome",
        confounders=["confounder"],
        treatment_is_binary=True,
        context=mock_user_context,
    )

    # Check recovery of effect
    assert result.counterfactual_outcome == pytest.approx(2.0, abs=0.2)
    assert result.refutation_status == "PASSED"


def test_complex_multiple_confounders(mock_user_context) -> None:
    """
    Complex Case 2: Multiple Confounders (some irrelevant).
    Y = T + C1 + C2 + 0*C3.
    Effect = 1.0.
    """
    np.random.seed(42)
    n = 500
    C1 = np.random.normal(0, 1, n)
    C2 = np.random.normal(0, 1, n)
    C3 = np.random.normal(0, 1, n)  # Irrelevant
    T = C1 + C2 + np.random.normal(0, 0.1, n)
    Y = 1.0 * T + C1 + C2 + np.random.normal(0, 0.1, n)

    df = pd.DataFrame({"T": T, "Y": Y, "C1": C1, "C2": C2, "C3": C3})

    estimator = CausalEstimator(df)
    result = estimator.estimate_effect(
        treatment="T", outcome="Y", confounders=["C1", "C2", "C3"], context=mock_user_context
    )

    assert result.counterfactual_outcome == pytest.approx(1.0, abs=0.2)


def test_edge_case_empty_data(mock_user_context) -> None:
    """
    Edge Case 1: Empty DataFrame.
    """
    df = pd.DataFrame({"T": [], "Y": [], "C": []})
    estimator = CausalEstimator(df)
    # Dowhy/Pandas should raise error on empty data or init
    with pytest.raises((ValueError, KeyError)):
        estimator.estimate_effect("T", "Y", ["C"], context=mock_user_context)


def test_edge_case_collinear_confounders(mock_user_context) -> None:
    """
    Edge Case 3: Perfectly Collinear Confounders.
    C2 = 2 * C1.
    LinearDML (Regularized) should handle this or fail gracefully.
    sklearn's LinearRegression handles collinearity by ignoring one or unstable coeffs,
    but since we care about Treatment effect (T), not C coefficients, it might be fine.
    """
    np.random.seed(42)
    n = 500
    C1 = np.random.normal(0, 1, n)
    C2 = 2 * C1  # Perfectly collinear
    T = C1 + np.random.normal(0, 0.1, n)
    Y = 3.0 * T + C1 + np.random.normal(0, 0.1, n)

    df = pd.DataFrame({"T": T, "Y": Y, "C1": C1, "C2": C2})

    estimator = CausalEstimator(df)
    # It should not crash. Effect recovery might be stable because T is distinct from C1/C2 block.
    result = estimator.estimate_effect("T", "Y", ["C1", "C2"], context=mock_user_context)

    assert result.counterfactual_outcome == pytest.approx(3.0, abs=0.2)


def test_refutation_failure_logic(synthetic_data: pd.DataFrame, mock_user_context) -> None:
    """
    Test logic when refutation FAILS (p-value < 0.05).
    We mock the refutation result.
    """
    estimator = CausalEstimator(synthetic_data)

    with patch("coreason_inference.analysis.estimator.CausalModel") as MockModel:
        mock_instance = MockModel.return_value
        mock_instance.identify_effect.return_value = MagicMock()

        mock_estimate = MagicMock()
        mock_estimate.value = 10.0
        mock_estimate.get_confidence_intervals.return_value = (9.0, 11.0)
        mock_instance.estimate_effect.return_value = mock_estimate

        # Mock SIGNIFICANT refutation (p < 0.05) -> FAILED
        mock_refutation = MagicMock()
        mock_refutation.refutation_result = {
            "is_statistically_significant": True,  # Significant means placebo HAD an effect -> FAILURE
            "p_value": 0.01,
        }
        mock_instance.refute_estimate.return_value = mock_refutation

        result = estimator.estimate_effect("treatment", "outcome", ["confounder"], context=mock_user_context)

        assert result.refutation_status == "FAILED"
        assert result.counterfactual_outcome is None


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    # Small dummy data for the mock test
    return pd.DataFrame({"treatment": [1, 2], "outcome": [3, 4], "confounder": [0, 1]})
