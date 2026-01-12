# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

import numpy as np
import pandas as pd
import pytest

from coreason_inference.estimator import CausalEstimator


def test_estimator_linear_synthetic() -> None:
    # Synthetic Data: Y = T + 2*X + Noise
    # True Effect = 1.0
    np.random.seed(42)
    n = 200
    X = np.random.normal(size=n)
    T = 0.5 * X + np.random.normal(size=n)  # T correlated with X
    Y = 1.0 * T + 2.0 * X + np.random.normal(0, 0.1, size=n)

    df = pd.DataFrame({"T": T, "Y": Y, "X": X})

    estimator = CausalEstimator(treatment_is_binary=False)
    result = estimator.estimate_effect(df, "T", "Y", ["X"])

    # Check accuracy (allow some tolerance)
    assert 0.9 <= result.effect <= 1.1
    # Check refutation
    assert result.refutation_passed is True
    assert result.refutation_p_value >= 0.05


def test_estimator_binary_treatment() -> None:
    # Binary Treatment
    np.random.seed(42)
    n = 200
    X = np.random.normal(size=n)
    # T depends on X via sigmoid
    logit = X
    prob = 1 / (1 + np.exp(-logit))
    T = np.random.binomial(1, prob)
    # Y depends on T and X
    Y = 2.0 * T + 1.0 * X + np.random.normal(0, 0.1, size=n)

    df = pd.DataFrame({"T": T, "Y": Y, "X": X})

    estimator = CausalEstimator(treatment_is_binary=True)
    result = estimator.estimate_effect(df, "T", "Y", ["X"])

    # True effect is 2.0
    # DML might be slightly less precise with small N, loosen bounds
    assert 1.8 <= result.effect <= 2.2
    assert result.refutation_passed is True


def test_estimator_empty_data() -> None:
    estimator = CausalEstimator()
    with pytest.raises(ValueError, match="Data is empty"):
        estimator.estimate_effect(pd.DataFrame(), "T", "Y", ["X"])


def test_estimator_missing_columns() -> None:
    estimator = CausalEstimator()
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    with pytest.raises(ValueError, match="Missing columns"):
        estimator.estimate_effect(df, "T", "Y", ["X"])


def test_refutation_failure_flag() -> None:
    # To force refutation failure, we can try to find an effect where there is none,
    # OR rely on a mock to simulate the failure. Mocking is more reliable for testing the logic.
    from unittest.mock import MagicMock, patch

    df = pd.DataFrame({"T": [1], "Y": [1], "X": [1]})  # Dummy data

    with patch("coreason_inference.estimator.CausalModel") as MockModel:
        mock_instance = MockModel.return_value
        # Mock estimate
        mock_estimate = MagicMock()
        mock_estimate.value = 5.0
        mock_instance.estimate_effect.return_value = mock_estimate

        # Mock refuter to return significant p-value (low p-value = refutation failed to be null)
        mock_refute = MagicMock()
        mock_refute.refutation_result = {"p_value": 0.01}  # < 0.05 implies failure
        mock_instance.refute_estimate.return_value = mock_refute

        estimator = CausalEstimator()
        result = estimator.estimate_effect(df, "T", "Y", ["X"])

        assert result.refutation_passed is False
        assert result.refutation_p_value == 0.01
