# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from coreason_inference.analysis.estimator import METHOD_FOREST, CausalEstimator
from coreason_inference.schema import RefutationStatus


def test_refutation_failure_wipes_cate_estimates(mock_causal_model: MagicMock) -> None:
    """
    Edge Case: Causal Forest method used.
    If refutation fails, ensure 'cate_estimates' (which are expensive to compute)
    are also invalidated (set to None) to prevent usage of a flawed model.
    """
    data = pd.DataFrame(
        {
            "T": np.random.binomial(1, 0.5, 10),
            "Y": np.random.randn(10),
            "X": np.random.randn(10),
            "patient_id": [str(i) for i in range(10)],
        }
    )
    estimator = CausalEstimator(data)

    # Configure Mock
    mock_instance = mock_causal_model

    # Mock Estimate with CATEs
    mock_estimate = mock_instance.estimate_effect.return_value
    mock_estimate.value = 5.0
    mock_estimate.get_confidence_intervals.return_value = (4.0, 6.0)

    # Mocking internal estimator.effect(X) for EconML
    mock_econml = MagicMock()
    # Return 10 CATEs
    mock_econml.effect.return_value = np.random.randn(10, 1)
    mock_estimate.estimator = mock_econml

    # Mock FAILED Refutation (Significant Placebo Effect)
    mock_instance.refute_estimate.return_value.refutation_result = {
        "is_statistically_significant": True,
        "p_value": 0.001
    }

    result = estimator.estimate_effect("T", "Y", ["X"], method=METHOD_FOREST)

    assert result.refutation_status == RefutationStatus.FAILED
    assert result.counterfactual_outcome is None
    assert result.cate_estimates is None  # CRITICAL: Must be wiped


def test_refutation_failure_invalidates_personalized_inference(mock_causal_model: MagicMock) -> None:
    """
    Edge Case: User asks for a specific patient's effect (target_patient_id).
    If the global model fails refutation, the personalized estimate must also be invalid.
    """
    data = pd.DataFrame(
        {
            "T": [0, 1] * 5,
            "Y": np.random.randn(10),
            "X": np.random.randn(10),
            "patient_id": ["P1"] + [str(i) for i in range(9)],
        }
    )
    estimator = CausalEstimator(data)

    mock_instance = mock_causal_model
    mock_estimate = mock_instance.estimate_effect.return_value
    mock_estimate.value = 5.0

    # Setup CATEs so we can extract for P1 (index 0)
    mock_econml = MagicMock()
    cates = np.zeros((10, 1))
    cates[0] = 99.0  # Specific value for P1
    mock_estimate.estimator.effect.return_value = cates

    # Mock FAILED Refutation
    mock_instance.refute_estimate.return_value.refutation_result = {
        "is_statistically_significant": True, "p_value": 0.02
    }

    result = estimator.estimate_effect("T", "Y", ["X"], method=METHOD_FOREST, target_patient_id="P1")

    assert result.patient_id == "P1"
    assert result.refutation_status == RefutationStatus.FAILED
    # Even though we found P1's raw CATE (99.0), it should be suppressed
    assert result.counterfactual_outcome is None
    assert result.cate_estimates is None


@pytest.mark.parametrize(
    "p_value, expected_status, is_valid",
    [
        (0.06, RefutationStatus.PASSED, True),  # p > 0.05 -> Null accepted (Placebo has no effect) -> Valid
        (0.04, RefutationStatus.FAILED, False),  # p < 0.05 -> Null rejected (Placebo has effect) -> Invalid
        (0.05, RefutationStatus.FAILED, False),  # p <= 0.05 -> Null rejected -> Invalid (Standard significance)
    ],
)
def test_refutation_boundary_logic(mock_causal_model: MagicMock, p_value: float, expected_status: str, is_valid: bool) -> None:
    """
    Edge Case: Boundary testing for p-values.
    """
    data = pd.DataFrame({"T": [1], "Y": [1], "X": [1]})
    estimator = CausalEstimator(data)

    mock_instance = mock_causal_model

    # Logic: If p <= 0.05, it is significant (Placebo != 0) -> FAILED
    is_sig = p_value <= 0.05

    # Override Refutation
    mock_instance.refute_estimate.return_value.refutation_result = {
        "is_statistically_significant": is_sig, "p_value": p_value
    }

    result = estimator.estimate_effect("T", "Y", ["X"])

    # Expected Status
    expected_status_enum = RefutationStatus.FAILED if is_sig else RefutationStatus.PASSED

    assert result.refutation_status == expected_status_enum

    if is_valid:
        assert result.counterfactual_outcome == 10.0
    else:
        assert result.counterfactual_outcome is None


def test_binary_treatment_refutation_failure(mock_causal_model: MagicMock) -> None:
    """
    Edge Case: Binary Treatment flag shouldn't bypass safety.
    """
    data = pd.DataFrame({"T": [0, 1], "Y": [0, 1], "X": [0, 1]})
    estimator = CausalEstimator(data)

    mock_instance = mock_causal_model

    # Mock FAILED
    mock_instance.refute_estimate.return_value.refutation_result = {
        "is_statistically_significant": True, "p_value": 0.01
    }

    result = estimator.estimate_effect("T", "Y", ["X"], treatment_is_binary=True)

    assert result.refutation_status == RefutationStatus.FAILED
    assert result.counterfactual_outcome is None
