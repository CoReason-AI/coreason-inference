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

from coreason_inference.analysis.estimator import CausalEstimator
from coreason_inference.schema import RefutationStatus


def test_estimate_effect_refutation_failure_invalidates_result(mock_causal_model: MagicMock) -> None:
    """
    Test that the estimator correctly invalidates the result (returns None)
    when the refutation fails (is statistically significant).
    """
    # 1. Setup Data
    data = pd.DataFrame(
        {
            "T": np.random.binomial(1, 0.5, 100),
            "Y": np.random.randn(100),
            "X": np.random.randn(100),
            "patient_id": [str(i) for i in range(100)],
        }
    )

    estimator = CausalEstimator(data)

    # 2. Configure Mock
    # Default behavior of fixture is Passing, so we override for Failure
    mock_causal_model.refute_estimate.return_value.refutation_result = {
        "is_statistically_significant": True,  # Failure
        "p_value": 0.01,
    }

    # 3. Call Method
    result = estimator.estimate_effect(treatment="T", outcome="Y", confounders=["X"])

    # 4. Assertions
    assert result.refutation_status == RefutationStatus.FAILED
    assert result.counterfactual_outcome is None
    assert result.cate_estimates is None
