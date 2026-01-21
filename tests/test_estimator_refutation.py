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

import numpy as np
import pandas as pd
import pytest

from coreason_inference.analysis.estimator import CausalEstimator, RefutationFailedError


def test_estimate_effect_refutation_failure_invalidates_result() -> None:
    """
    Test that the estimator correctly invalidates the result (raises RefutationFailedError)
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

    # 2. Mock DoWhy CausalModel
    with patch("coreason_inference.analysis.estimator.CausalModel") as MockModel:
        mock_instance = MockModel.return_value

        # Mock identify_effect
        mock_instance.identify_effect.return_value = MagicMock()

        # Mock estimate_effect (return a dummy estimate)
        mock_estimate = MagicMock()
        mock_estimate.value = 10.0
        mock_estimate.get_confidence_intervals.return_value = (5.0, 15.0)
        mock_instance.estimate_effect.return_value = mock_estimate

        # Mock refute_estimate
        # Refutation FAILED means 'is_statistically_significant' is True (Placebo has effect)
        mock_refutation = MagicMock()
        mock_refutation.refutation_result = {
            "is_statistically_significant": True,  # This implies FAILURE of the test
            "p_value": 0.01,
        }
        mock_instance.refute_estimate.return_value = mock_refutation

        # 3. Call Method & Assert Exception
        with pytest.raises(RefutationFailedError, match="Estimate invalidated due to failed refutation"):
            estimator.estimate_effect(treatment="T", outcome="Y", confounders=["X"])
