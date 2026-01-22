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

from coreason_inference.analysis.estimator import CausalEstimator
from coreason_inference.schema import RefutationStatus


@pytest.fixture
def sample_data() -> pd.DataFrame:
    return pd.DataFrame({"T": [0, 1, 0, 1] * 10, "Y": [1, 2, 1, 3] * 10, "X": [0.1, 0.2, 0.3, 0.4] * 10})


def test_estimator_failure_handling(sample_data: pd.DataFrame) -> None:
    """
    Test that CausalEstimator gracefully handles cases where DoWhy returns None
    for the estimated effect (e.g., due to missing data/confounders).
    """
    estimator = CausalEstimator(sample_data)

    # Mock CausalModel to return an estimate with value=None
    with patch("coreason_inference.analysis.estimator.CausalModel") as mock_model_cls:
        mock_model = mock_model_cls.return_value

        # Mock identify_effect
        mock_model.identify_effect.return_value = MagicMock()

        # Mock estimate_effect returning None value
        mock_estimate = MagicMock()
        mock_estimate.value = None  # SIMULATE FAILURE
        mock_model.estimate_effect.return_value = mock_estimate

        # If the fix is NOT implemented, this should raise TypeError:
        # float() argument must be a string or a real number, not 'NoneType'
        # If fixed, it should return an InterventionResult with FAILED status

        result = estimator.estimate_effect(treatment="T", outcome="Y", confounders=["X"])

        assert result.refutation_status == RefutationStatus.FAILED
        assert result.counterfactual_outcome is None
        assert result.patient_id == "ERROR"
