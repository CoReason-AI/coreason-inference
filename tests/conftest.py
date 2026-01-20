# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import Generator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_causal_model() -> Generator[MagicMock, None, None]:
    """
    Fixture that mocks dowhy.CausalModel.
    Returns the mock instance where you can configure return values
    for estimate_effect and refute_estimate.
    """
    with patch("coreason_inference.analysis.estimator.CausalModel") as MockModel:
        mock_instance = MockModel.return_value

        # Default Identify Mock
        mock_instance.identify_effect.return_value = MagicMock()

        # Default Estimate Mock (valid result)
        mock_estimate = MagicMock()
        mock_estimate.value = 10.0
        mock_estimate.get_confidence_intervals.return_value = (5.0, 15.0)
        mock_instance.estimate_effect.return_value = mock_estimate

        # Default Refutation Mock (Passing)
        mock_refutation = MagicMock()
        mock_refutation.refutation_result = {
            "is_statistically_significant": False,  # PASSED
            "p_value": 0.5,
        }
        mock_instance.refute_estimate.return_value = mock_refutation

        yield mock_instance
