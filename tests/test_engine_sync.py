# Copyright (c) 2026 CoReason, Inc.
# Licensed under the Prosperity Public License 3.0.0

from unittest.mock import patch

import pandas as pd

from coreason_inference.engine import InferenceEngine
from coreason_inference.schema import InterventionResult, RefutationStatus


def test_sync_facade_call() -> None:
    """Explicitly verify that InferenceEngine facade calls operate synchronously via anyio.run."""
    engine = InferenceEngine()
    engine.augmented_data = pd.DataFrame({"X1": [1], "T": [0], "Y": [1]})

    # Mock result
    mock_result = InterventionResult(
        patient_id="POPULATION_ATE",
        intervention="do(T)",
        counterfactual_outcome=0.5,
        confidence_interval=(0.4, 0.6),
        refutation_status=RefutationStatus.PASSED,
        cate_estimates=[0.1],
    )

    with patch("coreason_inference.engine.CausalEstimator") as MockEstimator:
        instance = MockEstimator.return_value
        instance.estimate_effect.return_value = mock_result

        # This call should block and return result, not coroutine
        result = engine.estimate_effect("T", "Y", ["X1"])

        assert result == mock_result
        # Ensure it was called
        instance.estimate_effect.assert_called_once()
