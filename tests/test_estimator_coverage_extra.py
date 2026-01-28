import numpy as np
import pandas as pd
import pytest

from coreason_inference.analysis.estimator import METHOD_FOREST, CausalEstimator


def test_estimator_personalized_inference_missing_column(mock_user_context) -> None:
    """
    Test error when patient_id_col is missing.
    """
    # Use sufficient data to avoid SVD failure (N=20)
    data = pd.DataFrame({"T": [0, 1] * 10, "Y": [0, 1] * 10, "X": np.random.normal(0, 1, 20)})
    # patient_id column is missing

    estimator = CausalEstimator(data)

    with pytest.raises(ValueError, match="Patient ID column 'pid' not found"):
        estimator.estimate_effect("T", "Y", ["X"], patient_id_col="pid", method=METHOD_FOREST, target_patient_id="P1", context=mock_user_context)
