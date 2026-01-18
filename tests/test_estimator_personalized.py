import numpy as np
import pandas as pd
import pytest

from coreason_inference.analysis.estimator import METHOD_FOREST, CausalEstimator


def test_estimator_personalized_inference() -> None:
    """
    Test retrieving a specific patient's CATE.
    """
    np.random.seed(42)
    n = 500  # Increased sample size for better forest convergence
    # Patient IDs
    pids = [f"P{i:03d}" for i in range(n)]

    # Feature X
    X = np.linspace(-1, 1, n)
    # Treatment T
    T = np.random.binomial(1, 0.5, n)

    # Effect depends on X: Effect = 10 * X
    true_cate = 10 * X

    Y = true_cate * T + np.random.normal(0, 0.1, n)

    data = pd.DataFrame({"patient_id": pids, "T": T, "Y": Y, "X": X})

    estimator = CausalEstimator(data)

    # 1. Test retrieving for specific patient (e.g., last patient has X ~ 1, Effect ~ 10)
    target_pid = pids[-1]

    result = estimator.estimate_effect(
        treatment="T",
        outcome="Y",
        confounders=["X"],
        patient_id_col="patient_id",
        method=METHOD_FOREST,
        target_patient_id=target_pid,
    )

    assert result.patient_id == target_pid
    # Forest estimate should be positive and significant (e.g. > 5.0)
    # We relax the check to avoid flakiness, as Forests can smooth extrema.
    assert result.counterfactual_outcome > 5.0

    # 2. Test retrieving for first patient (X ~ -1, Effect ~ -10)
    target_pid_2 = pids[0]

    result_2 = estimator.estimate_effect(
        treatment="T",
        outcome="Y",
        confounders=["X"],
        patient_id_col="patient_id",
        method=METHOD_FOREST,
        target_patient_id=target_pid_2,
    )

    assert result_2.patient_id == target_pid_2
    assert result_2.counterfactual_outcome < -5.0


def test_estimator_personalized_inference_missing_id() -> None:
    """
    Test error handling when target ID is missing.
    """
    # Use sufficient data to avoid SVD non-convergence in EconML
    data = pd.DataFrame(
        {
            "patient_id": [f"P{i}" for i in range(20)],
            "T": [0, 1] * 10,
            "Y": [0, 1] * 10,
            "X": np.random.normal(0, 1, 20),
        }
    )
    estimator = CausalEstimator(data)

    with pytest.raises(ValueError, match="not found"):
        estimator.estimate_effect("T", "Y", ["X"], method=METHOD_FOREST, target_patient_id="C")
