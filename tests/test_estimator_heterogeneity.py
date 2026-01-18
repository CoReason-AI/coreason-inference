import numpy as np
import pandas as pd

from coreason_inference.analysis.estimator import CausalEstimator
from coreason_inference.schema import RefutationStatus


def test_estimator_heterogeneity_linear_regression() -> None:
    """
    Test that LinearDML (default) returns None for cate_estimates.
    """
    # Create synthetic data
    # y = T + X + noise
    n = 100
    data = pd.DataFrame(
        {
            "T": np.random.binomial(1, 0.5, n),
            "X": np.random.normal(0, 1, n),
            "W": np.random.normal(0, 1, n),
        }
    )
    data["Y"] = data["T"] + data["X"] + 0.1 * np.random.normal(0, 1, n)

    estimator = CausalEstimator(data)
    result = estimator.estimate_effect(treatment="T", outcome="Y", confounders=["X", "W"], method="linear")

    assert result.cate_estimates is None
    assert result.patient_id == "POPULATION_ATE"
    # Effect should be roughly 1.0
    assert 0.8 < result.counterfactual_outcome < 1.2
    assert result.refutation_status == RefutationStatus.PASSED or result.refutation_status == RefutationStatus.FAILED
    # Placebo usually passes (FAILED to reject null of 0 effect on placebo) -> Status PASSED


def test_estimator_heterogeneity_causal_forest() -> None:
    """
    Test that CausalForestDML returns cate_estimates and captures heterogeneity.
    """
    # Create synthetic data with Heterogeneous Effect
    # Effect = X (if X > 0 else 0)
    # Y = Effect * T + W + noise
    np.random.seed(42)
    n = 200
    X = np.random.normal(0, 1, n)
    W = np.random.normal(0, 1, n)
    T = np.random.binomial(1, 0.5, n)

    # True Effect: 2.0 if X > 0, else 0.0
    true_cate = np.where(X > 0, 2.0, 0.0)

    Y = true_cate * T + 0.5 * W + 0.1 * np.random.normal(0, 1, n)

    data = pd.DataFrame({"T": T, "Y": Y, "X": X, "W": W})

    estimator = CausalEstimator(data)

    # We treat X and W as confounders/modifiers
    # The updated code sets effect_modifiers = confounders if method="forest"
    result = estimator.estimate_effect(treatment="T", outcome="Y", confounders=["X", "W"], method="forest")

    assert result.cate_estimates is not None
    assert len(result.cate_estimates) == n

    # Check if estimated CATE correlates with True CATE
    estimated_cate = np.array(result.cate_estimates)
    correlation = np.corrcoef(true_cate, estimated_cate)[0, 1]

    # Causal Forests require more data to be precise, but correlation should be positive
    # With n=200, it might be noisy but > 0.3 usually
    assert correlation > 0.2, f"CATE correlation {correlation} too low"

    # Check High CATE group vs Low CATE group
    high_cate_indices = np.where(X > 0)[0]
    low_cate_indices = np.where(X <= 0)[0]

    mean_high = np.mean(estimated_cate[high_cate_indices])
    mean_low = np.mean(estimated_cate[low_cate_indices])

    assert mean_high > mean_low, "Forest failed to identify subgroup difference"


def test_estimator_invalid_method() -> None:
    """
    Test fallback or error for invalid method.
    Currently implementation might fail or default.
    Reviewing implementation: It checks `if method == "forest"`, else linear.
    So "invalid" becomes linear.
    """
    data = pd.DataFrame({"T": [0, 1] * 10, "Y": [0, 1] * 10, "X": [0] * 20})
    estimator = CausalEstimator(data)
    result = estimator.estimate_effect("T", "Y", ["X"], method="invalid")
    assert result.cate_estimates is None  # Should default to linear
