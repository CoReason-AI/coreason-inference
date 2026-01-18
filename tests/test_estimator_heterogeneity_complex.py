import numpy as np
import pandas as pd
import pytest

from coreason_inference.analysis.estimator import CausalEstimator


def test_heterogeneity_binary_treatment_complex() -> None:
    """
    Test CausalForestDML with a binary treatment and complex heterogeneity.
    Scenario: Treatment T works (Y increases) only if Age > 50 AND Biomarker < 0.
    """
    np.random.seed(101)
    n = 500

    # Confounders / Modifiers
    age = np.random.uniform(20, 80, n)
    biomarker = np.random.normal(0, 1, n)
    # Nuisance
    noise_var = np.random.normal(0, 1, n)

    # Binary Treatment (randomized)
    T = np.random.binomial(1, 0.5, n)

    # Heterogeneous Effect Logic
    # Effect is 5.0 if Age > 50 and Biomarker < 0, else 0.0
    # Create boolean mask
    responders = (age > 50) & (biomarker < 0)
    true_cate = np.where(responders, 5.0, 0.0)

    # Outcome
    # Y = T * Effect + f(Age) + noise
    # Add strong confounding on outcome from Age to make it harder
    baseline_risk = 0.1 * age
    Y = true_cate * T + baseline_risk + 0.1 * np.random.normal(0, 1, n)

    data = pd.DataFrame({"T": T, "Y": Y, "Age": age, "Biomarker": biomarker, "Noise": noise_var})

    estimator = CausalEstimator(data)
    result = estimator.estimate_effect(
        treatment="T", outcome="Y", confounders=["Age", "Biomarker", "Noise"], treatment_is_binary=True, method="forest"
    )

    assert result.cate_estimates is not None
    estimated_cate = np.array(result.cate_estimates)

    # Verification:
    # 1. Correlation with ground truth
    corr = np.corrcoef(true_cate, estimated_cate)[0, 1]
    assert corr > 0.3, f"CATE correlation {corr} too low for binary complex scenario"

    # 2. Check group means
    mean_responders = np.mean(estimated_cate[responders])
    mean_non_responders = np.mean(estimated_cate[~responders])

    # Responders should have significantly higher effect
    assert mean_responders > 2.0
    assert mean_non_responders < 1.0
    assert (mean_responders - mean_non_responders) > 2.0


def test_heterogeneity_empty_confounders_error() -> None:
    """
    Edge Case: Calling method="forest" with no confounders.
    Causal Forest requires at least one variable to split on.
    """
    data = pd.DataFrame({"T": [0, 1] * 10, "Y": [0, 1] * 10})

    estimator = CausalEstimator(data)

    # This should likely fail or raise a warning.
    # If implementation does not handle it, it might raise ValueError from EconML (X=None).
    # We want to verify behavior.

    # DoWhy/EconML usually raises ValueError when X is None or empty for CausalForest
    with pytest.raises(ValueError):
        estimator.estimate_effect(
            treatment="T",
            outcome="Y",
            confounders=[],  # Empty
            method="forest",
        )


def test_heterogeneity_small_data() -> None:
    """
    Edge Case: Extremely small dataset (N=10).
    Forest might struggle or warn, but shouldn't crash ungracefully?
    Or EconML might require min samples.
    """
    np.random.seed(42)
    data = pd.DataFrame(
        {"T": np.random.binomial(1, 0.5, 10), "Y": np.random.normal(0, 1, 10), "X": np.random.normal(0, 1, 10)}
    )

    estimator = CausalEstimator(data)

    # Should run without crashing, even if results are junk
    try:
        result = estimator.estimate_effect(treatment="T", outcome="Y", confounders=["X"], method="forest")
        # Result might be valid object
        assert result.refutation_status is not None
        assert result.cate_estimates is not None
        assert len(result.cate_estimates) == 10
    except ValueError as e:
        # If EconML throws specific error about sample size, that's acceptable too,
        # but ideally we handle it.
        # EconML usually warns or runs with poor splits.
        pytest.fail(f"Small data caused crash: {e}")


def test_heterogeneity_collinear_features() -> None:
    """
    Complex Scenario: Perfectly collinear confounders.
    X1 = X2.
    """
    np.random.seed(42)
    n = 100
    X1 = np.random.normal(0, 1, n)
    X2 = X1.copy()  # Duplicate
    T = np.random.binomial(1, 0.5, n)
    Y = T * (X1 > 0) + X1 + np.random.normal(0, 0.1, n)

    data = pd.DataFrame({"T": T, "Y": Y, "X1": X1, "X2": X2})

    estimator = CausalEstimator(data)

    result = estimator.estimate_effect(treatment="T", outcome="Y", confounders=["X1", "X2"], method="forest")

    assert result.cate_estimates is not None
    # Forest handles collinearity well (just picks one split).
    # Check simple correlation
    cate = np.array(result.cate_estimates)
    true_cate = (X1 > 0).astype(float)
    assert np.corrcoef(cate, true_cate)[0, 1] > 0.0
