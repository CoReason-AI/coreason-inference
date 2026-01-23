# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import Any, Tuple

import numpy as np
import pandas as pd
import pytest

from coreason_inference.analysis.rule_inductor import RuleInductor
from coreason_inference.schema import OptimizationOutput


@pytest.fixture
def synthetic_cate_data() -> Tuple[pd.DataFrame, np.ndarray[Any, Any]]:
    """
    Generates synthetic data where Age > 50 and BMI < 30 leads to high CATE.
    """
    np.random.seed(42)
    n = 200

    age = np.random.randint(20, 80, size=n)
    bmi = np.random.uniform(18, 40, size=n)
    biomarker = np.random.normal(0, 1, size=n)

    features = pd.DataFrame({"Age": age, "BMI": bmi, "Biomarker": biomarker})

    # CATE Generation:
    # Baseline 0.1
    # Age > 50: +0.4
    # BMI < 30: +0.3
    # If both, +0.7 -> High Response
    cate = np.full(n, 0.1)

    mask_age = age > 50
    mask_bmi = bmi < 30

    cate[mask_age] += 0.4
    cate[mask_bmi] += 0.3

    # Add some noise
    cate += np.random.normal(0, 0.05, size=n)

    return features, cate


def test_rule_inductor_initialization() -> None:
    inductor = RuleInductor(max_depth=3)
    assert inductor.max_depth == 3
    assert inductor.tree_model is None


def test_rule_inductor_fit(synthetic_cate_data: Tuple[pd.DataFrame, np.ndarray[Any, Any]]) -> None:
    features, cate = synthetic_cate_data
    inductor = RuleInductor()
    inductor.fit(features, cate)
    assert inductor.tree_model is not None
    assert inductor.feature_names == ["Age", "BMI", "Biomarker"]


def test_rule_inductor_induce(synthetic_cate_data: Tuple[pd.DataFrame, np.ndarray[Any, Any]]) -> None:
    """Test the main method (without data re-input) for coverage."""
    features, cate = synthetic_cate_data
    inductor = RuleInductor(max_depth=3, min_samples_leaf=10)
    inductor.fit(features, cate)

    # Call the original induce_rules method which relies on tree statistics (Mean CATE)
    # instead of recalculating PoS from data.
    result = inductor.induce_rules(cate)

    assert isinstance(result, OptimizationOutput)
    assert len(result.new_criteria) > 0
    # Original Pos and Optimized Pos are placeholders or baselines in this method
    assert result.original_pos == result.optimized_pos
    # Should have a safety flag
    assert len(result.safety_flags) > 0
    assert "inaccurate without feature data" in result.safety_flags[0]


def test_rule_inductor_induce_with_data(synthetic_cate_data: Tuple[pd.DataFrame, np.ndarray[Any, Any]]) -> None:
    features, cate = synthetic_cate_data
    inductor = RuleInductor(max_depth=3, min_samples_leaf=10)
    inductor.fit(features, cate)

    # Use the method that takes data
    result = inductor.induce_rules_with_data(features, cate)

    assert isinstance(result, OptimizationOutput)
    assert len(result.new_criteria) > 0
    # With 100% responders, optimized_pos cannot exceed original_pos
    assert result.optimized_pos >= result.original_pos

    # Check if logic roughly matches
    # We expect rules involving Age and BMI
    rule_features = [r.feature for r in result.new_criteria]
    assert "Age" in rule_features or "BMI" in rule_features

    # Check PoS Logic
    # Baseline PoS (Proportion > 0). With base 0.1, everyone is > 0 actually?
    # Wait, my synthetic data has base 0.1. So 100% are positive.
    # Let's adjust synthetic data to have non-responders.

    # Create non-responders
    cate_mixed = cate.copy()
    cate_mixed -= 0.5  # Shift so baseline is -0.4, Age>50 -> 0.0, Both -> 0.3
    # Now baseline is negative.

    inductor.fit(features, cate_mixed)
    result_mixed = inductor.induce_rules_with_data(features, cate_mixed)

    assert result_mixed.optimized_pos > result_mixed.original_pos
    # Original PoS should be low (only double positive or single positive + noise > 0)
    # Optimized PoS should be high (focusing on the group)


def test_empty_input() -> None:
    inductor = RuleInductor()
    with pytest.raises(ValueError):
        inductor.fit(pd.DataFrame(), np.array([]))


def test_mismatched_length_input(synthetic_cate_data: Tuple[pd.DataFrame, np.ndarray[Any, Any]]) -> None:
    """Test mismatch length error (Coverage for line 50)."""
    features, cate = synthetic_cate_data
    # Pass mismatched lengths
    with pytest.raises(ValueError, match="must have the same length"):
        inductor = RuleInductor()
        inductor.fit(features, cate[:-1])


def test_unfitted_error(synthetic_cate_data: Tuple[pd.DataFrame, np.ndarray[Any, Any]]) -> None:
    features, cate = synthetic_cate_data
    inductor = RuleInductor()
    with pytest.raises(ValueError):
        inductor.induce_rules(cate)
    with pytest.raises(ValueError):
        inductor.induce_rules_with_data(features, cate)


def test_non_numeric_features_pass() -> None:
    """Test that non-numeric check just passes (coverage for line 59 pass)."""
    # DecisionTreeRegressor actually fails if strings are passed, unless we encode.
    # But our check logic had a `pass`. We should verify it doesn't crash BEFORE the model fit.
    # However, if we pass strings, model.fit WILL crash.
    # The coverage is for the `if not numeric: pass` block.

    # We can mock `np.issubdtype` or just pass mixed types that sklearn might choke on later,
    # but we want to hit the `pass` line.

    # Note: I removed the `if not numeric: pass` block in refactor.
    # So this test is less relevant for that specific line, but good for robustness.

    df = pd.DataFrame({"A": ["a", "b"]})
    cate = np.array([1, 2])
    inductor = RuleInductor()

    # It will raise ValueError from sklearn, but should hit our check first.
    with pytest.raises(ValueError):  # ValueError: could not convert string to float
        inductor.fit(df, cate)


def test_no_valid_subgroup() -> None:
    # Data where everything is random noise, mean 0
    np.random.seed(42)
    features = pd.DataFrame({"A": np.random.rand(100)})
    cate = np.random.normal(0, 0.1, 100)  # Random noise

    # Force tree to not find splits easily or all leaves similar
    inductor = RuleInductor(max_depth=1, min_samples_leaf=50)
    inductor.fit(features, cate)

    result = inductor.induce_rules_with_data(features, cate)

    # Even with noise, it might pick a leaf that is slightly better by chance.
    # But structure should be valid.
    assert isinstance(result, OptimizationOutput)

    # Also test induce_rules (empty/fail case coverage)
    # We need to simulate a tree where no leaf is better?
    # Hard with random noise, as "best" is always >= others.
    # The code handles "best_leaf_idx = -1" but that only happens if list is empty.
    # A decision tree always has at least one leaf (root if no splits).
    # So that branch might be dead code unless tree logic allows 0 nodes (impossible after fit).
