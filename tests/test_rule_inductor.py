# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from coreason_inference.analysis.rule_inductor import RuleInductor
from coreason_inference.schema import OptimizationOutput


@pytest.fixture
def synthetic_cate_data() -> Tuple[pd.DataFrame, np.ndarray]:
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


def test_rule_inductor_fit(synthetic_cate_data: Tuple[pd.DataFrame, np.ndarray]) -> None:
    features, cate = synthetic_cate_data
    inductor = RuleInductor()
    inductor.fit(features, cate)
    assert inductor.tree_model is not None
    assert inductor.feature_names == ["Age", "BMI", "Biomarker"]


def test_rule_inductor_induce(synthetic_cate_data: Tuple[pd.DataFrame, np.ndarray]) -> None:
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


def test_unfitted_error(synthetic_cate_data: Tuple[pd.DataFrame, np.ndarray]) -> None:
    features, cate = synthetic_cate_data
    inductor = RuleInductor()
    with pytest.raises(ValueError):
        inductor.induce_rules_with_data(features, cate)


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
