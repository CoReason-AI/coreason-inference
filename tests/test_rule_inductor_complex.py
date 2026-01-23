import numpy as np
import pandas as pd
import pytest
from coreason_inference.analysis.rule_inductor import RuleInductor

def test_outlier_dominance() -> None:
    """
    Verifies that the Mean CATE optimization adheres strictly to the mean,
    even if it's driven by a single outlier (High Variance).

    Group A: Stable Moderate Response (Mean = 1.0, Variance = 0)
    Group B: Mostly Non-Responders, but one Super-Super-Responder (Mean = 2.0)

    Expected: Select Group B.
    """
    # Group A: 10 people, all 1.0
    grp_a = np.ones(10) * 1.0
    # Group B: 10 people. 9 are 0.0, 1 is 20.0. Mean = 20/10 = 2.0.
    grp_b = np.zeros(10)
    grp_b[-1] = 20.0

    cate = np.concatenate([grp_a, grp_b])
    features = pd.DataFrame({
        "Group": np.concatenate([np.zeros(10), np.ones(10)])
    })

    inductor = RuleInductor(max_depth=1, min_samples_leaf=5)
    inductor.fit(features, cate)

    result = inductor.induce_rules_with_data(features, cate)

    # Should select Group B (Mean 2.0) over Group A (Mean 1.0)
    # Group B is Group > 0.5
    rule = result.new_criteria[0]
    assert rule.feature == "Group"
    assert rule.operator == ">"
    assert result.optimized_pos == pytest.approx(2.0)

def test_interaction_recovery() -> None:
    """
    Verifies that the inductor correctly extracts rules for a subgroup
    defined by an interaction (AND logic).

    Target: Age > 50 AND Biomarker > 0 -> Mean CATE 5.0
    Others: Mean CATE 0.0
    """
    np.random.seed(42)
    n = 200
    age = np.random.randint(20, 80, size=n)
    biomarker = np.random.normal(0, 1, size=n)

    # Construct CATE
    # Default 0
    cate = np.zeros(n)

    # Target Group
    mask = (age > 50) & (biomarker > 0)
    cate[mask] = 5.0

    # Add slight noise to avoid perfect splits issues
    cate += np.random.normal(0, 0.1, size=n)

    features = pd.DataFrame({"Age": age, "Biomarker": biomarker})

    # Needs depth at least 2 to find interaction
    inductor = RuleInductor(max_depth=2, min_samples_leaf=10)
    inductor.fit(features, cate)

    result = inductor.induce_rules_with_data(features, cate)

    # We expect optimized PoS close to 5.0
    assert result.optimized_pos > 4.0

    # Check rules. Should contain Age and Biomarker
    feats = [r.feature for r in result.new_criteria]
    assert "Age" in feats
    assert "Biomarker" in feats

    # Verify values roughly
    for r in result.new_criteria:
        if r.feature == "Age":
            assert r.operator == ">"
            assert r.value >= 40 # Split might not be exactly 50 depending on data distribution
        if r.feature == "Biomarker":
            assert r.operator == ">"
            # Split around 0

def test_least_harmful_selection() -> None:
    """
    Verifies behavior when ALL subgroups have negative effects.
    Should select the 'least bad' (highest arithmetic mean).

    Group A: Mean -5.0 (Toxic)
    Group B: Mean -1.0 (Less Toxic)
    """
    # Group A (0): -5.0
    # Group B (1): -1.0

    features = pd.DataFrame({"Type": np.concatenate([np.zeros(50), np.ones(50)])})
    cate = np.concatenate([np.full(50, -5.0), np.full(50, -1.0)])

    inductor = RuleInductor(max_depth=1, min_samples_leaf=10)
    inductor.fit(features, cate)

    result = inductor.induce_rules_with_data(features, cate)

    # Should pick Group B (Type > 0.5) with Mean -1.0
    assert result.optimized_pos == pytest.approx(-1.0)
    rule = result.new_criteria[0]
    assert rule.feature == "Type"
    assert rule.operator == ">"

def test_robustness_noise() -> None:
    """
    Verifies that the system produces a valid output structure even with pure noise.
    """
    np.random.seed(99)
    n = 100
    features = pd.DataFrame(np.random.rand(n, 5), columns=[f"F{i}" for i in range(5)])
    cate = np.random.normal(0, 1, size=n)

    inductor = RuleInductor(max_depth=2, min_samples_leaf=5)
    inductor.fit(features, cate)

    result = inductor.induce_rules_with_data(features, cate)

    # Must return a valid object
    assert len(result.new_criteria) > 0
    # Optimized pos should be reasonable (within range of data)
    assert result.optimized_pos <= cate.max()
    assert result.optimized_pos >= cate.min()
    # Rationale populated
    assert "Mean Effect" in result.new_criteria[0].rationale
