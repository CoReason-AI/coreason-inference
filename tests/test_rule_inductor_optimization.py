import numpy as np
import pandas as pd
import pytest
from coreason_inference.analysis.rule_inductor import RuleInductor

def test_rule_inductor_optimizes_mean_cate_over_pos() -> None:
    """
    Verifies that the RuleInductor prioritizes Mean CATE over Probability of Success (Responder Rate).

    Scenario:
    - Group A: 100% Responder Rate, but low Mean CATE (0.1).
    - Group B: 60% Responder Rate, but high Mean CATE (1.16).

    Expected: Group B should be selected.
    """
    # Group A: High PoS (100%), Low Mean CATE (0.1)
    # Feature 'Group' = 0
    n_a = 100
    group_a_feat = np.zeros(n_a)
    group_a_cate = np.full(n_a, 0.1) # Mean = 0.1, PoS = 1.0

    # Group B: Lower PoS (60%), High Mean CATE (2.0 and -0.1)
    # Feature 'Group' = 1
    n_b = 100
    group_b_feat = np.ones(n_b)
    # 60% responders with high effect, 40% non-responders with slight negative effect
    n_b_pos = 60
    n_b_neg = 40
    group_b_cate = np.concatenate([np.full(n_b_pos, 2.0), np.full(n_b_neg, -0.1)])
    # Mean B = (60*2.0 + 40*-0.1) / 100 = (120 - 4) / 100 = 1.16
    # PoS B = 0.6

    features = pd.DataFrame({'Group': np.concatenate([group_a_feat, group_b_feat])})
    cate = np.concatenate([group_a_cate, group_b_cate])

    # Shuffle data to ensure tree sees it naturally
    idx = np.arange(len(cate))
    np.random.seed(42)
    np.random.shuffle(idx)
    features = features.iloc[idx].reset_index(drop=True)
    cate = cate[idx]

    inductor = RuleInductor(max_depth=1, min_samples_leaf=10) # Depth 1 to split on Group 0 vs 1
    inductor.fit(features, cate)

    result = inductor.induce_rules_with_data(features, cate)

    # Assertions
    assert len(result.new_criteria) > 0
    rule = result.new_criteria[0]

    # Check Rationale
    assert "Optimizes Mean Effect" in rule.rationale
    assert "CATE" in rule.rationale

    # Check Optimized Value
    # Should be close to 1.16
    assert result.optimized_pos == pytest.approx(1.16, abs=0.01)

    # Check Rule Logic
    # Should pick Group B, which is Group > 0.5 (since Group values are 0 and 1)
    assert rule.feature == "Group"
    assert rule.operator == ">"
    assert rule.value == 0.5

def test_rule_inductor_handles_negative_effects() -> None:
    """
    Verifies robust handling when all effects are negative.
    Optimization should pick the 'least bad' or max mean effect.
    """
    n = 100
    features = pd.DataFrame({'X': np.random.rand(n)})
    # All negative effects.
    # Group X < 0.5: Mean -0.5
    # Group X >= 0.5: Mean -2.0
    cate = np.zeros(n)
    mask = features['X'] < 0.5
    cate[mask] = -0.5
    cate[~mask] = -2.0
    # Add noise
    cate += np.random.normal(0, 0.01, size=n)

    inductor = RuleInductor(max_depth=1, min_samples_leaf=10)
    inductor.fit(features, cate)

    result = inductor.induce_rules_with_data(features, cate)

    # Should pick X < 0.5 (Mean ~ -0.5)
    assert result.optimized_pos == pytest.approx(-0.5, abs=0.1)
    assert result.optimized_pos > -1.0

    # Ensure it didn't default to -1.0 or 0.0 from initialization issues
    assert result.optimized_pos != -1.0
