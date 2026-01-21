# Copyright (C) 2026 CoReason
# Licensed under the Prosperity Public License 3.0.0

from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from coreason_inference.analysis.latent import LatentMiner
from coreason_inference.analysis.virtual_simulator import VirtualSimulator
from coreason_inference.schema import CausalGraph, CausalNode, InterventionResult, ProtocolRule, RefutationStatus


@pytest.fixture
def mock_miner() -> MagicMock:
    miner = MagicMock(spec=LatentMiner)
    # Default behavior: generate a dataframe with columns A and B
    # Note: In new implementation, batch size is 5x target.
    # The miner should return new data each call ideally, or at least respect the call count.
    # For simple mocks, return_value is static.
    miner.generate.return_value = pd.DataFrame(
        {"A": [10, 20, 30, 40, 50], "B": [1.0, 2.0, 3.0, 4.0, 5.0], "C": [100, 100, 100, 100, 100]}
    )
    miner.feature_names = ["A", "B", "C"]
    return miner


@pytest.fixture
def simulator() -> VirtualSimulator:
    return VirtualSimulator()


def test_generate_synthetic_cohort_no_rules(simulator: VirtualSimulator, mock_miner: MagicMock) -> None:
    """Test generation without filtering."""
    # Adaptive sampling asks for 5x target. Target=5, asks for 25.
    cohort = simulator.generate_synthetic_cohort(mock_miner, n_samples=5)

    # Verify call with 5x batch
    mock_miner.generate.assert_called_with(25)
    # Mock returns 5 rows. Loop continues until max_retries or target met.
    # If mock returns 5 rows every time:
    # Batch 1: 5. Total 5. Target 5. Stop.
    assert len(cohort) == 5
    assert list(cohort.columns) == ["A", "B", "C"]


def test_generate_synthetic_cohort_miner_empty(simulator: VirtualSimulator, mock_miner: MagicMock) -> None:
    """Test when miner returns empty DataFrame."""
    mock_miner.generate.return_value = pd.DataFrame()
    # Mock miner should also have feature_names to test column preservation
    mock_miner.feature_names = ["A", "B", "C"]

    cohort = simulator.generate_synthetic_cohort(mock_miner, n_samples=5)

    assert cohort.empty
    # New logic: loop retries 10 times.
    assert mock_miner.generate.call_count == 10
    # Check preserved columns
    assert list(cohort.columns) == ["A", "B", "C"]


def test_generate_synthetic_cohort_with_filtering(simulator: VirtualSimulator, mock_miner: MagicMock) -> None:
    """Test generation with inclusion criteria."""
    rules = [
        ProtocolRule(feature="A", operator=">", value=20.0, rationale="Test"),
        ProtocolRule(feature="B", operator="<=", value=4.0, rationale="Test"),
    ]

    # Mock behavior: 5 rows provided.
    # Filter: A>20 -> 30, 40, 50. B<=4 -> 3.0, 4.0.
    # 2 survivors per batch.
    # Target 5.
    # Batch 1: 2 survivors. Total 2.
    # Batch 2: 2 survivors. Total 4.
    # Batch 3: 2 survivors. Total 6. Stop.
    # Final: 6 rows, trimmed to 5.

    cohort = simulator.generate_synthetic_cohort(mock_miner, n_samples=5, rules=rules)

    assert len(cohort) == 5
    # Since all batches are identical [30, 40], the result is [30, 40, 30, 40, 30]
    assert cohort["A"].tolist() == [30, 40, 30, 40, 30]

    # Check if miner was called multiple times (expected 3)
    assert mock_miner.generate.call_count == 3


def test_generate_synthetic_cohort_empty_result(simulator: VirtualSimulator, mock_miner: MagicMock) -> None:
    """Test when rules filter out everyone."""
    rules = [ProtocolRule(feature="A", operator=">", value=100.0, rationale="Impossible")]
    mock_miner.feature_names = ["A", "B", "C"]

    cohort = simulator.generate_synthetic_cohort(mock_miner, n_samples=5, rules=rules)

    assert cohort.empty
    # Should contain columns even if empty
    assert list(cohort.columns) == ["A", "B", "C"]
    # Should have retried max times
    assert mock_miner.generate.call_count == 10


def test_generate_synthetic_cohort_missing_feature(simulator: VirtualSimulator, mock_miner: MagicMock) -> None:
    """Test handling of rules for non-existent features."""
    rules = [ProtocolRule(feature="Z", operator=">", value=0.0, rationale="Missing")]

    # Should skip the rule and return full cohort
    # Mock returns 5. Target 5. Done in 1 batch.
    cohort = simulator.generate_synthetic_cohort(mock_miner, n_samples=5, rules=rules)

    assert len(cohort) == 5
    assert mock_miner.generate.call_count == 1


def test_generate_synthetic_cohort_miner_failure(simulator: VirtualSimulator, mock_miner: MagicMock) -> None:
    """Test error handling when miner fails."""
    mock_miner.generate.side_effect = ValueError("VAE Error")

    with pytest.raises(ValueError, match="VAE Error"):
        simulator.generate_synthetic_cohort(mock_miner, n_samples=5)


def test_apply_rules_operators(simulator: VirtualSimulator) -> None:
    """Test all supported operators in _apply_rules directly."""
    df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})

    # >
    res = simulator._apply_rules(df, [ProtocolRule(feature="val", operator=">", value=3, rationale="")])
    assert res["val"].tolist() == [4, 5]

    # >=
    res = simulator._apply_rules(df, [ProtocolRule(feature="val", operator=">=", value=3, rationale="")])
    assert res["val"].tolist() == [3, 4, 5]

    # <
    res = simulator._apply_rules(df, [ProtocolRule(feature="val", operator="<", value=3, rationale="")])
    assert res["val"].tolist() == [1, 2]

    # <=
    res = simulator._apply_rules(df, [ProtocolRule(feature="val", operator="<=", value=3, rationale="")])
    assert res["val"].tolist() == [1, 2, 3]

    # ==
    res = simulator._apply_rules(df, [ProtocolRule(feature="val", operator="==", value=3, rationale="")])
    assert res["val"].tolist() == [3]

    # !=
    res = simulator._apply_rules(df, [ProtocolRule(feature="val", operator="!=", value=3, rationale="")])
    assert res["val"].tolist() == [1, 2, 4, 5]

    # Invalid operator
    res = simulator._apply_rules(df, [ProtocolRule(feature="val", operator="??", value=3, rationale="")])
    assert len(res) == 5  # Should ignore invalid operator


def test_scan_safety(simulator: VirtualSimulator) -> None:
    """Test safety scan graph traversal."""
    # A -> B -> C (Risk)
    # A -> D (Safe)
    nodes = [
        CausalNode(id="A", codex_concept_id=1, is_latent=False),
        CausalNode(id="B", codex_concept_id=2, is_latent=False),
        CausalNode(id="C", codex_concept_id=3, is_latent=False),
        CausalNode(id="D", codex_concept_id=4, is_latent=False),
    ]
    edges = [("A", "B"), ("B", "C"), ("A", "D")]
    graph = CausalGraph(nodes=nodes, edges=edges, loop_dynamics=[], stability_score=1.0)

    # Risk found
    flags = simulator.scan_safety(graph, treatment="A", adverse_outcomes=["C"])
    assert len(flags) == 1
    assert "Risk path detected: A -> B -> C" in flags[0]

    # No risk found (disconnected or different path)
    flags_safe = simulator.scan_safety(graph, treatment="A", adverse_outcomes=["Z"])  # Z not in graph
    assert len(flags_safe) == 0

    # No path
    flags_no_path = simulator.scan_safety(graph, treatment="D", adverse_outcomes=["B"])
    assert len(flags_no_path) == 0

    # Test treatment not in graph
    flags_missing_treat = simulator.scan_safety(graph, treatment="MISSING", adverse_outcomes=["A"])
    assert len(flags_missing_treat) == 0


def test_scan_safety_path_error(simulator: VirtualSimulator) -> None:
    """Test error handling during path finding."""
    # Ensure treatment 'A' is in the graph by adding an edge, so we pass the "not in G" check.
    nodes = [
        CausalNode(id="A", codex_concept_id=1, is_latent=False),
        CausalNode(id="B", codex_concept_id=2, is_latent=False),
    ]
    # Edge A->B ensures A is in G
    graph = CausalGraph(nodes=nodes, edges=[("A", "B")], loop_dynamics=[], stability_score=0.0)

    # We need to mock nx.has_path to return True but shortest_path to raise
    with patch("networkx.has_path", return_value=True):
        with patch("networkx.shortest_path", side_effect=Exception("Path error")):
            flags = simulator.scan_safety(graph, "A", ["A"])
            assert len(flags) == 0  # Should catch exception and not add flag


@patch("coreason_inference.analysis.virtual_simulator.CausalEstimator")
def test_simulate_trial(mock_estimator_cls: MagicMock, simulator: VirtualSimulator) -> None:
    """Test simulation calls estimator correctly."""

    # Mock instance
    mock_instance = mock_estimator_cls.return_value
    mock_instance.estimate_effect.return_value = InterventionResult(
        patient_id="POPULATION_ATE",
        intervention="do(Drug)",
        counterfactual_outcome=0.5,
        confidence_interval=(0.4, 0.6),
        refutation_status=RefutationStatus.PASSED,
    )

    cohort = pd.DataFrame({"Drug": [0, 1], "Outcome": [0.1, 0.6], "Conf": [1, 1]})

    result = simulator.simulate_trial(cohort, "Drug", "Outcome", ["Conf"])

    mock_estimator_cls.assert_called_once_with(cohort)
    mock_instance.estimate_effect.assert_called_once_with(
        treatment="Drug", outcome="Outcome", confounders=["Conf"], method="forest", num_simulations=5
    )
    assert result.counterfactual_outcome == 0.5


def test_simulate_trial_validation(simulator: VirtualSimulator) -> None:
    """Test validation in simulate_trial."""
    cohort = pd.DataFrame({"Drug": [0, 1]})

    # Missing outcome
    with pytest.raises(ValueError, match="Outcome 'Y' not found"):
        simulator.simulate_trial(cohort, "Drug", "Y", [])

    # Missing treatment
    with pytest.raises(ValueError, match="Treatment 'Z' not found"):
        simulator.simulate_trial(cohort, "Z", "Drug", [])

    # Empty cohort
    with pytest.raises(ValueError, match="empty cohort"):
        simulator.simulate_trial(pd.DataFrame(), "Drug", "Y", [])


@patch("coreason_inference.analysis.virtual_simulator.CausalEstimator")
def test_simulate_trial_failure(mock_estimator_cls: MagicMock, simulator: VirtualSimulator) -> None:
    """Test exception propagation from estimator."""
    mock_instance = mock_estimator_cls.return_value
    mock_instance.estimate_effect.side_effect = Exception("Estimator Failed")

    cohort = pd.DataFrame({"Drug": [0, 1], "Outcome": [0, 1]})

    with pytest.raises(Exception, match="Estimator Failed"):
        simulator.simulate_trial(cohort, "Drug", "Outcome", [])


def test_apply_rules_error(simulator: VirtualSimulator) -> None:
    """Test exception handling within rule application loop."""
    df = pd.DataFrame({"val": [1, 2, 3]})

    # We use a mocked BadObj to trigger exception during comparison in _apply_rules
    class BadObj:
        def __lt__(self, other: Any) -> Any:
            raise Exception("Comparison Error")

        def __gt__(self, other: Any) -> Any:
            raise Exception("Comparison Error")

        def __le__(self, other: Any) -> Any:
            raise Exception("Comparison Error")

        def __ge__(self, other: Any) -> Any:
            raise Exception("Comparison Error")

    # To trigger the except block in _apply_rules, we patch pandas Series comparison.
    with patch("pandas.Series.__gt__", side_effect=Exception("Filter Error")):
        # Only applies to Series > ...
        res = simulator._apply_rules(df, [ProtocolRule(feature="val", operator=">", value=0, rationale="")])
        # It should log error and return original (or current) data
        assert len(res) == 3
