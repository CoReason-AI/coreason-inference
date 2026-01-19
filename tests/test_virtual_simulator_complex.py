# Copyright (C) 2026 CoReason
# Licensed under the Prosperity Public License 3.0.0

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from coreason_inference.analysis.virtual_simulator import VirtualSimulator
from coreason_inference.schema import CausalGraph, CausalNode, ProtocolRule


class TestComplexVirtualSimulator:
    @pytest.fixture
    def simulator(self) -> VirtualSimulator:
        return VirtualSimulator()

    def test_safety_scan_cyclic_and_multi_path(self, simulator: VirtualSimulator) -> None:
        """
        Tests safety scan on a graph with feedback loops and multiple paths.
        Structure:
        T -> M1 -> A (Path 1)
        T -> M2 -> A (Path 2)
        M1 -> M2 -> M1 (Cycle)
        T -> Safe
        """
        nodes = [
            CausalNode(id="T", codex_concept_id=1, is_latent=False),
            CausalNode(id="M1", codex_concept_id=2, is_latent=False),
            CausalNode(id="M2", codex_concept_id=3, is_latent=False),
            CausalNode(id="A", codex_concept_id=4, is_latent=False),
            CausalNode(id="Safe", codex_concept_id=5, is_latent=False),
        ]
        edges = [
            ("T", "M1"),
            ("M1", "A"),
            ("T", "M2"),
            ("M2", "A"),
            ("M1", "M2"),
            ("M2", "M1"),  # Cycle
            ("T", "Safe"),
        ]
        graph = CausalGraph(nodes=nodes, edges=edges, loop_dynamics=[], stability_score=1.0)

        flags = simulator.scan_safety(graph, "T", ["A"])
        assert len(flags) > 0
        # It should find at least one path. nx.shortest_path finds one.
        # It handles cycles by finding the shortest path (e.g. T->M1->A or T->M2->A).
        assert "Risk path detected" in flags[0]

    def test_safety_scan_disconnected_components(self, simulator: VirtualSimulator) -> None:
        """
        Tests safety scan when treatment and outcome are in disjoint subgraphs.
        Component 1: T -> M
        Component 2: X -> A
        """
        nodes = [
            CausalNode(id="T", codex_concept_id=1, is_latent=False),
            CausalNode(id="M", codex_concept_id=2, is_latent=False),
            CausalNode(id="X", codex_concept_id=3, is_latent=False),
            CausalNode(id="A", codex_concept_id=4, is_latent=False),
        ]
        edges = [("T", "M"), ("X", "A")]
        graph = CausalGraph(nodes=nodes, edges=edges, loop_dynamics=[], stability_score=1.0)

        flags = simulator.scan_safety(graph, "T", ["A"])
        assert len(flags) == 0

    def test_synthetic_cohort_complex_rules(self, simulator: VirtualSimulator) -> None:
        """
        Tests filtering with overlapping and exclusionary rules.
        """
        mock_miner = MagicMock()
        # Create data with mix of values
        data = pd.DataFrame(
            {
                "Age": [20, 30, 40, 50, 60],
                "BMI": [20, 25, 30, 35, 40],
                "Risk": [0, 0, 1, 1, 1],
            }
        )
        mock_miner.generate.return_value = data

        # Rule: Age >= 30 AND Age <= 50 AND BMI != 30
        rules = [
            ProtocolRule(feature="Age", operator=">=", value=30, rationale=""),
            ProtocolRule(feature="Age", operator="<=", value=50, rationale=""),
            ProtocolRule(feature="BMI", operator="!=", value=30, rationale=""),
        ]

        # Filter Logic:
        # Age >= 30: [30, 40, 50, 60] (indices 1, 2, 3, 4)
        # Age <= 50: [30, 40, 50] (indices 1, 2, 3)
        # BMI != 30: 30 corresponds to index 2 (Age 40). So remove index 2.
        # Result: indices 1 (Age 30, BMI 25), 3 (Age 50, BMI 35)

        cohort = simulator.generate_synthetic_cohort(mock_miner, n_samples=5, rules=rules)
        assert len(cohort) == 2
        assert cohort["Age"].tolist() == [30, 50]

    @patch("coreason_inference.analysis.virtual_simulator.CausalEstimator")
    def test_simulate_trial_constant_features(self, mock_est_cls: MagicMock, simulator: VirtualSimulator) -> None:
        """
        Tests behavior when rules filter data such that a confounder becomes constant.
        Uses mocks to simulate estimator failure behavior in this edge case.
        """
        mock_est_instance = mock_est_cls.return_value

        # Generated cohort where "Gender" is always 1
        cohort = pd.DataFrame(
            {
                "Treatment": [0, 1, 0, 1],
                "Outcome": [1.0, 1.2, 1.1, 1.3],
                "Gender": [1, 1, 1, 1],
            }
        )

        # We simulate the estimator failing due to constant column (a common issue in DML/Forests)
        mock_est_instance.estimate_effect.side_effect = ValueError("Constant column")

        with pytest.raises(ValueError, match="Constant column"):
            simulator.simulate_trial(cohort, "Treatment", "Outcome", ["Gender"])
