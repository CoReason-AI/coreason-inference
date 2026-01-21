# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from coreason_inference.schema import ExperimentProposal
from coreason_inference.utils.logger import logger


class ActiveScientist:
    """
    Identifies causal ambiguity (Markov Equivalence Classes) and proposes experiments
    to resolve them.
    """

    def __init__(self) -> None:
        self.cpdag: pd.DataFrame | None = None
        self.graph: nx.Graph | None = None
        self.labels: List[str] = []

    def fit(self, data: pd.DataFrame) -> None:
        """
        Discovers the Markov Equivalence Class (CPDAG) from observational data using the PC algorithm.

        Args:
            data: DataFrame containing observational data.
        """
        if data.empty:
            raise ValueError("Input data is empty.")

        self.labels = list(data.columns)
        logger.info(f"Fitting ActiveScientist (PC Algorithm) to {len(self.labels)} variables.")

        # Run PC Algorithm
        try:
            cg = pc(data.values, alpha=0.05, verbose=False)
        except Exception as e:
            logger.error(f"PC Algorithm failed: {e}")
            raise e

        # Extract graph structure (Adjacency Matrix)
        self.cpdag = cg.G.graph

    def propose_experiments(self) -> List[ExperimentProposal]:
        """
        Identifies undirected edges in the CPDAG and proposes the single best experiment
        that resolves the most ambiguity (Information Gain Heuristic).

        Returns:
            List[ExperimentProposal]: A list containing the optimal proposal.
        """
        if self.cpdag is None:
            raise ValueError("Model not fitted. Call fit() first.")

        undirected_edges = self._get_undirected_edges()

        if not undirected_edges:
            logger.info("No ambiguous edges found. No experiments needed.")
            return []

        # Calculate heuristic score for each candidate node
        # Score = Number of undirected edges incident to the node.
        # This approximates Information Gain (Centrality in the ambiguity graph).
        candidates: Dict[int, int] = {}
        for u, v in undirected_edges:
            candidates[u] = candidates.get(u, 0) + 1
            candidates[v] = candidates.get(v, 0) + 1

        # Select candidate with max score
        best_node_idx = -1
        max_score = -1

        for node_idx, score in candidates.items():
            if score > max_score:
                max_score = score
                best_node_idx = node_idx
            elif score == max_score:
                # Tie-breaking: lower index for determinism, or random.
                # Here we stick to lower index (implied by loop order if dict is ordered, but explicit check is better)
                if node_idx < best_node_idx:
                    best_node_idx = node_idx

        if best_node_idx == -1:
            return []

        target_var = self.labels[best_node_idx]
        logger.info(f"Selected optimal intervention target: {target_var} (Resolves {max_score} edges)")

        rationale = (
            f"Ambiguity detected in {len(undirected_edges)} edges. "
            f"Intervening on {target_var} is calculated to resolve {max_score} ambiguous edges "
            f"(Heuristic: Degree Centrality)."
        )

        proposal = ExperimentProposal(
            target=target_var,
            action="Intervention_Knockout",
            confidence_gain="High",
            rationale=rationale,
        )

        return [proposal]

    def _get_undirected_edges(self) -> List[Tuple[int, int]]:
        """
        Identify all undirected edges in the CPDAG.
        Returns a list of (i, j) tuples where i < j.
        """
        if self.cpdag is None:
            return []

        undirected = []
        n_nodes = len(self.labels)

        # Traverse upper triangle
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                end_j = self.cpdag[i, j]  # Endpoint at j
                end_i = self.cpdag[j, i]  # Endpoint at i

                # Valid edge (endpoints != 0) and Symmetric endpoints (e.g. Tail-Tail)
                if end_j != 0 and end_i != 0:
                    if end_j == end_i:
                        undirected.append((i, j))

        return undirected
