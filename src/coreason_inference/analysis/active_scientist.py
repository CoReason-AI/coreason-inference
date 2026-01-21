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
import numpy as np
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
        self.cpdag: np.ndarray | None = None
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

        # Store adjacency matrix
        # causal-learn represents graph as numpy array where:
        # matrix[i, j] = Endpoint at j from i
        self.cpdag = cg.G.graph

    def propose_experiments(self) -> List[ExperimentProposal]:
        """
        Identifies undirected edges in the CPDAG and proposes the BEST experiment
        to resolve directionality, prioritizing nodes involved in the most ambiguities.

        Returns:
            List[ExperimentProposal]: A list containing the optimal experiment(s).
        """
        if self.cpdag is None:
            raise ValueError("Model not fitted. Call fit() first.")

        n_nodes = len(self.labels)
        undirected_edges: List[Tuple[int, int]] = []
        node_degrees: Dict[int, int] = {i: 0 for i in range(n_nodes)}

        # 1. Identify Undirected Edges and Build Ambiguity Map
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                end_j = self.cpdag[i, j]  # Endpoint at j
                end_i = self.cpdag[j, i]  # Endpoint at i

                # Standard causal-learn: Tail (-1) at both ends means Undirected
                if end_j != 0 and end_i != 0:  # Edge exists
                    if end_j == end_i:  # Symmetric endpoints -> Undirected
                        undirected_edges.append((i, j))
                        node_degrees[i] += 1
                        node_degrees[j] += 1

        if not undirected_edges:
            logger.info("No undirected edges found. CPDAG is fully oriented (DAG).")
            return []

        # 2. Select Best Target (Max Degree Heuristic)
        # Sort nodes by degree (descending)
        sorted_nodes = sorted(node_degrees.items(), key=lambda item: item[1], reverse=True)
        best_node_idx, max_degree = sorted_nodes[0]

        if max_degree == 0:
            return []  # pragma: no cover

        target_var = self.labels[best_node_idx]

        logger.info(
            f"Selected Optimal Experiment: Intervene on '{target_var}' "
            f"(Resolves {max_degree} adjacent undirected edges)."
        )

        # 3. Construct Proposal
        rationale = (
            f"Target '{target_var}' is centrally located in the ambiguity graph "
            f"(Degree: {max_degree}). Intervening here maximizes potential for edge orientation."
        )

        proposal = ExperimentProposal(
            target=target_var,
            action="Intervention_Knockout",  # Generic biological action
            confidence_gain="High",
            rationale=rationale,
        )

        return [proposal]
