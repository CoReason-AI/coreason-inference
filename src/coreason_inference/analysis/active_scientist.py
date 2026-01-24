# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import Any, List

import networkx as nx
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc

from coreason_inference.schema import ExperimentProposal
from coreason_inference.utils.logger import logger

# Graph Endpoint Constants (based on causal-learn conventions)
# Matrix M[i, j] represents the endpoint at j for the edge between i and j.
ENDPOINT_NULL = 0
ENDPOINT_TAIL = -1  # Tail (-)
ENDPOINT_HEAD = 1  # Arrowhead (>)


class ActiveScientist:
    """
    Identifies causal ambiguity (Markov Equivalence Classes) and proposes experiments
    to resolve them.
    """

    def __init__(self) -> None:
        self.cpdag: np.ndarray[Any, Any] | None = None
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
        to resolve directionality using the Max-Degree Heuristic.

        This heuristic selects the node with the highest number of incident undirected
        edges (Degree), approximating high information gain by targeting central
        uncertainty nodes (Hubs).

        Returns:
            List[ExperimentProposal]: A list containing the optimal experiment(s).
        """
        if self.cpdag is None:
            raise ValueError("Model not fitted. Call fit() first.")

        n_nodes = len(self.labels)

        # Calculate Undirected Degree for each node
        # Undirected Edge (i, j): M[i, j] == TAIL and M[j, i] == TAIL

        degrees = np.zeros(n_nodes, dtype=int)
        has_undirected = False

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if self.cpdag[i, j] == ENDPOINT_TAIL and self.cpdag[j, i] == ENDPOINT_TAIL:
                    degrees[i] += 1
                    degrees[j] += 1
                    has_undirected = True

        if not has_undirected:
            logger.info("No undirected edges found. CPDAG is fully oriented (DAG).")
            return []

        best_node_idx = int(np.argmax(degrees))
        max_degree = degrees[best_node_idx]

        target_var = self.labels[best_node_idx]

        logger.info(
            f"Selected Optimal Experiment via Max-Degree Heuristic: "
            f"Intervene on '{target_var}' (Undirected Degree: {max_degree})."
        )

        rationale = (
            f"Target '{target_var}' selected by Max-Degree Heuristic. "
            f"It has the highest number of incident undirected edges ({max_degree}), "
            f"indicating it is a central node where intervention maximizes information gain."
        )

        proposal = ExperimentProposal(
            target=target_var,
            action="Intervention_Knockout",
            confidence_gain="High",
            rationale=rationale,
        )

        return [proposal]
