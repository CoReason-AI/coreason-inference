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
from causallearn.graph.Endpoint import Endpoint
from causallearn.search.ConstraintBased.PC import pc

from coreason_inference.schema import ExperimentProposal
from coreason_inference.utils.logger import logger


class ActiveScientist:
    """The Active Scientist: Resolves causal ambiguity.

    Identifies the Markov Equivalence Class (all valid graphs fitting the data)
    and proposes physical experiments (Interventions) to resolve directionality
    where the data is ambiguous.

    Implements the Max-Degree Heuristic to approximate Information Gain.
    """

    def __init__(self) -> None:
        """Initializes the Active Scientist."""
        self.cpdag: np.ndarray[Any, Any] | None = None
        self.graph: nx.Graph | None = None
        self.labels: List[str] = []

    def fit(self, data: pd.DataFrame) -> None:
        """Discovers the Markov Equivalence Class (CPDAG) from observational data.

        Uses the PC (Peter-Clark) algorithm.

        Args:
            data: DataFrame containing observational data.

        Raises:
            ValueError: If input data is empty.
            Exception: If the PC algorithm execution fails.
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
        """Identifies undirected edges and proposes the BEST experiment.

        Uses the Max-Degree Heuristic to select the node with the highest number
        of incident undirected edges (Degree), approximating high information gain.

        Returns:
            List[ExperimentProposal]: A list containing the optimal experiment(s).

        Raises:
            ValueError: If the model has not been fitted via `fit()`.
        """
        if self.cpdag is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Calculate Undirected Degree for each node
        # Undirected Edge (i, j): M[i, j] == TAIL and M[j, i] == TAIL
        # Vectorized check for undirected edges
        is_tail_at_j = self.cpdag == Endpoint.TAIL.value
        is_tail_at_i = self.cpdag.T == Endpoint.TAIL.value
        undirected_matrix = is_tail_at_j & is_tail_at_i

        # Since matrix is symmetric for undirected edges, sum along rows gives total undirected degree
        degrees = np.sum(undirected_matrix, axis=1)

        if degrees.sum() == 0:
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
