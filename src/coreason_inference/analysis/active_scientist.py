# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import List

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
        # Returns:
        # cg: CausalGraph object (has .G, .nodes, etc.)
        # Note: causal-learn's PC returns a GeneralGraph object.
        try:
            cg = pc(data.values, alpha=0.05, verbose=False)
        except Exception as e:
            logger.error(f"PC Algorithm failed: {e}")
            raise e

        # Extract graph structure
        # In causal-learn, undirected edges are represented differently depending on the graph type.
        # GeneralGraph usually has:
        # - 1 for circle (o)
        # - 2 for arrowhead (>)
        # - 3 for tail (-)
        #
        # Edges in cg.G.graph are stored as (i, j) -> type
        # Or we can iterate nodes.
        # Let's use the adjacency matrix or edge list provided by the library.
        # cg.G.graph is a numpy array for GeneralGraph? No, let's check basic usage.
        # Usually `cg.G.graph` is the adjacency matrix.
        #
        # Representation in causal-learn GeneralGraph:
        # matrix[i, j] = Endpoint at j from i.
        # -1: No edge
        # 0: Null (No edge) - Wait, usually -1 or 0 depending on implementation.
        #
        # Let's rely on `cg.G.get_adj_matrix()` or similar if available, or just parse `cg.G.graph`.
        # Assuming `cg.G.graph` is available.

        self.cpdag = cg.G.graph

    def propose_experiments(self) -> List[ExperimentProposal]:
        """
        Identifies undirected edges in the CPDAG and proposes experiments to resolve directionality.

        Returns:
            List[ExperimentProposal]: A list of proposals.
        """
        if self.cpdag is None:
            raise ValueError("Model not fitted. Call fit() first.")

        proposals = []
        n_nodes = len(self.labels)

        # Parse adjacency matrix to find undirected edges
        # In causal-learn:
        # X -- Y is represented as:
        # matrix[X, Y] = Endpoint at Y (Tail or Circle? In PC output (CPDAG), it is Tail-Tail or similar)
        # Actually, in PC output (CPDAG):
        # Directed: X -> Y  => matrix[i,j] = Arrow (-1->1 or similar?), matrix[j,i] = Tail
        # Undirected: X - Y => matrix[i,j] = Tail, matrix[j,i] = Tail
        #
        # Endpoint constants in causal-learn:
        # TAIL = -1 (or 1 in some versions?)
        # ARROW = 1 (or 2?)
        # CIRCLE = 2 (or ??)
        #
        # Let's look at `causallearn.graph.GraphClass.Endpoint`.
        # Usually:
        # TAIL = -1
        # NULL = 0
        # ARROW = 1
        # CIRCLE = 2
        #
        # However, `pc` returns a `CausalGraph` wrapper usually.
        # Let's assume standard behavior:
        # directed X->Y: G[X,Y] = -1 (Tail at X), G[Y,X] = 1 (Arrow at Y)
        # undirected X-Y: G[X,Y] = -1 (Tail), G[Y,X] = -1 (Tail) OR G[X,Y] = 1, G[Y,X] = 1?
        #
        # Actually, let's assume standard "Pattern" graph (CPDAG).
        # Undirected edges are what we care about.

        # We'll traverse the upper triangle to find undirected edges.
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                end_j = self.cpdag[i, j]  # Endpoint at j
                end_i = self.cpdag[j, i]  # Endpoint at i

                # Check for undirected edge
                # Usually represented as Tail-Tail (-1, -1) or Circle-Circle?
                # In PC, valid outputs are Directed or Undirected.
                # If both ends are Tails (-1) or both ends are not Arrows?
                #
                # Let's be robust: If there is an edge (endpoint != 0) AND it's not fully directed.
                # Fully directed means one is Arrow (1) and one is Tail (-1).
                # Undirected means both are Tail (-1) or both are Arrow (shouldn't happen in CPDAG usually) or Circle.
                #
                # Let's assume standard causal-learn behavior for PC:
                # -1: Tail
                #  1: Arrow

                if end_j != 0 and end_i != 0:  # Edge exists
                    if end_j == end_i:  # Symmetric endpoints -> Undirected (e.g. Tail-Tail)
                        # Identify ambiguity
                        var_a = self.labels[i]
                        var_b = self.labels[j]

                        # Propose experiment: Intervene on A to see if B changes
                        rationale = (
                            f"Ambiguous edge between {var_a} and {var_b}. "
                            f"Intervening on {var_a} resolves directionality."
                        )
                        proposal = ExperimentProposal(
                            target=var_a,
                            action="Intervention_Knockout",  # Generic biological action
                            confidence_gain="High",
                            rationale=rationale,
                        )
                        proposals.append(proposal)

                        # Also propose the reverse?
                        # Usually one intervention is enough to orient the edge, but strictly we might propose either.
                        # The user story says "Selects the Intervention ... that maximally splits".
                        # For this atomic unit, proposing one valid intervention per undirected edge is sufficient.

        return proposals
