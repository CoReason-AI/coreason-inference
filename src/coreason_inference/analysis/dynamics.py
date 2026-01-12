# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torchdiffeq import odeint

from coreason_inference.schema import CausalGraph, CausalNode, LoopType
from coreason_inference.utils.logger import logger


class ODEFunc(nn.Module):  # type: ignore
    """
    Neural ODE function approximating dy/dt = f(y).
    This module uses a linear layer to allow for straightforward Jacobian extraction
    to identify feedback loops, serving as a foundation for cyclic discovery.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        # Use a single linear layer to capture dependencies (Jacobian ~ Weight Matrix)
        # In a more complex version, this could be an MLP.
        # Bias is enabled to handle data shifted by StandardScaler (y' = Wy + b)
        self.linear = nn.Linear(input_dim, input_dim, bias=True)
        # Initialize with larger weights to help convergence to interaction terms (target ~ 1.0)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.3)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates the derivative dy/dt at time t.
        Neural ODEs often ignore 't' if the system is autonomous.
        """
        return self.linear(y)


class DynamicsEngine:
    """
    The Dynamics Engine uses Neural ODEs to model system dynamics and discover feedback loops.
    """

    def __init__(self, learning_rate: float = 0.05, epochs: int = 500, method: str = "dopri5"):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.method = method
        self.model: Optional[ODEFunc] = None
        self.variable_names: List[str] = []
        self.scaler: Optional[StandardScaler] = None

    def fit(self, data: pd.DataFrame, time_col: str, variable_cols: List[str]) -> None:
        """
        Fits a Neural ODE to the time-series data.

        Args:
            data: DataFrame containing time-series data.
            time_col: Name of the column representing time.
            variable_cols: List of column names representing the variables (nodes).
        """
        if data.empty:
            raise ValueError("Input data is empty.")

        if data.isnull().values.any():
            raise ValueError("Input data contains NaN values.")

        logger.info(f"Fitting DynamicsEngine to {len(variable_cols)} variables.")
        self.variable_names = variable_cols

        # Prepare Data
        # Sort by time just in case
        df = data.sort_values(by=time_col)

        # Check for sufficient data points
        if len(df) < 2:
            raise ValueError("Insufficient data points for time-series analysis (minimum 2 required).")

        t_raw = df[time_col].values
        y_raw = df[variable_cols].values

        # Scaling: Normalize variables to mean 0, std 1
        # This is critical for convergence when variables have different scales.
        self.scaler = StandardScaler()
        y_scaled = self.scaler.fit_transform(y_raw)

        t = torch.tensor(t_raw, dtype=torch.float32)
        y = torch.tensor(y_scaled, dtype=torch.float32)

        # Normalize time to start at 0
        t = t - t[0]

        dim = len(variable_cols)
        self.model = ODEFunc(dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training Loop
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Integrate from initial state y[0] across time points t
            pred_y = odeint(self.model, y[0], t, method=self.method)

            # pred_y shape: (len(t), batch_size=1, dim) -> (len(t), dim)
            loss = torch.mean((pred_y - y) ** 2)
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {loss.item()}")

        logger.info(f"Training complete. Final Loss: {loss.item()}")

    def discover_loops(self, threshold: float = 0.1) -> CausalGraph:
        """
        Analyzes the learned dynamics to discover feedback loops.

        Args:
            threshold: Minimum weight magnitude to consider an edge exists.

        Returns:
            CausalGraph: The discovered graph structure with loops.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        # Extract weights as adjacency matrix
        weights = self.model.linear.weight.detach().numpy()

        # Log weights for debugging
        logger.debug(f"Learned Interaction Matrix (Weights):\n{weights}")

        nodes = []
        edges = []
        loop_dynamics = []

        # Create Nodes
        for name in self.variable_names:
            # Placeholder for codex_concept_id (hashing name for uniqueness in this atomic unit)
            concept_id = abs(hash(name)) % 100000
            nodes.append(CausalNode(id=name, codex_concept_id=concept_id, is_latent=False))

        # Detect Edges and Loops
        # 1. Identify Edges
        adj_matrix = np.zeros_like(weights)
        for i in range(len(self.variable_names)):  # Target
            for j in range(len(self.variable_names)):  # Source
                w = weights[i, j]
                if abs(w) > threshold:
                    edges.append((self.variable_names[j], self.variable_names[i]))
                    adj_matrix[i, j] = w

        # 2. simple loop detection (Length 2 and Self-loops)

        # Self-loops (diagonal)
        for i in range(len(self.variable_names)):
            w = adj_matrix[i, i]
            if w != 0:
                # If negative, it's negative feedback (stabilizing)
                loop_type = LoopType.NEGATIVE_FEEDBACK if w < 0 else LoopType.POSITIVE_FEEDBACK
                loop_dynamics.append(
                    {"path": [self.variable_names[i], self.variable_names[i]], "type": loop_type.value}
                )

        # Length-2 loops
        for i in range(len(self.variable_names)):
            for j in range(i + 1, len(self.variable_names)):
                # Check if i->j and j->i
                w_ij = adj_matrix[i, j]  # j to i
                w_ji = adj_matrix[j, i]  # i to j

                if w_ij != 0 and w_ji != 0:
                    # Determine loop type based on product of signs
                    # If product is negative -> Negative Feedback (Stabilizing)
                    # If product is positive -> Positive Feedback (Runaway)
                    is_negative = (w_ij * w_ji) < 0
                    loop_type = LoopType.NEGATIVE_FEEDBACK if is_negative else LoopType.POSITIVE_FEEDBACK

                    loop_dynamics.append(
                        {
                            "path": [self.variable_names[i], self.variable_names[j], self.variable_names[i]],
                            "type": loop_type.value,
                        }
                    )

        # Stability Score (Eigenvalues of Jacobian)
        try:
            eigenvalues = np.linalg.eigvals(weights)
            max_real_eig = np.max(np.real(eigenvalues))
            stability_score = float(max_real_eig)
        except Exception:  # pragma: no cover
            # Fallback for numerical errors
            stability_score = 0.0

        return CausalGraph(nodes=nodes, edges=edges, loop_dynamics=loop_dynamics, stability_score=stability_score)
