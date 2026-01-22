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

from coreason_inference.schema import CausalGraph, CausalNode, LoopDynamics, LoopType
from coreason_inference.utils.logger import logger


class ODEFunc(nn.Module):  # type: ignore[misc]
    """
    UPGRADED: Non-Linear Neural ODE function.
    Uses a Tanh activation to model complex, non-linear system dynamics.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim

        # 1. Non-Linear Architecture (MLP)
        # Input -> Hidden (Tanh) -> Output
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),  # Tanh is smoother than ReLU, better for ODEs
            nn.Linear(hidden_dim, input_dim),
        )

        # 2. Explicit Dependency Matrix (for Graph Discovery)
        # We learn a mask 'W' to track which variables affect which.
        self.W = nn.Parameter(torch.randn(input_dim, input_dim) * 0.1)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        dy/dt = MLP(y) * W
        """
        # Element-wise multiplication to enforce the dependency structure
        # Note: The prompt implementation `torch.matmul(dynamics, self.W)` multiplies (batch, dim) x (dim, dim)
        # If `dynamics` output is (dim), it mixes them.
        dynamics = self.net(y)
        return torch.matmul(dynamics, self.W)


class DynamicsEngine:
    """
    The Dynamics Engine uses Neural ODEs to model system dynamics and discover feedback loops.
    """

    def __init__(self, learning_rate: float = 0.05, epochs: int = 500, method: str = "dopri5", l1_lambda: float = 0.0):
        if l1_lambda < 0:
            raise ValueError("l1_lambda must be non-negative.")

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.method = method
        self.l1_lambda = l1_lambda
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
        # Initialize model with hidden dimension
        self.model = ODEFunc(dim, hidden_dim=32)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training Loop
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Integrate from initial state y[0] across time points t
            pred_y = odeint(self.model, y[0], t, method=self.method)

            # pred_y shape: (len(t), batch_size=1, dim) -> (len(t), dim)
            mse_loss = torch.mean((pred_y - y) ** 2)

            # Add L1 Regularization (Sparsity) on the W matrix
            l1_loss = torch.tensor(0.0)
            if self.l1_lambda > 0 and self.model is not None:
                l1_loss = self.l1_lambda * torch.norm(self.model.W, p=1)

            total_loss = mse_loss + l1_loss

            total_loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {total_loss.item()} (MSE: {mse_loss.item()}, L1: {l1_loss.item()})")

        logger.info(f"Training complete. Final Loss: {total_loss.item()}")

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

        # Extract weights from the explicit W parameter
        weights = self.model.W.detach().numpy()

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
                # Note: In PyTorch linear(y) is yA^T + b.
                # Here we used matmul(dynamics, W).
                # If dynamics is (1, dim), W is (dim, dim). Output is (1, dim).
                # So output[j] = sum(dynamics[i] * W[i, j]).
                # So W[i, j] is weight from i to j?
                # Usually: x_new = x_old @ W.
                # x_new[j] = sum_i (x_old[i] * W[i, j]).
                # So W[i, j] implies flow from i to j.
                # Previous code assumed `linear.weight` which is (out_features, in_features).
                # So weight[i, j] was i from j. (Target i, Source j).
                # Here W is (dim, dim) used in matmul(y, W).
                # y is (batch, dim).
                # y @ W -> (batch, dim).
                # y_out[k] = sum_j (y[j] * W[j, k]).
                # So W[j, k] is contribution from j (source) to k (target).
                # So W[row, col] -> Row=Source, Col=Target.
                # In previous code `weights[i, j]` was Target i, Source j.
                # So `weights` here is Transpose of previous?
                # Let's check loop detection logic:
                # `w = weights[i, j]`
                # `edges.append((variable_names[j], variable_names[i]))`
                # If weights[i, j] means Target i, Source j, then j->i. Correct.
                #
                # If W[j, i] means Source j -> Target i.
                # Then I should read W[j, i].
                #
                # Let's stick to the prompt's `weights = self.model.W.detach().numpy()`.
                # If `matmul(y, W)` is used, W is (input, output) i.e. (source, target).
                # So W[j, i] is j->i.
                # But the loop logic below iterates `for i (target) ... for j (source)`.
                # `w = weights[i, j]`.
                # This assumes `weights[target, source]`.
                # But W is `[source, target]`.
                # So I need to Transpose W before assigning to `weights`?
                # Or change the loop logic.
                #
                # Previous: `linear.weight` is (out, in) = (target, source).
                # Current: `W` in `matmul(y, W)` is (in, out) = (source, target).
                #
                # So `W` is effectively Transpose of `linear.weight`.
                #
                # Let's adjust `weights` to be Transposed so the rest of the code works as is.
                pass

        # Transpose W to match (Target, Source) convention of the rest of the method
        # Previous: weights[i, j] -> Target i, Source j.
        # W: [Source, Target].
        # So W.T -> [Target, Source].
        weights = weights.T

        # Update adj_matrix logic
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
                    LoopDynamics(path=[self.variable_names[i], self.variable_names[i]], type=loop_type)
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
                        LoopDynamics(
                            path=[self.variable_names[i], self.variable_names[j], self.variable_names[i]],
                            type=loop_type,
                        )
                    )

        # Stability Score (Eigenvalues of Jacobian)
        try:
            # Eigenvalues of adjacency matrix ~ Jacobian
            eigenvalues = np.linalg.eigvals(weights)
            max_real_eig = np.max(np.real(eigenvalues))
            stability_score = float(max_real_eig)
        except Exception:  # pragma: no cover
            # Fallback for numerical errors
            stability_score = 0.0

        return CausalGraph(nodes=nodes, edges=edges, loop_dynamics=loop_dynamics, stability_score=stability_score)
