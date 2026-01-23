# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import List, Optional, cast

import numpy as np
import pandas as pd
import torch
import torch.autograd.functional as F
import torch.nn as nn
import torch.optim as optim
from coreason_inference.schema import CausalGraph, CausalNode, LoopDynamics, LoopType
from coreason_inference.utils.logger import logger
from sklearn.preprocessing import StandardScaler
from torchdiffeq import odeint


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

    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 500,
        method: str = "dopri5",
        l1_lambda: float = 0.0,
        jacobian_lambda: float = 0.0,
    ):
        if l1_lambda < 0:
            raise ValueError("l1_lambda must be non-negative.")
        if jacobian_lambda < 0:
            raise ValueError("jacobian_lambda must be non-negative.")

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.method = method
        self.l1_lambda = l1_lambda
        self.jacobian_lambda = jacobian_lambda
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

        # Pre-define Jacobian function to avoid loop closure issues
        model_ref = self.model

        def func_for_jacobian(y_input: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, model_ref(torch.tensor(0.0), y_input))

        # Training Loop
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Integrate from initial state y[0] across time points t
            pred_y = odeint(self.model, y[0], t, method=self.method)

            # pred_y shape: (len(t), batch_size=1, dim) -> (len(t), dim)
            mse_loss = torch.mean((pred_y - y) ** 2)

            # Add L1 Regularization (Sparsity) on the W matrix
            l1_loss = torch.tensor(0.0)
            if self.l1_lambda > 0:
                l1_loss = self.l1_lambda * torch.norm(self.model.W, p=1)

            # Add Jacobian Regularization (Stability)
            jacobian_loss = torch.tensor(0.0)
            if self.jacobian_lambda > 0:
                # Compute Jacobian of the vector field at the initial state y[0]
                # Jacobian J is (dim, dim)
                J = F.jacobian(func_for_jacobian, y[0])

                # Penalize Frobenius norm of Jacobian to encourage stability (Lipshitz continuity)
                jacobian_loss = self.jacobian_lambda * torch.norm(J, p="fro")

            total_loss = mse_loss + l1_loss + jacobian_loss

            total_loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                logger.debug(
                    f"Epoch {epoch}, Loss: {total_loss.item()} "
                    f"(MSE: {mse_loss.item()}, L1: {l1_loss.item()}, Jac: {jacobian_loss.item()})"
                )

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
        # Transpose W to match (Target, Source) convention of the rest of the method
        # Previous: weights[i, j] -> Target i, Source j.
        # W: [Source, Target].
        # So W.T -> [Target, Source].
        weights = weights.T

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
