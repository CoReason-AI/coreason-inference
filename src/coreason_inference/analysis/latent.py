# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from coreason_inference.utils.logger import logger


class CausalVAE(nn.Module):  # type: ignore[misc]
    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 5):
        super().__init__()
        # Encoder
        self.encoder_hidden = nn.Linear(input_dim, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_output = nn.Linear(hidden_dim, input_dim)

        self.activation = nn.ReLU()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode
        h_enc = self.activation(self.encoder_hidden(x))
        mu = self.mu_layer(h_enc)
        logvar = self.logvar_layer(h_enc)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        h_dec = self.activation(self.decoder_hidden(z))
        x_hat = self.decoder_output(h_dec)

        return x_hat, mu, logvar

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper method for SHAP explanation. Returns only the mean of the latent distribution.
        """
        h_enc = self.activation(self.encoder_hidden(x))
        return self.mu_layer(h_enc)


class LatentMiner:
    """
    Discovers latent confounders using a Beta-VAE.
    """

    def __init__(
        self,
        latent_dim: int = 5,
        beta: float = 4.0,
        learning_rate: float = 1e-3,
        epochs: int = 1000,
        batch_size: int = 64,
    ):
        self.latent_dim = latent_dim
        self.beta = beta
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model: Optional[CausalVAE] = None
        self.scaler = StandardScaler()
        self.input_dim: int = 0

    def fit(self, data: pd.DataFrame) -> None:
        if data.empty:
            raise ValueError("Input data is empty.")

        # Robustness check: NaNs or Infs
        if data.isnull().values.any() or np.isinf(data.values).any():
            raise ValueError("Input data contains NaN or infinite values.")

        # Preprocessing
        self.input_dim = data.shape[1]
        X_scaled = self.scaler.fit_transform(data.values)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Initialize Model
        self.model = CausalVAE(self.input_dim, latent_dim=self.latent_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        logger.info(
            f"Training LatentMiner (Beta-VAE) with input_dim={self.input_dim}, "
            f"latent_dim={self.latent_dim}, beta={self.beta}"
        )

        # Training Loop
        self.model.train()
        for epoch in range(self.epochs):
            # Full batch gradient descent for simplicity in this atomic unit
            optimizer.zero_grad()
            x_hat, mu, logvar = self.model(X_tensor)

            # Loss Calculation
            # Reconstruction Loss (MSE)
            recon_loss = nn.functional.mse_loss(x_hat, X_tensor, reduction="sum")

            # KL Divergence Loss
            # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Total Loss
            loss = recon_loss + self.beta * kld_loss

            loss.backward()

            optimizer.step()

            if epoch % 100 == 0:
                logger.debug(f"Epoch {epoch}: Loss={loss.item()} (Recon={recon_loss.item()}, KLD={kld_loss.item()})")

        logger.info("LatentMiner training complete.")

    def discover_latents(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Maps input data to the latent space (Z).
        Returns a DataFrame of latent variables.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if data.isnull().values.any() or np.isinf(data.values).any():
            raise ValueError("Input data contains NaN or infinite values.")

        # Preprocessing
        X_scaled = self.scaler.transform(data.values)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Inference
        self.model.eval()
        with torch.no_grad():
            _, mu, _ = self.model(X_tensor)

        # Convert to DataFrame
        latent_cols = [f"Z_{i}" for i in range(self.latent_dim)]
        return pd.DataFrame(mu.numpy(), columns=latent_cols, index=data.index)

    def interpret_latents(self, data: pd.DataFrame, samples: int = 100) -> pd.DataFrame:
        """
        Interprets the discovered latent variables using SHAP values.
        Returns a DataFrame where rows are Latent Variables and columns are Input Features,
        representing the mean absolute SHAP value (Global Feature Importance).

        Args:
            data: The input dataframe to use for explanation.
            samples: Number of samples to use for the background dataset (if data is large).

        Returns:
            pd.DataFrame: Global feature importance matrix (Latent x Features).
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if data.empty:
            raise ValueError("Input data is empty.")

        # Preprocessing
        # We need scaled data for the model
        X_scaled = self.scaler.transform(data.values)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Background selection: Use a random subset if data is large, or K-Means
        # For simplicity and stability, we use a random subset as background
        if len(data) > samples:
            indices = np.random.choice(len(data), samples, replace=False)
            background_tensor = X_tensor[indices]
        else:
            background_tensor = X_tensor

        # We want to explain the `encode_mu` part of the model.
        # However, shap.DeepExplainer/GradientExplainer usually takes a module.
        # We can pass a wrapped method if we are careful, or just the whole model if it returns the target.
        # But our model returns a tuple.
        # So we create a lightweight wrapper module just for SHAP.

        class EncoderWrapper(nn.Module):  # type: ignore[misc]
            def __init__(self, model: CausalVAE):
                super().__init__()
                self.model = model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.model.encode_mu(x)

        wrapped_model = EncoderWrapper(self.model)
        wrapped_model.eval()

        # Define explain_tensor early to avoid UnboundLocalError
        explain_tensor = X_tensor[:samples] if len(X_tensor) > samples else X_tensor

        # Use GradientExplainer (better for PyTorch than DeepExplainer in some versions, or vice versa)
        # DeepExplainer is generally preferred for DL models in SHAP.
        try:
            # Note: DeepExplainer sometimes struggles with exact gradients if not set up perfectly.
            # GradientExplainer is robust for PyTorch.
            explainer = shap.DeepExplainer(wrapped_model, background_tensor)

            # Explain the whole dataset (or a subset if too large)
            shap_values = explainer.shap_values(explain_tensor)
        except Exception as e:
            logger.warning(f"DeepExplainer failed ({e}), falling back to KernelExplainer.")

            # Fallback to KernelExplainer (Model Agnostic, slower)
            # KernelExplainer expects a function that takes numpy array and returns numpy array
            def predict_fn(x_np: np.ndarray) -> np.ndarray:
                x_torch = torch.tensor(x_np, dtype=torch.float32)
                with torch.no_grad():
                    out = wrapped_model(x_torch)
                return out.numpy()

            # Using a smaller background for KernelExplainer as it is slow
            background_small = background_tensor.numpy()[:10]  # Very small background
            explainer = shap.KernelExplainer(predict_fn, background_small)
            shap_values = explainer.shap_values(explain_tensor.numpy())

        # SHAP values structure:
        # If output is (N_samples, N_latent), shap_values is usually a list of (N_samples, N_features) arrays,
        # one for each latent dimension. Or a single array (N_samples, N_latent, N_features).
        # DeepExplainer usually returns a list of arrays (one per output node).

        # Let's handle both cases.
        feature_importance_matrix = np.zeros((self.latent_dim, self.input_dim))

        if isinstance(shap_values, list):
            # List of [N_samples, N_features]
            for i, s_vals in enumerate(shap_values):
                # Global importance: Mean of absolute values
                feature_importance_matrix[i, :] = np.mean(np.abs(s_vals), axis=0)
        else:
            # Maybe array [N_samples, N_features] (if 1 output?) or [N_samples, N_features, N_outputs]?
            # Check shape
            # If shape is (N_samples, N_latent, N_features)
            # Actually shap often outputs list.
            # If latent_dim=1, it might be just array.
            s_vals = np.array(shap_values)
            if s_vals.ndim == 3:
                # If structure is (N_samples, N_features, N_latents) which often happens if not list
                # Check dimensions
                if s_vals.shape[2] == self.latent_dim:
                    # (N, Features, Latent)
                    # Average over samples (axis 0) -> (Features, Latent)
                    # Transpose to (Latent, Features)
                    feature_importance_matrix = np.mean(np.abs(s_vals), axis=0).T
                elif s_vals.shape[1] == self.latent_dim:
                    # (N, Latent, Features)
                    # Average over samples -> (Latent, Features)
                    feature_importance_matrix = np.mean(np.abs(s_vals), axis=0)
                else:
                    logger.error(f"Unexpected SHAP values shape: {s_vals.shape}")
            elif s_vals.ndim == 2:
                # Single output? (N, Features)
                if self.latent_dim == 1:
                    feature_importance_matrix[0, :] = np.mean(np.abs(s_vals), axis=0)

        # Verify shape just in case
        if feature_importance_matrix.shape != (self.latent_dim, self.input_dim):  # pragma: no cover
            logger.warning(f"Shape mismatch: {feature_importance_matrix.shape}, transposing.")
            if feature_importance_matrix.shape == (self.input_dim, self.latent_dim):
                feature_importance_matrix = feature_importance_matrix.T

        # Construct DataFrame
        # Rows: Z_0, Z_1...
        # Columns: Original Feature Names
        latent_index = [f"Z_{i}" for i in range(self.latent_dim)]
        feature_names = data.columns.tolist()

        return pd.DataFrame(feature_importance_matrix, index=latent_index, columns=feature_names)
