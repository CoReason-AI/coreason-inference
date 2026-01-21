# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import Any, List, Optional, Tuple, cast

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

        return cast(torch.Tensor, x_hat), mu, logvar

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper method for SHAP explanation. Returns only the mean of the latent distribution.
        """
        h_enc = self.activation(self.encoder_hidden(x))
        return cast(torch.Tensor, self.mu_layer(h_enc))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent vectors z back to the input space x_hat.
        """
        h_dec = self.activation(self.decoder_hidden(z))
        x_hat = self.decoder_output(h_dec)
        return cast(torch.Tensor, x_hat)


class _ShapEncoderWrapper(nn.Module):  # type: ignore[misc]
    """
    Helper wrapper for SHAP explanation to isolate the encoder mean.
    """

    def __init__(self, model: CausalVAE):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode_mu(x)


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
        self.feature_names: List[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _preprocess(self, data: pd.DataFrame, fit_scaler: bool = False) -> torch.Tensor:
        """
        Validates, scales, and converts input data to a tensor on the correct device.
        """
        if data.empty:
            raise ValueError("Input data is empty.")

        # Robustness check: NaNs or Infs
        if data.isnull().values.any() or np.isinf(data.values).any():
            raise ValueError("Input data contains NaN or infinite values.")

        if fit_scaler:
            self.input_dim = data.shape[1]
            self.feature_names = data.columns.tolist()
            X_scaled = self.scaler.fit_transform(data.values)
        else:
            X_scaled = self.scaler.transform(data.values)

        return torch.tensor(X_scaled, dtype=torch.float32, device=self.device)

    def fit(self, data: pd.DataFrame) -> None:
        X_tensor = self._preprocess(data, fit_scaler=True)

        # Initialize Model
        self.model = CausalVAE(self.input_dim, latent_dim=self.latent_dim).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        logger.info(
            f"Training LatentMiner (Beta-VAE) with input_dim={self.input_dim}, "
            f"latent_dim={self.latent_dim}, beta={self.beta} on {self.device}"
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

    def generate(self, n_samples: int) -> pd.DataFrame:
        """
        Generates synthetic data ('Digital Twins') by sampling from the latent space.

        Args:
            n_samples: Number of synthetic samples to generate.

        Returns:
            pd.DataFrame: Generated data in the original feature space.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if n_samples <= 0:
            return pd.DataFrame(columns=self.feature_names)

        # Sample from Prior N(0, I)
        z = torch.randn(n_samples, self.latent_dim, device=self.device)

        # Decode
        self.model.eval()
        with torch.no_grad():
            x_hat_scaled = self.model.decode(z).cpu().numpy()

        # Inverse Transform
        x_hat = self.scaler.inverse_transform(x_hat_scaled)

        # Return DataFrame
        return pd.DataFrame(x_hat, columns=self.feature_names)

    def discover_latents(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Maps input data to the latent space (Z).
        Returns a DataFrame of latent variables.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X_tensor = self._preprocess(data, fit_scaler=False)

        # Inference
        self.model.eval()
        with torch.no_grad():
            _, mu, _ = self.model(X_tensor)

        # Convert to DataFrame
        latent_cols = [f"Z_{i}" for i in range(self.latent_dim)]
        return pd.DataFrame(mu.cpu().numpy(), columns=latent_cols, index=data.index)

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

        # preprocess returns tensor on self.device
        X_tensor = self._preprocess(data, fit_scaler=False)

        # Background selection
        if len(data) > samples:
            indices = np.random.choice(len(data), samples, replace=False)
            background_tensor = X_tensor[indices]
        else:
            background_tensor = X_tensor

        wrapped_model = _ShapEncoderWrapper(self.model)
        wrapped_model.eval()

        # SHAP often expects tensors if model is nn.Module, but for KernelExplainer it needs functions.
        # DeepExplainer usually handles device if model and data are on same device.

        explain_tensor = X_tensor[:samples] if len(X_tensor) > samples else X_tensor

        try:
            # DeepExplainer is robust for PyTorch.
            explainer = shap.DeepExplainer(wrapped_model, background_tensor)

            # Explain the whole dataset (or a subset if too large)
            shap_values = explainer.shap_values(explain_tensor)
        except Exception as e:
            logger.warning(f"DeepExplainer failed ({e}), falling back to KernelExplainer.")

            # Fallback to KernelExplainer
            # Expects function: numpy -> numpy
            def predict_fn(x_np: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
                x_torch = torch.tensor(x_np, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    out = wrapped_model(x_torch)
                return cast(np.ndarray[Any, Any], out.cpu().numpy())

            # Using a smaller background for KernelExplainer as it is slow
            background_small = background_tensor.cpu().numpy()[:10]  # Very small background
            explainer = shap.KernelExplainer(predict_fn, background_small)
            shap_values = explainer.shap_values(explain_tensor.cpu().numpy())

        # SHAP values structure handling (same as before)
        feature_importance_matrix = np.zeros((self.latent_dim, self.input_dim))

        if isinstance(shap_values, list):
            for i, s_vals in enumerate(shap_values):
                feature_importance_matrix[i, :] = np.mean(np.abs(s_vals), axis=0)
        else:
            s_vals = np.array(shap_values)
            if s_vals.ndim == 3:
                if s_vals.shape[2] == self.latent_dim:
                    feature_importance_matrix = np.mean(np.abs(s_vals), axis=0).T
                elif s_vals.shape[1] == self.latent_dim:
                    feature_importance_matrix = np.mean(np.abs(s_vals), axis=0)
                else:
                    logger.error(f"Unexpected SHAP values shape: {s_vals.shape}")
            elif s_vals.ndim == 2:
                if self.latent_dim == 1:
                    feature_importance_matrix[0, :] = np.mean(np.abs(s_vals), axis=0)

        if feature_importance_matrix.shape != (self.latent_dim, self.input_dim):  # pragma: no cover
            logger.warning(f"Shape mismatch: {feature_importance_matrix.shape}, transposing.")
            if feature_importance_matrix.shape == (self.input_dim, self.latent_dim):
                feature_importance_matrix = feature_importance_matrix.T

        latent_index = [f"Z_{i}" for i in range(self.latent_dim)]
        feature_names = data.columns.tolist()

        return pd.DataFrame(feature_importance_matrix, index=latent_index, columns=feature_names)
