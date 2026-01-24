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
    """Causal Variational Autoencoder network."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 5):
        """Initializes the CausalVAE.

        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden dimension.
            latent_dim: Latent dimension.
        """
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
        """Reparameterization trick to sample from N(mu, var).

        Args:
            mu: Mean of the latent Gaussian.
            logvar: Log variance of the latent Gaussian.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the VAE.

        Args:
            x: Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                (Reconstructed input, Mean, Log Variance, Latent Vector)
        """
        # Encode
        h_enc = self.activation(self.encoder_hidden(x))
        mu = self.mu_layer(h_enc)
        logvar = self.logvar_layer(h_enc)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        h_dec = self.activation(self.decoder_hidden(z))
        x_hat = self.decoder_output(h_dec)

        # Return z as well for FactorVAE
        return x_hat, mu, logvar, z

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        """Helper method for SHAP explanation. Returns only the mean of the latent distribution.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Mean of the latent distribution.
        """
        h_enc = self.activation(self.encoder_hidden(x))
        return cast(torch.Tensor, self.mu_layer(h_enc))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latent vectors z back to the input space x_hat.

        Args:
            z: Latent tensor.

        Returns:
            torch.Tensor: Reconstructed input.
        """
        h_dec = self.activation(self.decoder_hidden(z))
        x_hat = self.decoder_output(h_dec)
        return cast(torch.Tensor, x_hat)


class Discriminator(nn.Module):  # type: ignore[misc]
    """Discriminator network for FactorVAE.

    Distinguishes between samples from the aggregate posterior q(z)
    and the product of marginals prod(q(z_j)).
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 32):
        """Initializes the Discriminator.

        Args:
            latent_dim: Latent dimension.
            hidden_dim: Hidden dimension.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 2),  # Logits for 2 classes: Permuted(0) vs Real(1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Latent vector.

        Returns:
            torch.Tensor: Logits.
        """
        return cast(torch.Tensor, self.net(z))


def permute_dims(z: torch.Tensor) -> torch.Tensor:
    """Permutes each dimension of the latent batch independently.

    Approximates the product of marginals q(z_1) * ... * q(z_d).

    Args:
        z: Latent batch tensor (Batch, Dim).

    Returns:
        torch.Tensor: Permuted latent batch.
    """
    assert z.dim() == 2
    B, D = z.size()
    perm_z = torch.zeros_like(z)
    # Shuffle each dimension independently
    for d in range(D):
        idx = torch.randperm(B)
        perm_z[:, d] = z[idx, d]
    return perm_z


class _ShapEncoderWrapper(nn.Module):  # type: ignore[misc]
    """Helper wrapper for SHAP explanation to isolate the encoder mean."""

    def __init__(self, model: CausalVAE):
        """Initializes the wrapper.

        Args:
            model: The CausalVAE model.
        """
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Encoded mean.
        """
        return self.model.encode_mu(x)


class LatentMiner:
    """The Latent Miner: Representation Learning & Disentanglement.

    Uses a FactorVAE (Causal VAE with Total Correlation penalty) to discover
    independent latent factors (Z) that explain the variance in the observed data.
    Enables generation of "Digital Twins" by sampling from the latent space.
    """

    def __init__(
        self,
        latent_dim: int = 5,
        beta: float = 1.0,  # Standard VAE weight
        gamma: float = 6.0,  # Total Correlation weight (FactorVAE)
        learning_rate: float = 1e-3,
        epochs: int = 1000,
        batch_size: int = 64,
    ):
        """Initializes the Latent Miner.

        Args:
            latent_dim: Dimension of the latent space (number of factors).
            beta: Weight for KLD term (standard VAE).
            gamma: Weight for Total Correlation term (FactorVAE).
            learning_rate: Learning rate for optimizer.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
        """
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model: Optional[CausalVAE] = None
        self.discriminator: Optional[Discriminator] = None
        self.scaler = StandardScaler()
        self.input_dim: int = 0
        self.feature_names: List[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _preprocess(self, data: pd.DataFrame, fit_scaler: bool = False) -> torch.Tensor:
        """Validates, scales, and converts input data to a tensor on the correct device.

        Args:
            data: Input dataframe.
            fit_scaler: Whether to fit the scaler.

        Returns:
            torch.Tensor: Preprocessed data tensor.

        Raises:
            ValueError: If data is empty or contains NaNs/Infs.
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
        """Fits the FactorVAE to the data to learn latent representations.

        Args:
            data: Input dataframe.

        Raises:
            ValueError: If data is empty or invalid.
        """
        X_tensor = self._preprocess(data, fit_scaler=True)

        # Initialize Models
        self.model = CausalVAE(self.input_dim, latent_dim=self.latent_dim).to(self.device)
        self.discriminator = Discriminator(self.latent_dim, hidden_dim=32).to(self.device)

        # Optimizers
        vae_optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)

        logger.info(
            f"Training LatentMiner (FactorVAE) with input_dim={self.input_dim}, "
            f"latent_dim={self.latent_dim}, beta={self.beta}, gamma={self.gamma} on {self.device}"
        )

        criterion = nn.CrossEntropyLoss()

        # Targets for Discriminator
        # Real (1) for samples from q(z)
        # Fake (0) for samples from prod q(z_j)
        # Note: We create them dynamically in loop to match batch size if we were doing batching,
        # but here we use full batch X_tensor.
        B = X_tensor.size(0)
        ones = torch.ones(B, dtype=torch.long, device=self.device)
        zeros = torch.zeros(B, dtype=torch.long, device=self.device)

        # Training Loop
        self.model.train()
        self.discriminator.train()

        for epoch in range(self.epochs):
            # 1. Forward Pass VAE
            vae_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            x_hat, mu, logvar, z = self.model(X_tensor)

            # 2. Train Discriminator
            # We want D to distinguish z (Real) from z_perm (Fake)
            # D needs to be updated based on detached z so we don't backprop into VAE yet
            z_detached = z.detach()
            z_perm = permute_dims(z_detached)

            d_real_logits = self.discriminator(z_detached)
            d_fake_logits = self.discriminator(z_perm)

            # Discriminator Loss: Maximize log(D(z)) + log(1 - D(z_perm))
            # Equivalent to minimizing CrossEntropy(D(z), 1) + CrossEntropy(D(z_perm), 0)
            disc_loss = criterion(d_real_logits, ones) + criterion(d_fake_logits, zeros)

            disc_loss.backward()
            disc_optimizer.step()

            # 3. Train VAE
            # We re-evaluate D(z) to get gradients for VAE (via Reparameterization Trick)
            # The Discriminator is fixed for this step.

            # Reconstruction Loss (MSE)
            recon_loss = nn.functional.mse_loss(x_hat, X_tensor, reduction="sum")

            # KL Divergence Loss
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Total Correlation Loss (from Discriminator)
            # TC(z) approx E[log(D(z)/(1-D(z)))]
            # If D outputs logits, log(p/(1-p)) is simply (logit_1 - logit_0)
            d_logits = self.discriminator(z)
            tc_loss = (d_logits[:, 1] - d_logits[:, 0]).sum()
            # Note: reduction='sum' for consistency with recon/kld which are sum over batch.
            # Usually TC is weighted.

            # Total VAE Loss
            vae_loss = recon_loss + self.beta * kld_loss + self.gamma * tc_loss

            vae_loss.backward()
            vae_optimizer.step()

            if epoch % 100 == 0:
                logger.debug(
                    f"Epoch {epoch}: VAE Loss={vae_loss.item():.2f} "
                    f"(Recon={recon_loss.item():.2f}, KLD={kld_loss.item():.2f}, TC={tc_loss.item():.2f}), "
                    f"Disc Loss={disc_loss.item():.2f}"
                )

        logger.info("LatentMiner training complete.")

    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generates synthetic data ('Digital Twins') by sampling from the latent space.

        Args:
            n_samples: Number of synthetic samples to generate.

        Returns:
            pd.DataFrame: Generated data in the original feature space.

        Raises:
            ValueError: If model is not trained.
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
        """Maps input data to the latent space (Z).

        Args:
            data: Input dataframe matching training features.

        Returns:
            pd.DataFrame: A DataFrame of latent variables (Z_0, Z_1, ...).

        Raises:
            ValueError: If model is not trained.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X_tensor = self._preprocess(data, fit_scaler=False)

        # Inference
        self.model.eval()
        with torch.no_grad():
            _, mu, _, _ = self.model(X_tensor)

        # Convert to DataFrame
        latent_cols = [f"Z_{i}" for i in range(self.latent_dim)]
        return pd.DataFrame(mu.cpu().numpy(), columns=latent_cols, index=data.index)

    def interpret_latents(self, data: pd.DataFrame, samples: int = 100) -> pd.DataFrame:
        """Interprets the discovered latent variables using SHAP values.

        Returns a DataFrame where rows are Latent Variables and columns are Input Features,
        representing the mean absolute SHAP value (Global Feature Importance).

        Args:
            data: The input dataframe to use for explanation.
            samples: Number of samples to use for the background dataset (if data is large).

        Returns:
            pd.DataFrame: Global feature importance matrix (Latent x Features).

        Raises:
            ValueError: If model is not trained.
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
