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

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from coreason_inference.utils.logger import logger
from sklearn.preprocessing import StandardScaler


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
