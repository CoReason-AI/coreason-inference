# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

import numpy as np
import pandas as pd
import pytest
import torch

from coreason_inference.analysis.latent import CausalVAE, LatentMiner


class TestCausalVAE:
    def test_init(self) -> None:
        input_dim = 10
        latent_dim = 5
        model = CausalVAE(input_dim, latent_dim=latent_dim)

        assert model.encoder_hidden.in_features == input_dim
        assert model.mu_layer.out_features == latent_dim
        assert model.logvar_layer.out_features == latent_dim
        assert model.decoder_output.out_features == input_dim

    def test_forward(self) -> None:
        input_dim = 10
        latent_dim = 5
        batch_size = 32
        model = CausalVAE(input_dim, latent_dim=latent_dim)

        x = torch.randn(batch_size, input_dim)
        x_hat, mu, logvar = model(x)

        assert x_hat.shape == (batch_size, input_dim)
        assert mu.shape == (batch_size, latent_dim)
        assert logvar.shape == (batch_size, latent_dim)

    def test_reparameterize(self) -> None:
        latent_dim = 5
        batch_size = 32
        model = CausalVAE(10, latent_dim=latent_dim)

        mu = torch.zeros(batch_size, latent_dim)
        logvar = torch.zeros(batch_size, latent_dim)

        # Determine strict seeds for reproducibility in test
        torch.manual_seed(42)
        z = model.reparameterize(mu, logvar)

        assert z.shape == (batch_size, latent_dim)
        # Should not be exactly equal to mu (because of noise)
        assert not torch.allclose(z, mu)


class TestLatentMiner:
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        np.random.seed(42)
        # Create correlated data
        n_samples = 100
        z_true = np.random.normal(0, 1, (n_samples, 2))
        x1 = z_true[:, 0] + np.random.normal(0, 0.1, n_samples)
        x2 = z_true[:, 0] - z_true[:, 1] + np.random.normal(0, 0.1, n_samples)
        x3 = z_true[:, 1] * 2 + np.random.normal(0, 0.1, n_samples)

        df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
        return df

    def test_fit_and_discover(self, sample_data: pd.DataFrame) -> None:
        miner = LatentMiner(latent_dim=2, epochs=50)  # Low epochs for speed

        # Test Fit
        miner.fit(sample_data)
        assert miner.model is not None
        assert miner.input_dim == 3

        # Test Discover
        latents = miner.discover_latents(sample_data)

        assert isinstance(latents, pd.DataFrame)
        assert latents.shape == (100, 2)
        assert list(latents.columns) == ["Z_0", "Z_1"]

    def test_fit_empty_data_error(self) -> None:
        miner = LatentMiner()
        with pytest.raises(ValueError, match="Input data is empty"):
            miner.fit(pd.DataFrame())

    def test_discover_without_fit_error(self, sample_data: pd.DataFrame) -> None:
        miner = LatentMiner()
        with pytest.raises(ValueError, match="Model not trained"):
            miner.discover_latents(sample_data)
