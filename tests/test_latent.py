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

    def test_decode(self) -> None:
        latent_dim = 5
        input_dim = 10
        model = CausalVAE(input_dim, latent_dim=latent_dim)
        z = torch.randn(32, latent_dim)
        x_hat = model.decode(z)
        assert x_hat.shape == (32, input_dim)


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

    def test_generate(self, sample_data: pd.DataFrame) -> None:
        miner = LatentMiner(latent_dim=2, epochs=10)
        miner.fit(sample_data)

        # Generate data
        n_samples = 50
        generated_data = miner.generate(n_samples)

        assert isinstance(generated_data, pd.DataFrame)
        # 3 features + 2 latents = 5 columns
        assert generated_data.shape == (n_samples, 5)
        assert list(generated_data.columns) == ["x1", "x2", "x3", "Z_0", "Z_1"]

        # Check values range (since original is ~N(0, 1), generated features should be roughly within -5, 5)
        # We only check features cols for this range
        feature_data = generated_data[["x1", "x2", "x3"]]
        assert feature_data.values.max() < 10
        assert feature_data.values.min() > -10

    def test_generate_without_fit_error(self) -> None:
        miner = LatentMiner()
        with pytest.raises(ValueError, match="Model not trained"):
            miner.generate(10)

    def test_generate_zero_samples(self, sample_data: pd.DataFrame) -> None:
        miner = LatentMiner(latent_dim=2, epochs=10)
        miner.fit(sample_data)
        generated_data = miner.generate(0)
        assert isinstance(generated_data, pd.DataFrame)
        assert generated_data.shape == (0, 3)
        assert list(generated_data.columns) == ["x1", "x2", "x3"]

    def test_generate_large_samples(self, sample_data: pd.DataFrame) -> None:
        miner = LatentMiner(latent_dim=2, epochs=10)
        miner.fit(sample_data)
        n = 1000
        generated_data = miner.generate(n)
        assert generated_data.shape == (n, 5)  # 3 features + 2 latents

    def test_refit_updates_columns(self) -> None:
        miner = LatentMiner(latent_dim=2, epochs=2)

        # Fit 1: Cols A, B (plus 2 latents)
        df1 = pd.DataFrame(np.random.randn(10, 2), columns=["A", "B"])
        miner.fit(df1)
        gen1 = miner.generate(5)
        assert list(gen1.columns) == ["A", "B", "Z_0", "Z_1"]

        # Fit 2: Cols C, D, E (plus 2 latents)
        df2 = pd.DataFrame(np.random.randn(10, 3), columns=["C", "D", "E"])
        miner.fit(df2)
        gen2 = miner.generate(5)
        assert list(gen2.columns) == ["C", "D", "E", "Z_0", "Z_1"]
        assert gen2.shape == (5, 5)

    def test_generate_reproducibility(self, sample_data: pd.DataFrame) -> None:
        miner = LatentMiner(latent_dim=2, epochs=10)
        miner.fit(sample_data)

        torch.manual_seed(123)
        gen1 = miner.generate(10)

        torch.manual_seed(123)
        gen2 = miner.generate(10)

        pd.testing.assert_frame_equal(gen1, gen2)

    def test_generate_distribution_shift(self) -> None:
        """
        Verify that generated data respects the shift of the input data.
        If input is shifted by +100, output should be around +100.
        """
        # Data centered at 100
        n = 100
        df = pd.DataFrame({"A": np.random.normal(100, 1, n)})

        miner = LatentMiner(latent_dim=1, epochs=500, learning_rate=0.01)  # More epochs to learn mean
        miner.fit(df)

        gen = miner.generate(50)
        mean_val = gen["A"].mean()

        # VAEs with standard scaler should reconstruct the mean reasonably well.
        # Allow some margin (e.g., +/- 5)
        assert 90 < mean_val < 110
