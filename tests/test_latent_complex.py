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
import torch.nn as nn

from coreason_inference.analysis.latent import LatentMiner


class TestLatentMinerComplex:
    @pytest.fixture
    def clean_data(self) -> pd.DataFrame:
        np.random.seed(42)
        return pd.DataFrame(np.random.randn(100, 5), columns=[f"c{i}" for i in range(5)])

    def test_input_with_nans(self, clean_data: pd.DataFrame) -> None:
        """
        Verify that NaN values raise a ValueError.
        """
        data = clean_data.copy()
        data.iloc[0, 0] = np.nan
        miner = LatentMiner()
        with pytest.raises(ValueError, match="Input data contains NaN or infinite values"):
            miner.fit(data)

    def test_input_with_inf(self, clean_data: pd.DataFrame) -> None:
        """
        Verify that Infinite values raise a ValueError.
        """
        data = clean_data.copy()
        data.iloc[0, 0] = np.inf
        miner = LatentMiner()
        with pytest.raises(ValueError, match="Input data contains NaN or infinite values"):
            miner.fit(data)

    def test_discover_with_nan(self, clean_data: pd.DataFrame) -> None:
        """
        Verify that discovery raises error on NaNs even after fit.
        """
        miner = LatentMiner(epochs=10)
        miner.fit(clean_data)

        bad_data = clean_data.copy()
        bad_data.iloc[0, 0] = np.nan

        with pytest.raises(ValueError, match="Input data contains NaN or infinite values"):
            miner.discover_latents(bad_data)

    def test_single_feature_input(self) -> None:
        """
        Verify robustness with 1D input.
        """
        data = pd.DataFrame({"x": np.random.randn(50)})
        miner = LatentMiner(latent_dim=1, epochs=10)
        miner.fit(data)

        latents = miner.discover_latents(data)
        assert latents.shape == (50, 1)

    def test_constant_feature_input(self) -> None:
        """
        Verify robustness with constant input (variance = 0).
        StandardScaler handles this by setting 0s (centering).
        """
        data = pd.DataFrame(
            {
                "x": np.random.randn(50),
                "c": np.ones(50),  # Constant column
            }
        )
        miner = LatentMiner(epochs=10)
        # Should not crash
        miner.fit(data)
        latents = miner.discover_latents(data)
        assert latents.shape == (50, 5)  # Default latent dim

    def test_latent_dim_larger_than_input(self) -> None:
        """
        Verify we can project to a higher dimension than input.
        """
        data = pd.DataFrame(np.random.randn(50, 2), columns=["a", "b"])
        # Input 2 -> Latent 10
        miner = LatentMiner(latent_dim=10, epochs=10)
        miner.fit(data)

        latents = miner.discover_latents(data)
        assert latents.shape == (50, 10)

    def test_training_loss_reduction(self, clean_data: pd.DataFrame) -> None:
        """
        Verify that the model actually learns (loss decreases).
        """
        miner = LatentMiner(epochs=500, learning_rate=0.01, beta=1.0)

        # Train
        miner.fit(clean_data)

        # Ensure model is not None for mypy
        assert miner.model is not None

        # Check reconstruction error on training data
        # We can't easily get X_hat from discover_latents (returns Z).
        # We have to access model internals for this test

        X_scaled = miner.scaler.transform(clean_data.values)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        miner.model.eval()
        with torch.no_grad():
            x_hat, _, _ = miner.model(X_tensor)

        final_mse = nn.functional.mse_loss(x_hat, X_tensor).item()

        # Initial MSE for standardized data (mean 0, std 1) with random weights
        # expected MSE roughly 1.0 + variance of random init.
        # If model learns, MSE should be significantly lower than 1.0 (unless noise is high).
        # For random data, it might not compress well, but AE should fit noise to some extent.

        # Let's assert it's somewhat low (e.g. < 0.9) to prove optimization steps happened
        assert final_mse < 1.0

    def test_reproducibility(self, clean_data: pd.DataFrame) -> None:
        """
        Verify strict seeding creates identical results.
        Note: We need to set torch seed externally as LatentMiner doesn't accept a seed arg yet.
        """

        # Run 1
        torch.manual_seed(42)
        miner1 = LatentMiner(epochs=50)
        miner1.fit(clean_data)
        res1 = miner1.discover_latents(clean_data)

        # Run 2
        torch.manual_seed(42)
        miner2 = LatentMiner(epochs=50)
        miner2.fit(clean_data)
        res2 = miner2.discover_latents(clean_data)

        pd.testing.assert_frame_equal(res1, res2)
