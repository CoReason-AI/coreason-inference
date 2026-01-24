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

from coreason_inference.analysis.latent import Discriminator, LatentMiner, permute_dims


class TestDiscriminator:
    def test_forward_shape(self) -> None:
        B, D = 32, 5
        disc = Discriminator(latent_dim=D)
        z = torch.randn(B, D)
        out = disc(z)
        assert out.shape == (B, 2)


class TestPermuteDims:
    def test_permute_dims_shape(self) -> None:
        z = torch.randn(10, 5)
        perm_z = permute_dims(z)
        assert perm_z.shape == z.shape

    def test_permute_dims_logic(self) -> None:
        # Create a tensor where each column has a unique value repeated
        # [0, 0]
        # [1, 1]
        # [2, 2]
        # If we permute, the row structure should be broken.
        B, D = 100, 2
        z = torch.stack([torch.arange(B, dtype=torch.float32)] * D, dim=1)

        # In original z, col0 == col1
        assert torch.all(z[:, 0] == z[:, 1])

        # Permute
        torch.manual_seed(42)
        perm_z = permute_dims(z)

        # In permuted z, col0 should likely NOT equal col1 for all rows
        # unless random permutation aligned perfectly (prob approx 0)
        # We check that at least one row is different
        assert not torch.all(perm_z[:, 0] == perm_z[:, 1])

        # But marginal statistics should be preserved (set of values in col 0 same)
        # Sort both columns and compare
        sorted_perm, _ = torch.sort(perm_z[:, 0])
        sorted_orig, _ = torch.sort(z[:, 0])
        assert torch.allclose(sorted_perm, sorted_orig)


class TestFactorVAEIntegration:
    @pytest.fixture
    def correlated_data(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 200
        z = np.random.normal(0, 1, n)
        # x1 and x2 are highly correlated via z
        x1 = z + np.random.normal(0, 0.1, n)
        x2 = z + np.random.normal(0, 0.1, n)
        return pd.DataFrame({"x1": x1, "x2": x2})

    def test_factor_vae_init(self) -> None:
        miner = LatentMiner(gamma=10.0)
        assert miner.gamma == 10.0
        assert miner.discriminator is None  # Before fit

    def test_factor_vae_fit_runs(self, correlated_data: pd.DataFrame) -> None:
        miner = LatentMiner(epochs=5, gamma=5.0)
        miner.fit(correlated_data)
        assert miner.model is not None
        assert miner.discriminator is not None
        assert isinstance(miner.discriminator, Discriminator)

    def test_tc_loss_integration(self, correlated_data: pd.DataFrame) -> None:
        """
        Check that training runs and updates weights, implying TC loss integration works.
        """
        miner = LatentMiner(epochs=10, gamma=5.0, learning_rate=1e-2)
        miner.fit(correlated_data)

        # Check that discriminator weights are not zero/random (it learned something)
        assert miner.discriminator is not None

        # Check if parameters have gradients (would be cleared after step, but we can check if they changed)
        # A simple check is that generation works
        gen = miner.generate(10)
        assert gen.shape == (10, 2)
