# Copyright (c) 2026 CoReason, Inc.
# Licensed under the Prosperity Public License 3.0.0

from unittest.mock import MagicMock

import torch

from coreason_inference.analysis.latent import LatentMiner


def test_latent_generation_includes_z_columns() -> None:
    """
    Reproduction test for 'Data Amnesia' bug.
    Ensures that generated data includes the latent variables (Z) used to create it.
    """
    # Setup LatentMiner with mocked model
    miner = LatentMiner(latent_dim=2)
    miner.feature_names = ["feature_1", "feature_2"]
    miner.input_dim = 2

    # Mock the VAE model
    miner.model = MagicMock()
    # Mock decode to return dummy data (shape: n_samples x input_dim)
    # n_samples=5
    dummy_output = torch.randn(5, 2)
    miner.model.decode.return_value = dummy_output

    # Mock scaler
    miner.scaler = MagicMock()
    # inverse_transform returns numpy array
    miner.scaler.inverse_transform.return_value = dummy_output.numpy()

    # Generate data
    generated_df = miner.generate(n_samples=5)

    # Assertions
    assert "feature_1" in generated_df.columns
    assert "feature_2" in generated_df.columns

    # This assertion is expected to FAIL before the fix
    assert "Z_0" in generated_df.columns
    assert "Z_1" in generated_df.columns

    # Verify shape
    # 2 features + 2 latents = 4 columns
    assert generated_df.shape == (5, 4)
