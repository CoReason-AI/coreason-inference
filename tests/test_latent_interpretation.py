# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import Any

import numpy as np
import pandas as pd
import pytest

from coreason_inference.analysis.latent import LatentMiner


class TestLatentInterpretation:
    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        np.random.seed(42)
        # Create correlated data
        n_samples = 50
        # z1 controls x1 and x2
        # z2 controls x3
        x1 = np.random.normal(0, 1, n_samples)
        x2 = x1 * 0.8 + np.random.normal(0, 0.1, n_samples)
        x3 = np.random.normal(0, 1, n_samples)

        # Add a noise column that shouldn't affect anything much
        x4 = np.random.normal(0, 0.01, n_samples)

        df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4})
        return df

    def test_interpret_latents_structure(self, sample_data: pd.DataFrame) -> None:
        """
        Tests that interpret_latents returns the correct DataFrame structure.
        """
        latent_dim = 2
        miner = LatentMiner(latent_dim=latent_dim, epochs=10)  # Quick training
        miner.fit(sample_data)

        # Interpret
        importance_df = miner.interpret_latents(sample_data, samples=20)

        # Checks
        assert isinstance(importance_df, pd.DataFrame)
        assert importance_df.shape == (latent_dim, 4)  # 2 latents, 4 features
        assert list(importance_df.index) == ["Z_0", "Z_1"]
        assert list(importance_df.columns) == ["x1", "x2", "x3", "x4"]

        # Check values are non-negative (mean absolute SHAP)
        assert (importance_df.values >= 0).all()

    def test_interpret_without_fit_error(self, sample_data: pd.DataFrame) -> None:
        miner = LatentMiner()
        with pytest.raises(ValueError, match="Model not trained"):
            miner.interpret_latents(sample_data)

    def test_interpret_empty_data_error(self) -> None:
        miner = LatentMiner(epochs=1)
        # We need to fit first to pass the first check
        miner.fit(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))

        with pytest.raises(ValueError, match="Input data is empty"):
            miner.interpret_latents(pd.DataFrame())

    def test_interpret_latents_value_logic(self, sample_data: pd.DataFrame) -> None:
        """
        Tests that the interpretation results are somewhat logical (non-zero).
        Since we train very briefly, we can't guarantee perfect disentanglement,
        but we can check that we get *some* importance values.
        """
        miner = LatentMiner(latent_dim=2, epochs=50)
        miner.fit(sample_data)

        importance_df = miner.interpret_latents(sample_data, samples=20)

        # Sum of importances should be > 0 (unless model died completely)
        assert importance_df.sum().sum() > 0

    def test_fallback_mechanism(self, sample_data: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Force DeepExplainer to fail to test fallback to KernelExplainer.
        """
        miner = LatentMiner(latent_dim=2, epochs=10)
        miner.fit(sample_data)

        import shap

        # Mock DeepExplainer to raise Exception
        def mock_deep_explainer(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("DeepExplainer failed")

        monkeypatch.setattr(shap, "DeepExplainer", mock_deep_explainer)

        # Should rely on KernelExplainer now
        # We can also mock KernelExplainer to verify it was called, but checking result is valid is enough
        importance_df = miner.interpret_latents(sample_data, samples=10)

        assert isinstance(importance_df, pd.DataFrame)
        assert importance_df.shape == (2, 4)
