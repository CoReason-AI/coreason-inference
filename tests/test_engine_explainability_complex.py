# Copyright (c) 2026 CoReason, Inc.
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

from coreason_inference.analysis.dynamics import DynamicsEngine
from coreason_inference.engine import InferenceEngine


class TestEngineExplainabilityComplex:
    """
    Complex integration tests for latent explainability, verifying
    end-to-end SHAP interpretation on synthetic data.
    """

    @pytest.fixture
    def synthetic_structured_data(self) -> pd.DataFrame:
        """
        Generates data with a strong latent structure.
        Latent Z determines Feature A and Feature B.
        Feature Noise is random.
        """
        np.random.seed(42)
        n_samples = 100
        t = np.linspace(0, 10, n_samples)

        # Latent driver
        z = np.sin(t)

        # Features coupled to Z
        feature_a = 2.0 * z + np.random.normal(0, 0.1, n_samples)
        feature_b = -1.5 * z + np.random.normal(0, 0.1, n_samples)

        # Noise feature independent of Z
        feature_noise = np.random.normal(0, 1.0, n_samples)

        return pd.DataFrame({"time": t, "Feature_A": feature_a, "Feature_B": feature_b, "Feature_Noise": feature_noise})

    def test_meaningful_explanation(self, synthetic_structured_data: pd.DataFrame, mock_user_context) -> None:
        """
        Verifies that explain_latents correctly identifies features driving the latent space.
        We expect at least one discovered latent to have high SHAP importance for
        Feature_A and Feature_B, and low importance for Feature_Noise.
        """
        # Inject robust dynamics engine (rk4) to handle data without underflow
        engine = InferenceEngine(dynamics_engine=DynamicsEngine(method="rk4"))

        # Run pipeline
        engine.analyze(
            data=synthetic_structured_data,
            time_col="time",
            variable_cols=["Feature_A", "Feature_B", "Feature_Noise"],
            context=mock_user_context,
        )

        # Explain latents
        # This runs the actual SHAP explainer (Kernel or Deep)
        explanation = engine.explain_latents(background_samples=50)

        assert not explanation.empty
        assert set(explanation.columns) == {"Feature_A", "Feature_B", "Feature_Noise"}

        # Check if any latent captures the structure
        # We look for a row where Importance(A) + Importance(B) > Importance(Noise) * factor

        # Use vectorized pandas operations (Performance Best Practice)
        # explanation shape: (n_latents, n_features)
        signal_strength = explanation["Feature_A"].abs() + explanation["Feature_B"].abs()
        noise_strength = explanation["Feature_Noise"].abs()

        # Check if any latent satisfies the condition
        # Returns a Series of booleans, .any() reduces to single bool
        structure_found = (signal_strength > 2.0 * noise_strength).any()

        # Note: In VAEs, sometimes the latent collapses or is distributed.
        # But with 100 samples and strong signal, it should find it.
        # If this is flaky, we might relax the check, but for now we enforce quality.
        assert structure_found, "SHAP did not identify Feature A/B as more important than Noise for any latent."

    def test_oversampling_behavior(self, mock_user_context) -> None:
        """
        Test that requesting more background samples than available rows works gracefully.
        """
        # Very small dataset
        df = pd.DataFrame({"time": range(10), "X": np.random.randn(10), "Y": np.random.randn(10)})

        # Inject robust dynamics engine (rk4) to handle random noise data without underflow
        engine = InferenceEngine(dynamics_engine=DynamicsEngine(method="rk4"))
        engine.analyze(df, "time", ["X", "Y"], context=mock_user_context)

        # Request 100 samples (dataset only has 10)
        explanation = engine.explain_latents(background_samples=100)

        assert not explanation.empty
        assert explanation.shape[1] == 2  # Columns X, Y

    def test_reanalysis_state_consistency(self, mock_user_context) -> None:
        """
        Test that running analyze() multiple times updates the latent feature state correctly.
        """
        df = pd.DataFrame(
            {"time": range(20), "A": np.random.randn(20), "B": np.random.randn(20), "C": np.random.randn(20)}
        )

        # Inject robust dynamics engine (rk4)
        engine = InferenceEngine(dynamics_engine=DynamicsEngine(method="rk4"))

        # 1. Analyze A, B
        engine.analyze(df, "time", ["A", "B"], context=mock_user_context)
        expl_1 = engine.explain_latents(background_samples=10)
        assert set(expl_1.columns) == {"A", "B"}

        # 2. Re-analyze B, C
        # Should overwrite previous state
        engine.analyze(df, "time", ["B", "C"], context=mock_user_context)
        expl_2 = engine.explain_latents(background_samples=10)
        assert set(expl_2.columns) == {"B", "C"}
        assert "A" not in expl_2.columns
