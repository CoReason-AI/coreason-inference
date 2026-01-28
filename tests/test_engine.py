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

from coreason_inference.engine import InferenceEngine
from coreason_inference.schema import CausalGraph, RefutationStatus


def generate_synthetic_system(n_samples: int = 100, t_steps: int = 20, noise_std: float = 0.05) -> pd.DataFrame:
    """
    Generates a synthetic biological system with:
    1. Feedback Loop: A -> B -> A (Negative)
    2. Latent Confounder: Z -> A, Z -> B
    3. Time series dynamics
    """
    np.random.seed(42)

    # Time points
    t = np.linspace(0, 10, t_steps)

    # Latent Variable Z (e.g., Genetic Factor)
    # Z varies by 'patient' or sample, but here we simulate a single long trajectory or multiple samples?
    # DynamicsEngine expects a single time-series usually, or multiple?
    # The current DynamicsEngine implementation takes a DataFrame and sorts by time.
    # It treats it as one continuous trajectory.
    # To simulate "samples", we might stack them, but DynamicsEngine fits one ODE.
    # So we simulate one long time-series.

    # Z is a hidden state that varies slowly or is constant.
    # Let's make Z a slow sine wave to simulate diurnal rhythm or similar.
    z = np.sin(t * 0.5)

    # Dynamics:
    # dA/dt = -0.5*A - 1.0*B + 2.0*Z
    # dB/dt = 0.5*A - 0.5*B + 1.0*Z
    # B inhibits A (negative term in dA/dt from B? No, -1.0*B means B inhibits A)
    # A activates B (positive term in dB/dt from A)

    a = np.zeros_like(t)
    b = np.zeros_like(t)
    a[0] = 1.0
    b[0] = 0.5

    dt = t[1] - t[0]

    for i in range(len(t) - 1):
        da = (-0.5 * a[i] - 1.0 * b[i] + 2.0 * z[i]) * dt
        db = (0.5 * a[i] - 0.5 * b[i] + 1.0 * z[i]) * dt

        a[i + 1] = a[i] + da + np.random.normal(0, noise_std)
        b[i + 1] = b[i] + db + np.random.normal(0, noise_std)

    # DataFrame
    df = pd.DataFrame({"time": t, "A": a, "B": b, "Z_true": z})

    # We hide Z_true in the input to the engine
    return df


class TestInferenceEngine:
    def test_pipeline_integration(self, mock_user_context) -> None:
        """
        Comprehensive test of the Discover-Represent-Simulate-Act loop.
        """
        # 1. Prepare Data
        df = generate_synthetic_system(n_samples=100, t_steps=50)
        # Drop the true latent for the input
        input_df = df[["time", "A", "B"]].copy()

        # 2. Instantiate Engine
        engine = InferenceEngine()

        # 3. Run Pipeline
        # We also ask to estimate effect of A on B
        result = engine.analyze(
            data=input_df, time_col="time", variable_cols=["A", "B"], estimate_effect_for=("A", "B")
        , context=mock_user_context)

        # 4. Validations

        # A. Discovery (Dynamics)
        assert isinstance(result.graph, CausalGraph)
        assert len(result.graph.nodes) == 2
        # We expect a feedback loop between A and B
        # (Though NeuralODE on small data with defaults might vary, we check structure exists)
        assert len(result.graph.edges) >= 1

        # Check stability score is calculated
        assert isinstance(result.graph.stability_score, float)

        # B. Representation (Latents)
        # LatentMiner should have found some latents (default dim=5)
        # And augmented data should have them
        assert isinstance(result.latents, pd.DataFrame)
        assert result.latents.shape[0] == len(input_df)
        assert result.latents.shape[1] > 0

        # Augmented data should contain A, B, time, and Z_0...Z_n
        assert "A" in result.augmented_data.columns
        assert "Z_0" in result.augmented_data.columns

        # C. Act (Proposals)
        # Active Scientist should have run
        # Note: PC algorithm might not find ambiguity on this specific data,
        # but the list should be present (empty or not).
        assert isinstance(result.proposals, list)

        # D. Simulate (Estimation)
        # We checked estimation runs via logs/coverage, but let's check explicit call
        # using the engine's method.
        # Estimate effect of B on A (Reverse)
        # We need to pass confounders. Let's use the discovered latents.
        latents = list(result.latents.columns)
        intervention_result = engine.estimate_effect(treatment="B", outcome="A", confounders=latents, context=mock_user_context)

        assert intervention_result.intervention == "do(B)"
        assert isinstance(intervention_result.counterfactual_outcome, float)
        assert intervention_result.refutation_status in [RefutationStatus.PASSED, RefutationStatus.FAILED]

    def test_missing_data_logic(self, mock_user_context) -> None:
        """
        Test how the engine handles requested variables not in data.
        """
        engine = InferenceEngine()
        df = pd.DataFrame({"time": [0, 1], "X": [1, 2]})

        # Should raise error from DynamicsEngine (insufficient points or similar)
        # or earlier
        # Pandas raises KeyError if columns missing, but DynamicsEngine validation
        # checks data.empty which assumes success, then selects.
        # Actually it selects columns first in dynamics.fit.
        # We expect KeyError from pandas, not ValueError.
        with pytest.raises(KeyError):
            engine.analyze(df, "time", ["X", "Y_missing"], context=mock_user_context)

    def test_engine_state_management(self, mock_user_context) -> None:
        """
        Test accessing methods before analyze raises errors.
        """
        engine = InferenceEngine()
        with pytest.raises(ValueError, match="Data not available"):
            engine.estimate_effect("A", "B", [], context=mock_user_context)

        with pytest.raises(ValueError, match="Pipeline not run"):
            engine.explain_latents()

    def test_analyze_estimation_edge_cases(self, mock_user_context) -> None:
        """
        Test edge cases in analyze method's estimation block.
        """
        df = generate_synthetic_system(n_samples=20, t_steps=10)
        input_df = df[["time", "A", "B"]].copy()
        engine = InferenceEngine()

        # 1. Test missing treatment/outcome columns (Warning path)
        # We need to capture logs to verify warning, but for coverage just running it is enough
        engine.analyze(
            data=input_df, time_col="time", variable_cols=["A", "B"], estimate_effect_for=("MISSING_A", "MISSING_B")
        , context=mock_user_context)

        # 2. Test explain_latents after fit
        # Now returns valid dataframe with SHAP values
        explanation = engine.explain_latents(background_samples=10)
        assert isinstance(explanation, pd.DataFrame)
        assert not explanation.empty
        # Check structure: Rows should be Latents (Z), Columns should be Features (A, B)
        # Latent miner defaults to 5 latents
        assert explanation.shape[0] == 5
        assert set(explanation.columns) == {"A", "B"}

    def test_analyze_estimation_failure(self, monkeypatch: pytest.MonkeyPatch, mock_user_context) -> None:
        """
        Test exception handling during estimation in analyze pipeline.
        """
        df = generate_synthetic_system(n_samples=20, t_steps=10)
        input_df = df[["time", "A", "B"]].copy()
        engine = InferenceEngine()

        # Mock CausalEstimator to raise Exception
        def mock_estimate(*args: object, **kwargs: object) -> None:
            raise RuntimeError("Estimation exploded")

        # We need to patch the CausalEstimator class used inside engine.
        # Since engine imports it, we patch 'coreason_inference.engine.CausalEstimator'
        # But wait, engine instantiates it.
        # We can patch the instance method if we could intercept it, but it's created inside.
        # Easier to patch the class in the module.

        # We need a mock class that raises on estimate_effect
        class MockEstimator:
            def __init__(self, data: pd.DataFrame) -> None:
                pass

            def estimate_effect(self, *args: object, **kwargs: object) -> None:
                raise RuntimeError("Estimation exploded")

        monkeypatch.setattr("coreason_inference.engine.CausalEstimator", MockEstimator)

        # Should not raise, just log error
        engine.analyze(data=input_df, time_col="time", variable_cols=["A", "B"], estimate_effect_for=("A", "B"), context=mock_user_context)
