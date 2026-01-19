# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import Any, cast
from unittest.mock import MagicMock

import pandas as pd
import pytest

from coreason_inference.engine import InferenceEngine
from coreason_inference.schema import CausalGraph


class TestEngineExplainability:
    """
    Tests for the latent explanation (explainability) functionality in InferenceEngine.
    """

    @pytest.fixture
    def engine(self) -> InferenceEngine:
        return InferenceEngine()

    @pytest.fixture
    def mock_data(self) -> pd.DataFrame:
        # Small mock dataframe
        return pd.DataFrame(
            {"feature_1": [1.0, 2.0, 3.0, 4.0, 5.0], "feature_2": [0.5, 1.5, 2.5, 3.5, 4.5], "time": [0, 1, 2, 3, 4]}
        )

    def test_explain_latents_without_analyze(self, engine: InferenceEngine) -> None:
        """Test that calling explain_latents before analyze raises ValueError."""
        with pytest.raises(ValueError, match="Pipeline not run"):
            engine.explain_latents()

    def test_explain_latents_flow(self, engine: InferenceEngine, mock_data: pd.DataFrame) -> None:
        """
        Test the successful flow of explain_latents.
        Mocks LatentMiner to avoid expensive SHAP calculation.
        """
        # Mock internal components to skip heavy computation
        # We cast to Any to avoid Mypy complaining about method assignment
        cast(Any, engine.dynamics_engine).fit = MagicMock()

        # Use a real Pydantic model for CausalGraph to satisfy InferenceResult validation
        empty_graph = CausalGraph(nodes=[], edges=[], loop_dynamics=[], stability_score=0.9)
        cast(Any, engine.dynamics_engine).discover_loops = MagicMock(return_value=empty_graph)

        cast(Any, engine.latent_miner).fit = MagicMock()
        # discover_latents returns random latents
        cast(Any, engine.latent_miner).discover_latents = MagicMock(
            return_value=pd.DataFrame({"Z_0": [0.1] * 5, "Z_1": [0.2] * 5}, index=mock_data.index)
        )
        # interpret_latents returns a dummy dataframe
        expected_explanation = pd.DataFrame({"feature_1": [0.5, 0.2], "feature_2": [0.3, 0.4]}, index=["Z_0", "Z_1"])
        cast(Any, engine.latent_miner).interpret_latents = MagicMock(return_value=expected_explanation)
        engine.latent_miner.model = MagicMock()  # Set model to not None

        cast(Any, engine.active_scientist).fit = MagicMock()
        cast(Any, engine.active_scientist).propose_experiments = MagicMock(return_value=[])

        # Run analyze
        engine.analyze(data=mock_data, time_col="time", variable_cols=["feature_1", "feature_2"])

        # Call explain_latents
        explanation = engine.explain_latents(background_samples=50)

        # Assertions
        assert explanation.equals(expected_explanation)

        # Verify interpret_latents was called with correct arguments
        cast(MagicMock, engine.latent_miner.interpret_latents).assert_called_once()

        # Check args passed to interpret_latents
        call_args = cast(MagicMock, engine.latent_miner.interpret_latents).call_args
        passed_df = call_args[0][0]
        passed_samples = call_args[1].get("samples")

        # passed_df should only contain "feature_1" and "feature_2", not "time" or "Z_*"
        assert list(passed_df.columns) == ["feature_1", "feature_2"]
        assert passed_samples == 50
        assert engine._latent_features == ["feature_1", "feature_2"]

    def test_explain_latents_missing_data(self, engine: InferenceEngine) -> None:
        """Test error when model is present but data is somehow missing (defensive)."""
        engine.latent_miner.model = MagicMock()  # Simulate fitted model
        engine.augmented_data = None

        with pytest.raises(ValueError, match="Data not available"):
            engine.explain_latents()

    def test_explain_latents_column_mismatch(self, engine: InferenceEngine, mock_data: pd.DataFrame) -> None:
        """Test error when stored columns are missing from augmented data."""
        # Set up state manually
        engine.latent_miner.model = MagicMock()
        engine._latent_features = ["feature_1", "missing_col"]
        engine.augmented_data = mock_data  # Missing "missing_col"

        with pytest.raises(ValueError, match="Could not retrieve original features"):
            engine.explain_latents()
