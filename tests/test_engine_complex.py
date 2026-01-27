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
from coreason_identity.models import UserContext

from coreason_inference.analysis.dynamics import DynamicsEngine
from coreason_inference.engine import InferenceEngine


class TestInferenceEngineComplex:
    def test_index_alignment_resilience(self) -> None:
        """
        Verify that data augmentation correctly handles non-standard indices.
        If indices are mismatched during concatenation, it would introduce NaNs.
        """
        # Create data with a DatetimeIndex that is NOT 0..N
        dates = pd.date_range(start="2023-01-01", periods=20, freq="h")
        data = pd.DataFrame(
            {
                "A": np.random.randn(20),
                "B": np.random.randn(20),
                # time column for dynamics (numeric)
                # Use small time steps to avoid numerical instability (underflow) in Neural ODE
                # with random noise data.
                "time_sec": np.arange(20, dtype=float),
            },
            index=dates,
        )

        # Shuffle the data to ensure order doesn't hide index issues
        data = data.sample(frac=1.0, random_state=42)

        # Inject rk4 to handle random noise without underflow
        engine = InferenceEngine(dynamics_engine=DynamicsEngine(method="rk4"))
        user = UserContext(user_id="test_user", email="test@example.com", claims={"tenant_id": "test_tenant"})
        result = engine.analyze(data, time_col="time_sec", variable_cols=["A", "B"], user_context=user)

        # Check augmented data
        # 1. Should have same index as input
        pd.testing.assert_index_equal(result.augmented_data.index, data.index)

        # 2. Should NOT have any NaNs (which happen if index merge fails)
        assert not result.augmented_data.isnull().values.any()

        # 3. Should contain latent columns
        assert any(c.startswith("Z_") for c in result.augmented_data.columns)

    def test_state_isolation_repeated_calls(self) -> None:
        """
        Verify that calling analyze() multiple times resets state correctly.
        """
        # Inject rk4 to handle random noise without underflow
        engine = InferenceEngine(dynamics_engine=DynamicsEngine(method="rk4"))

        # Run 1: 3 variables
        df1 = pd.DataFrame(
            {"t": np.arange(10), "X": np.random.randn(10), "Y": np.random.randn(10), "Z": np.random.randn(10)}
        )
        user = UserContext(user_id="test_user", email="test@example.com", claims={"tenant_id": "test_tenant"})
        result1 = engine.analyze(df1, "t", ["X", "Y", "Z"], user_context=user)
        assert len(result1.graph.nodes) == 3

        # Run 2: 2 variables
        df2 = pd.DataFrame({"t": np.arange(10), "A": np.random.randn(10), "B": np.random.randn(10)})
        result2 = engine.analyze(df2, "t", ["A", "B"], user_context=user)

        # Check Result 2
        assert len(result2.graph.nodes) == 2
        node_ids = {n.id for n in result2.graph.nodes}
        assert "A" in node_ids and "B" in node_ids
        assert "X" not in node_ids

        # Check Engine Internal State matches Run 2
        assert engine.graph is not None
        assert len(engine.graph.nodes) == 2
        assert "A" in engine.augmented_data.columns  # type: ignore[union-attr]
        assert "X" not in engine.augmented_data.columns  # type: ignore[union-attr]

    def test_empty_variable_input(self) -> None:
        """
        Test behavior when variable_cols is empty.
        DynamicsEngine might accept it (if implemented robustly) or fail.
        If it accepts, LatentMiner needs cols.
        """
        df = pd.DataFrame({"t": np.arange(10), "A": np.random.randn(10)})
        engine = InferenceEngine()

        # If we pass empty variable list, what happens?
        # Dynamics fit checks: data[variable_cols].values
        # If list is empty, values is empty array.
        # Dynamics scaler fit might fail or run.
        # Let's see.
        # Usually we expect a ValueError if there's nothing to model.

        # DynamicsEngine validation: "Input data is empty" if df is empty.
        # But if df is not empty, but variable_cols is empty?
        # df[[]] is Empty DataFrame.
        # So DynamicsEngine might raise "Input data is empty" or similar.

        # However, checking `DynamicsEngine.fit`:
        # `y_raw = df[variable_cols].values`
        # `self.scaler.fit_transform(y_raw)`
        # `ODEFunc(dim)` where dim=0.
        # This likely fails in PyTorch linear layer (input dim 0).

        user = UserContext(user_id="test_user", email="test@example.com", claims={"tenant_id": "test_tenant"})
        with pytest.raises(Exception):  # Likely runtime error or validation error # noqa: B017
            engine.analyze(df, "t", [], user_context=user)

    def test_propagation_of_component_errors(self) -> None:
        """
        Test that errors in sub-components propagate up clearly.
        """
        engine = InferenceEngine()
        # Data with NaNs - DynamicsEngine should raise ValueError
        df = pd.DataFrame({"t": [0, 1, 2], "A": [1.0, np.nan, 3.0]})

        user = UserContext(user_id="test_user", email="test@example.com", claims={"tenant_id": "test_tenant"})
        with pytest.raises(ValueError, match="NaN"):
            engine.analyze(df, "t", ["A"], user_context=user)
