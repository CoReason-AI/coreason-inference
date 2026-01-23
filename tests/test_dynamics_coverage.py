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


def test_dynamics_input_validation() -> None:
    engine = DynamicsEngine()

    # 1. Empty Data (covered by Line 84-85?)
    with pytest.raises(ValueError, match="Input data is empty"):
        engine.fit(pd.DataFrame(), "time", ["A"])

    # 2. NaN Values (Line 87-88?)
    df_nan = pd.DataFrame({"time": [0, 1], "A": [1, np.nan]})
    with pytest.raises(ValueError, match="Input data contains NaN values"):
        engine.fit(df_nan, "time", ["A"])

    # 3. Insufficient Data (Line 99?)
    df_short = pd.DataFrame({"time": [0], "A": [1]})
    with pytest.raises(ValueError, match="Insufficient data points"):
        engine.fit(df_short, "time", ["A"])


def test_dynamics_l1_regularization() -> None:
    # Test L1 lambda path (Line 133)
    # Use very small data/epochs to be fast
    df = pd.DataFrame({"time": [0, 1, 2], "A": [1, 0.5, 0.25]})

    # l1_lambda > 0
    engine = DynamicsEngine(l1_lambda=0.1, epochs=1, method="rk4")
    engine.fit(df, "time", ["A"])

    assert engine.model is not None
    # Just ensure no error raised and fit completed


def test_dynamics_constructor_validation() -> None:
    with pytest.raises(ValueError, match="l1_lambda must be non-negative"):
        DynamicsEngine(l1_lambda=-0.1)
