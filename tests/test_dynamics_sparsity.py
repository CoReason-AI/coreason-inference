import numpy as np
import pandas as pd
import torch

from coreason_inference.analysis.dynamics import DynamicsEngine


def test_sparsity_enforcement() -> None:
    """
    Verifies that L1 regularization suppresses weights for independent variables.
    """
    # 1. Generate Synthetic Data: A -> B, C is independent
    np.random.seed(42)
    torch.manual_seed(42)

    n_points = 50
    dt = 0.1
    t = np.arange(n_points) * dt

    # Simulate
    A = np.zeros(n_points)
    B = np.zeros(n_points)
    C = np.zeros(n_points)

    A[0], B[0], C[0] = 1.0, 0.0, 1.0

    for i in range(1, n_points):
        # Euler integration for ground truth
        # A and B are harmonic oscillators (coupled)
        # A' = B
        # B' = -A
        dA = B[i - 1]
        dB = -A[i - 1]
        # C is independent decay
        dC = -1.0 * C[i - 1]

        A[i] = A[i - 1] + dA * dt
        B[i] = B[i - 1] + dB * dt
        C[i] = C[i - 1] + dC * dt

    data = pd.DataFrame({"time": t, "A": A, "B": B, "C": C})

    # 2. Fit Model with L1 Regularization vs No Regularization
    # We expect weights connecting C to A/B (and vice versa) to be significantly smaller with L1.

    cols = ["A", "B", "C"]

    # Case A: No Regularization
    engine_no_reg = DynamicsEngine(epochs=1000, learning_rate=0.01, l1_lambda=0.0)
    engine_no_reg.fit(data, time_col="time", variable_cols=cols)
    assert engine_no_reg.model is not None
    weights_no_reg = engine_no_reg.model.W.detach().numpy()

    c_mag_no_reg = (
        np.abs(weights_no_reg[0, 2])
        + np.abs(weights_no_reg[1, 2])
        + np.abs(weights_no_reg[2, 0])
        + np.abs(weights_no_reg[2, 1])
    )

    # Case B: With L1 Regularization
    engine_l1 = DynamicsEngine(epochs=1000, learning_rate=0.01, l1_lambda=0.5)
    engine_l1.fit(data, time_col="time", variable_cols=cols)
    assert engine_l1.model is not None
    weights_l1 = engine_l1.model.W.detach().numpy()

    c_mag_l1 = np.abs(weights_l1[0, 2]) + np.abs(weights_l1[1, 2]) + np.abs(weights_l1[2, 0]) + np.abs(weights_l1[2, 1])

    print(f"Weights (No Reg):\n{weights_no_reg}")
    print(f"C Mag (No Reg): {c_mag_no_reg}")

    print(f"Weights (L1):\n{weights_l1}")
    print(f"C Mag (L1): {c_mag_l1}")

    # Assert L1 reduces spurious connections by at least 30%
    assert c_mag_l1 < c_mag_no_reg * 0.7, (
        f"L1 Regularization did not sufficiently reduce spurious weights. NoReg: {c_mag_no_reg}, L1: {c_mag_l1}"
    )
