import numpy as np
import pandas as pd

from coreason_inference.analysis.latent import LatentMiner


def test_latent_variables_present_in_generated_data() -> None:
    """
    Test that LatentMiner.generate() returns both features and latent variables (Z).
    This reproduces the "Data Amnesia" bug where Z was discarded.
    """
    # 1. Setup Data
    data = pd.DataFrame(np.random.randn(50, 3), columns=["A", "B", "C"])

    # 2. Fit LatentMiner
    miner = LatentMiner(latent_dim=2, epochs=1)  # Minimal epochs for speed
    miner.fit(data)

    # 3. Generate Data
    generated = miner.generate(n_samples=10)

    # 4. Assert columns exist
    # Expected columns: A, B, C, Z_0, Z_1
    expected_latent_cols = ["Z_0", "Z_1"]

    missing_cols = [col for col in expected_latent_cols if col not in generated.columns]

    assert not missing_cols, f"Generated data is missing latent columns: {missing_cols}"
    assert len(generated) == 10

    # Verify values are not NaN
    assert not generated[expected_latent_cols].isnull().values.any()
