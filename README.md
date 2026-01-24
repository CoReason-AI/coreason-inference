# coreason-inference

**The Causal Intelligence Engine for Discovery, Stratification, and Trial Optimization.**

`coreason-inference` is the deterministic "Principal Investigator" of the CoReason ecosystem. Unlike probabilistic models that predict correlation, this engine uncovers **Mechanism** and **Heterogeneity**. It enables the **Discover-Stratify-Simulate-Act Loop** to discover biological feedback loops, stratify patient subgroups (Super-Responders), and simulate counterfactual outcomes.

---

[![License: Prosperity 3.0](https://img.shields.io/badge/License-Prosperity%203.0-blue)](https://prosperitylicense.com/versions/3.0.0)
[![CI](https://github.com/CoReason-AI/coreason_inference/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason_inference/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Features

*   **The Dynamics Engine (Cyclic Discovery):** Discovers feedback loops and system dynamics using **Neural ODEs** and structure learning (NOTEARS-Cyclic).
*   **The Latent Miner (Representation Learning):** Discovers hidden confounders using **Causal Variational Autoencoders (VAEs)** with independence constraints.
*   **The De-Confounder & Stratifier (Effect Estimation):** Calculates True Effect and Individual Response (CATE) using **Double Machine Learning (DML)** and **Causal Forests**.
*   **The Active Scientist (Experimental Design):** Resolves causal ambiguity by identifying the Markov Equivalence Class and proposing optimal experiments (Interventions).
*   **The Rule Inductor (TPP Optimizer):** Translates complex CATE scores into human-readable **Clinical Protocols** to optimize Phase 3 Probability of Success.
*   **The Virtual Simulator (Safety & Efficacy):** "Re-runs" Phase 2 trials *in silico* by generating "Digital Twins" and scanning for toxicity pathways.

## Installation

```bash
pip install -r requirements.txt
```

or via Poetry:

```bash
poetry install
```

## Usage

```python
import pandas as pd
from coreason_inference.engine import InferenceEngine

# 1. Initialize the engine
engine = InferenceEngine()

# 2. Load your time-series observational data
# Ensure data contains a time column and variable columns
data = pd.DataFrame({
    "time": [0, 1, 2, 3, 4],
    "Drug_Dose": [10, 20, 10, 0, 10],
    "Blood_Pressure": [120, 110, 115, 125, 118],
    "Biomarker_X": [0.5, 0.4, 0.45, 0.6, 0.55]
})

# 3. Execute the full discovery pipeline
result = engine.analyze(
    data=data,
    time_col="time",
    variable_cols=["Drug_Dose", "Blood_Pressure", "Biomarker_X"],
    estimate_effect_for=("Drug_Dose", "Blood_Pressure")
)

# 4. Explore Results
print("Discovered Graph Edges:", result.graph.edges)
print("Feedback Loops:", result.graph.loop_dynamics)

# 5. Analyze Heterogeneity & Induce Rules
# (Requires sufficient data size for valid estimation)
try:
    forest_result = engine.analyze_heterogeneity(
        treatment="Drug_Dose",
        outcome="Blood_Pressure",
        confounders=["Biomarker_X"]
    )

    optimization = engine.induce_rules()
    print("Optimized Inclusion Criteria:", optimization.new_criteria)
except ValueError as e:
    print(f"Skipping heterogeneity analysis: {e}")
```

## License

This software is dual-licensed under the **Prosperity Public License 3.0**.
Commercial use beyond a 30-day trial requires a separate license.
See `LICENSE` for details.
