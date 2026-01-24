# coreason-inference

**Causal Discovery, Representation Learning, Active Experimentation, & Trial Optimization**

[![License](https://img.shields.io/badge/license-Prosperity%203.0-blue)](https://prosperitylicense.com/versions/3.0.0)
[![Build Status](https://github.com/CoReason-AI/coreason_inference/actions/workflows/build.yml/badge.svg)](https://github.com/CoReason-AI/coreason_inference/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation](https://img.shields.io/badge/docs-product_requirements-blue)](docs/product_requirements.md)

**coreason-inference** is the **Causal Intelligence** engine of the CoReason ecosystem. Unlike probabilistic models that predict correlation, this engine is designed to uncover **Mechanism** and **Heterogeneity**. It serves as the "Principal Investigator," discovering biological feedback loops, identifying latent confounders, and optimizing clinical trials through causal stratification and virtual simulation.

## Features

*   **Cyclic Discovery (Dynamics Engine):** Uses Neural ODEs to discover feedback loops and system dynamics in biological systems (Directed Cyclic Graphs).
*   **Latent Phenotyping (Latent Miner):** Disentangles hidden confounders using Causal VAEs to identify unmeasured variables driving outcomes.
*   **Heterogeneous Stratification (Causal Estimator):** Estimates Individual Treatment Effects (CATE) using Causal Forests to identify Super-Responders.
*   **Active Experimentation (Active Scientist):** Proposes physical experiments (e.g., Gene Knockouts) to resolve causal ambiguity using Information Gain heuristics.
*   **Protocol Optimization (Rule Inductor):** Translates CATE scores into human-readable clinical protocol rules to maximize Phase 3 Probability of Success (PoS).
*   **Virtual Trials (Virtual Simulator):** Simulates Phase 3 trials in-silico using synthetic "Digital Twins" to predict efficacy and scan for safety risks.

## Installation

```bash
pip install coreason-inference
```

## Usage

Here is a quick example of how to initialize and use the `InferenceEngine`:

```python
import pandas as pd
from coreason_inference.engine import InferenceEngine

# 1. Initialize the Engine
engine = InferenceEngine()

# 2. Load your data
# Data should contain time-series or observational data
data = pd.read_csv("patient_data.csv")
variable_cols = ["Glucose", "Insulin", "HbA1c"]
time_col = "Time"

# 3. Analyze: Discover Dynamics & Latents
result = engine.analyze(
    data=data,
    time_col=time_col,
    variable_cols=variable_cols,
    estimate_effect_for=("Insulin", "Glucose")
)

# 4. Inspect the Causal Graph
print(f"Discovered Graph Edges: {result.graph.edges}")
print(f"Detected Loops: {result.graph.loop_dynamics}")

# 5. Optimize for Heterogeneity (Identify Super-Responders)
# Estimate CATE for a specific treatment
engine.analyze_heterogeneity(
    treatment="Insulin",
    outcome="Glucose",
    confounders=["Age", "BMI"] + list(result.latents.columns)
)

# Induce rules to find the best subgroup
optimization_output = engine.induce_rules()

print("Recommended Protocol Criteria:")
for rule in optimization_output.new_criteria:
    print(f" - {rule.feature} {rule.operator} {rule.value} ({rule.rationale})")

print(f"Projected Probability of Success: {optimization_output.optimized_pos:.2f}")
```

## Documentation

For detailed requirements and architectural philosophy, please refer to the [Product Requirements Document](docs/product_requirements.md).
