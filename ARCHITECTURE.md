# The Architecture and Utility of coreason-inference

### 1. The Philosophy (The Why)

Biology is not a straight line; it is a tangled web of feedback loops. Standard data analysis often assumes acyclicity—that A causes B in a single direction—and treats patients as averages. However, the author of `coreason-inference` recognizes a critical truth: **"Biology has loops. Data has holes. Patients are unique. To cure, we must stratify and intervene."**

`coreason-inference` is built to be the "Principal Investigator" of the CoReason ecosystem. It rejects the passive role of merely predicting the next data point. Instead, it adopts a deterministic, mechanistic approach to uncover the "why" behind the data. Its mission is two-fold: to act as a **Scientist** that discovers biological feedback dynamics and proposes experiments to resolve ambiguity, and as a **Strategist** that saves clinical trials by finding specific subgroups ("Super-Responders") where a drug truly works. It solves the pain point of "average treatment effects" failing in Phase 3 trials by shifting the focus to Mechanism and Heterogeneity.

### 2. Under the Hood (The Dependencies & logic)

The package relies on a heavy-hitting stack designed for causal rigor and dynamic modeling:

*   **`torchdiffeq` & `torch`**: These power the **Dynamics Engine**. By using Neural Ordinary Differential Equations (Neural ODEs), the package models the continuous-time evolution of biological systems, capable of capturing the cyclic feedback loops that standard DAGs (Directed Acyclic Graphs) miss.
*   **`dowhy`**: Serves as the orchestration layer for causal reasoning, ensuring that every effect estimate is formally defined and, crucially, subjected to refutation tests (like Placebo Refuters) to guarantee validity.
*   **`econml`**: Provides the statistical muscle for the **De-Confounder & Stratifier**. It utilizes Causal Forests and Double Machine Learning to estimate Heterogeneous Treatment Effects (CATE), enabling the identification of patient subgroups with distinct responses.
*   **`causal-learn`**: Implements the PC/FCI algorithms used by the **Active Scientist** to discover the causal skeleton and identify Markov Equivalence Classes, which drives the logic for proposing new experiments.

The internal logic is orchestrated by the `InferenceEngine` class, which implements a "Discover-Stratify-Simulate-Act" loop. It moves from discovering system dynamics (using Neural ODEs), to learning latent representations of confounders (using VAEs), to estimating individual effects, and finally to inducing human-readable rules for trial optimization.

### 3. In Practice (The How)

The `InferenceEngine` allows developers to run a full causal discovery and optimization pipeline with a clean, high-level API.

```python
import pandas as pd
from coreason_inference.engine import InferenceEngine

# 1. Initialize the Principal Investigator
# The engine orchestrates Dynamics, Latent Mining, and Strategy.
engine = InferenceEngine()

# 2. Discover Mechanism (Loops & Latents)
# We feed it time-series data to discover feedback loops (Neural ODEs)
# and latent confounders (VAEs) that drive the system.
results = engine.analyze(
    data=df_patient_vitals,
    time_col="hours_since_dose",
    variable_cols=["insulin", "glucose", "cortisol"]
)

# 3. Stratify & Optimize (The Phase 3 Rescue)
# Instead of looking at the average, we analyze heterogeneity to find
# who really responds to the treatment.
engine.analyze_heterogeneity(
    treatment="experimental_drug",
    outcome="survival_months",
    confounders=["age", "biomarker_A", "biomarker_B"]
)

# 4. Generate Clinical Protocol Rules
# The Rule Inductor translates complex math into a clear protocol,
# finding the criteria that maximize the Probability of Success (PoS).
optimization = engine.induce_rules()
print(f"Original PoS: {optimization.original_pos:.2f}")
print(f"Optimized PoS: {optimization.optimized_pos:.2f}")
# Example Output:
# Original PoS: 0.30
# Optimized PoS: 0.75
# Rule: "Include if biomarker_A > 1.5"
```

In this workflow, the engine automates the heavy lifting: fitting differential equations, running causal forests, and ensuring statistical validity, allowing the user to focus on the biological insights and clinical strategy.
