# **Product Requirements Document: coreason-inference**

**Domain:** Causal Discovery, Representation Learning, Active Experimentation, & Trial Optimization

**Architectural Role:** The "Principal Investigator" / The Mechanism Engine

**Core Philosophy:** "Biology has loops. Data has holes. Patients are unique. To cure, we must stratify and intervene."

**Dependencies:** coreason-refinery (Data), coreason-graph-nexus (Topology), coreason-codex (Ontology), coreason-chronos (TimeSeries), dowhy, econml, torchdiffeq, causal-learn

## ---

**1. Executive Summary**

coreason-inference is the **Causal Intelligence** engine of the CoReason ecosystem. While standard LLMs are probabilistic engines that predict the next token (Correlation), coreason-inference is a deterministic engine designed to uncover **Mechanism** and **Heterogeneity**.

It serves two distinct but related masters:

1. **The Scientist:** Discovers biological feedback loops and proposes wet-lab experiments to resolve causal ambiguity.
2. **The Strategist:** Optimizes Phase 3 trials by identifying **Super-Responders** (Target Patient Profiles) and simulating counterfactual outcomes (Virtual Trials) to maximize Probability of Success (PoS).

## **2. Functional Philosophy**

The agent must implement the **Discover-Stratify-Simulate-Act Loop**:

1. **Cyclic Discovery (System Dynamics):** Biological systems are defined by feedback. We move beyond DAGs to **Directed Cyclic Graphs (DCGs)** using **Neural ODEs**, modeling (![][image1]).
2. **Latent Phenotyping (Representation):** Raw data is incomplete. We use **Causal Disentanglement (VAEs)** to identify *Latent Confounders* (e.g., "Frailty") that drive differential outcomes.
3. **Heterogeneous Stratification (TPP):** Drugs don't work for "averages." We use **Causal Forests** to estimate Individual Treatment Effects (CATE), isolating specific subgroups (Super-Responders) from the noise.
4. **Active Experimentation (VoI):** When data is ambiguous, the AI calculates the **Value of Information** to recommend physical experiments (e.g., "Knockout Gene Z") to distinct causal structures.

## ---

**3. Core Functional Requirements (Component Level)**

### **3.1 The Dynamics Engine (Cyclic Discovery)**

* **Goal:** Discovers feedback loops and system dynamics.
* **Problem:** Standard causal algorithms assume Acyclicity. Biology has Cycles (![][image2]).
* **Mechanism:** Uses **Neural Ordinary Differential Equations (Neural ODEs)** combined with structure learning (NOTEARS-Cyclic).
* **Output:** A DCG stored in graph-nexus with properties defining the feedback nature (e.g., loop_type: "negative_feedback").

### **3.2 The Latent Miner (Representation Learning)**

* **Goal:** Discovers hidden variables (Confounders).
* **Problem:** We see "Death" and "Dose," but miss the unmeasured "Genetic Mutation" causing both.
* **Mechanism:** Uses a **Causal Variational Autoencoder (VAE)** with independence constraints (iSL) to disentangle the hidden factor ![][image3].
* **Value:** Allows unbiased effect estimation even when critical columns are missing (Proxy controls).

### **3.3 The De-Confounder & Stratifier (Effect Estimation)**

* **Goal:** Calculates *True* Effect and *Individual* Response.
* **Mechanism:** Uses **Double Machine Learning (DML)** and **Causal Forests**.
  * **ATE (Average Treatment Effect):** Subtracts residuals to isolate pure causal signal.
  * **CATE (Conditional ATE):** Estimates effect per patient to identify **Super-Responders** (High CATE) vs Non-Responders.
* **Refutation:** Every estimate must pass a **Placebo Test**.

### **3.4 The Active Scientist (Experimental Design)**

* **Goal:** Resolves causal ambiguity.
* **Logic:**
  1. Identifies the **Markov Equivalence Class** (all valid graphs fitting the data).
  2. If set size > 1, simulates interventions.
  3. Selects the Intervention ![][image4] that maximally splits the set (Information Gain).
* **Output:** ExperimentProposal(target="Gene_A", action="Knockout", confidence_gain="High").

### **3.5 The Rule Inductor (TPP Optimizer) [NEW]**

* **Goal:** Translates complex CATE scores into human-readable Clinical Protocols.
* **Problem:** Mathematical models are not executable protocols.
* **Mechanism:** Uses **PRIM (Patient Rule Induction Method)** or pruned Decision Trees on CATE scores.
* **Output:** Optimized Inclusion/Exclusion criteria (e.g., "Exclude Patients with BMI > 30") that maximize Phase 3 PoS.

### **3.6 The Virtual Simulator (Safety & Efficacy) [NEW]**

* **Goal:** "Re-runs" Phase 2 trials in silico with new criteria.
* **Mechanism:**
  1. **Synthetic Cohort:** Generates "Digital Twins" matching the Rule Inductor's criteria.
  2. **Safety Scan:** Propagates drug effects through graph-nexus toxicity pathways.
* **Value:** Predicts dropout rates and severe adverse events before the trial starts.

## ---

**4. Integration Requirements**

* **coreason-refinery:** Provides tabular/time-series data. Must map columns to codex IDs.
* **coreason-graph-nexus:** Stores the Causal Graph (including Cyclic edges) and Knowledge Graph for toxicity traversal.
* **coreason-chronos:** Provides dense time-series vectors (imputation) for the Neural ODE solver.
* **coreason-protocol:** Consumes the **Rules** from Section 3.5 to auto-draft the Clinical Study Protocol text.

## ---

**5. User Stories**

### **Story A: The "Homeostasis" Check (Cyclic Graph)**

**Context:** Increasing dosage initially lowers blood sugar, but it spikes back up after 6 hours.

**Action:** inference.discover_dynamics().

**Result:** Fits a Neural ODE. Identifies a **Negative Feedback Loop** involving Insulin sensitivity.

**Outcome:** "System is adapting. Switch to pulsatile dosing."

### **Story B: The "Hidden Factor" (Latent Discovery)**

**Context:** Two sites produce the same biologic; Site B has higher adverse events.

**Action:** inference.discover_latent_confounders(data=[Site_A, Site_B]).

**Result:** VAE identifies Latent Vector ![][image3]. Decoding reveals ![][image3] correlates with "Shift Change Timing."

**Outcome:** Operational error found, not chemical.

### **Story C: The "Next Best Experiment" (Active Learning)**

**Context:** Gene X and Y are correlated. Directionality is unknown.

**Action:** inference.propose_experiment().

**Output:** "Markov Equivalence found. **Action:** Knockdown Gene X. If Y drops, X causes Y."

### **Story D: The "Phase 3 Rescue" (Optimization)**

**Context:** Phase 2 showed mediocre efficacy (30%). Asset is at risk.

**Action:** inference.analyze_heterogeneity() + inference.induce_rules().

**Result:** "Drug works in 85% of patients with *Low Albumin*. Recommend updating Inclusion Criteria."

**Outcome:** Phase 3 PoS increases from 30% to 75%.

## ---

**6. Data Schema**

### **CausalGraphDef**

```python
class LoopType(str, Enum):
    POSITIVE_FEEDBACK = "POSITIVE"
    NEGATIVE_FEEDBACK = "NEGATIVE"
    NONE = "ACYCLIC"

class CausalNode(BaseModel):
    id: str
    codex_concept_id: int
    is_latent: bool

class CausalGraph(BaseModel):
    nodes: List[CausalNode]
    edges: List[Tuple[str, str]]
    loop_dynamics: List[dict] # { "path": ["A","B","A"], "type": "NEGATIVE" }
```

### **OptimizationResult (New)**

```python
class ProtocolRule(BaseModel):
    feature: str             # "Albumin"
    operator: str            # "<"
    value: float             # 3.5
    rationale: str           # "High CATE driver"

class OptimizationOutput(BaseModel):
    new_criteria: List[ProtocolRule]
    original_pos: float      # 0.30
    optimized_pos: float     # 0.75
    safety_flags: List[str]  # ["Risk of Renal Failure increased by 5%"]
```
