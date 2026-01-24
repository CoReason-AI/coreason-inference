# Copyright (C) 2026 CoReason
# Licensed under the Prosperity Public License 3.0.0

from enum import Enum
from typing import List, Optional, Set, Tuple

import networkx as nx
from pydantic import BaseModel, Field, model_validator


class LoopType(str, Enum):
    """
    Enumeration of feedback loop types.
    """

    POSITIVE_FEEDBACK = "POSITIVE"  # Runaway (Cancer/Cytokine Storm)
    NEGATIVE_FEEDBACK = "NEGATIVE"  # Homeostasis
    NONE = "ACYCLIC"


class RefutationStatus(str, Enum):
    """
    Status of the placebo refutation test.
    """

    PASSED = "PASSED"
    FAILED = "FAILED"


class LoopDynamics(BaseModel):
    """
    Represents a feedback loop within the causal graph.

    Attributes:
        path: List of node IDs forming the loop path (e.g., ["A", "B", "A"]).
        type: The type of feedback dynamics (Positive or Negative).
    """

    path: List[str] = Field(..., min_length=2, description="List of node IDs forming the loop path")
    type: LoopType


class CausalNode(BaseModel):
    """
    Represents a node in the causal graph.

    Attributes:
        id: Unique identifier for the node (e.g., variable name).
        codex_concept_id: ID linking to the Codex Ontology.
        is_latent: Boolean indicating if the node was discovered by the VAE (Latent).
    """

    id: str  # "variable_alt_level"
    codex_concept_id: int  # Linked to Ontology
    is_latent: bool  # True if discovered by VAE


class CausalGraph(BaseModel):
    """
    The discovered Causal Graph, potentially containing cycles (DCG).

    Attributes:
        nodes: List of nodes in the graph.
        edges: List of directed edges as (source, target) tuples.
        loop_dynamics: List of discovered feedback loops.
        stability_score: System stability score (e.g., max real eigenvalue of Jacobian).
    """

    nodes: List[CausalNode]
    edges: List[Tuple[str, str]]
    loop_dynamics: List[LoopDynamics]
    stability_score: float

    @model_validator(mode="after")
    def validate_graph_structure(self) -> "CausalGraph":
        """
        Validates the integrity of the graph structure.

        Checks:
        1. All edges refer to existing nodes.
        2. Node IDs are unique.
        3. Loop dynamics paths correspond to actual edges.

        Raises:
            ValueError: If validation fails.

        Returns:
            CausalGraph: The validated model.
        """
        # 1. Check for duplicate node IDs
        node_ids: Set[str] = set()
        for node in self.nodes:
            if node.id in node_ids:
                raise ValueError(f"Duplicate node ID found: '{node.id}'")
            node_ids.add(node.id)

        # 2. Check edges refer to existing nodes
        existing_edges: Set[Tuple[str, str]] = set(self.edges)
        for u, v in self.edges:
            if u not in node_ids:
                raise ValueError(f"Edge source node '{u}' not found in nodes.")
            if v not in node_ids:
                raise ValueError(f"Edge target node '{v}' not found in nodes.")

        # 3. Check loop dynamics
        for loop in self.loop_dynamics:
            path = loop.path
            # Pydantic validates min_length=2, but we double check logic here if needed

            # Verify path edges exist
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if (u, v) not in existing_edges:
                    raise ValueError(f"Loop path edge ('{u}', '{v}') does not exist in graph edges.")

        return self

    def to_networkx(self) -> nx.DiGraph:
        """
        Converts the CausalGraph to a NetworkX DiGraph.

        Returns:
            nx.DiGraph: A NetworkX directed graph representation with node attributes.
        """
        G = nx.DiGraph()
        # Add nodes with attributes
        for node in self.nodes:
            G.add_node(node.id, codex_concept_id=node.codex_concept_id, is_latent=node.is_latent)
        # Add edges
        G.add_edges_from(self.edges)
        return G


class InterventionResult(BaseModel):
    """
    The result of a causal intervention estimation.

    Attributes:
        patient_id: Identifier of the patient (or 'POPULATION_ATE' / 'ERROR').
        intervention: Description of the intervention (e.g., "do(Treatment)").
        counterfactual_outcome: The estimated causal effect (None if refutation fails).
        confidence_interval: 95% Confidence Interval (Low, High).
        refutation_status: Result of the placebo refuter (PASSED/FAILED).
        cate_estimates: Optional list of Individual Treatment Effects for the population.
    """

    patient_id: str
    intervention: str  # "do(Drug_Dose = 50mg)"
    counterfactual_outcome: Optional[float] = Field(
        ..., description="The estimated causal effect. None if refutation fails."
    )
    confidence_interval: Tuple[float, float]
    refutation_status: RefutationStatus
    cate_estimates: Optional[List[float]] = Field(
        default=None, description="Conditional Average Treatment Effects per patient"
    )


class ExperimentProposal(BaseModel):
    """
    A proposed physical experiment to resolve causal ambiguity.

    Attributes:
        target: The target variable to intervene on (e.g., 'Gene_A').
        action: The action to perform (e.g., 'CRISPR_Knockout').
        confidence_gain: Qualitative assessment of information gain (e.g., 'High').
        rationale: Explanation for why this experiment was selected.
    """

    target: str = Field(..., description="The target variable to intervene on (e.g., 'Gene_A')")
    action: str = Field(..., description="The action to perform (e.g., 'CRISPR_Knockout')")
    confidence_gain: str = Field(..., description="Expected confidence gain (e.g., 'High')")
    rationale: str = Field(..., description="Explanation for the experiment (e.g., 'Resolve direction A-B')")


class ProtocolRule(BaseModel):
    """
    A clinical protocol rule for patient selection.

    Attributes:
        feature: The feature name (e.g., 'Albumin').
        operator: The operator (e.g., '<', '>=', '==').
        value: The threshold value.
        rationale: The reason for this rule (e.g., 'High CATE driver').
    """

    feature: str = Field(..., description="The feature name (e.g., 'Albumin')")
    operator: str = Field(..., description="The operator (e.g., '<', '>=')")
    value: float = Field(..., description="The threshold value")
    rationale: str = Field(..., description="Reason for this rule (e.g., 'High CATE driver')")


class OptimizationOutput(BaseModel):
    """
    Output of the Rule Inductor optimization process.

    Attributes:
        new_criteria: List of optimized inclusion/exclusion rules.
        original_pos: Baseline Probability of Success / Response Rate.
        optimized_pos: Optimized Probability of Success in the selected subgroup.
        safety_flags: List of safety warnings associated with the optimization.
    """

    new_criteria: List[ProtocolRule]
    original_pos: float = Field(..., description="Baseline Probability of Success / Response Rate")
    optimized_pos: float = Field(..., description="Optimized Probability of Success in subgroup")
    safety_flags: List[str] = Field(default_factory=list, description="List of safety warnings")


class VirtualTrialResult(BaseModel):
    """
    Results of a simulated Virtual Phase 3 Trial.

    Attributes:
        cohort_size: Size of the synthetic cohort after filtering by rules.
        safety_scan: List of detected safety risks (pathways to adverse outcomes).
        simulation_result: The estimated treatment effect in the virtual trial.
    """

    cohort_size: int = Field(..., description="Size of the synthetic cohort after filtering")
    safety_scan: List[str] = Field(default_factory=list, description="Detected safety risks")
    simulation_result: Optional[InterventionResult] = Field(
        default=None, description="Result of the virtual trial simulation"
    )
