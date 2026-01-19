# Copyright (C) 2026 CoReason
# Licensed under the Prosperity Public License 3.0.0

from enum import Enum
from typing import List, Optional, Set, Tuple

import networkx as nx
from pydantic import BaseModel, Field, model_validator


class LoopType(str, Enum):
    POSITIVE_FEEDBACK = "POSITIVE"  # Runaway (Cancer/Cytokine Storm)
    NEGATIVE_FEEDBACK = "NEGATIVE"  # Homeostasis
    NONE = "ACYCLIC"


class RefutationStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"


class LoopDynamics(BaseModel):
    path: List[str] = Field(..., min_length=2, description="List of node IDs forming the loop path")
    type: LoopType


class CausalNode(BaseModel):
    id: str  # "variable_alt_level"
    codex_concept_id: int  # Linked to Ontology
    is_latent: bool  # True if discovered by VAE


class CausalGraph(BaseModel):
    nodes: List[CausalNode]
    edges: List[Tuple[str, str]]
    loop_dynamics: List[LoopDynamics]
    stability_score: float

    @model_validator(mode="after")
    def validate_graph_structure(self) -> "CausalGraph":
        """
        Validates:
        1. All edges refer to existing nodes.
        2. Node IDs are unique.
        3. Loop dynamics paths correspond to actual edges.
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
        """
        G = nx.DiGraph()
        # Add nodes with attributes
        for node in self.nodes:
            G.add_node(node.id, codex_concept_id=node.codex_concept_id, is_latent=node.is_latent)
        # Add edges
        G.add_edges_from(self.edges)
        return G


class InterventionResult(BaseModel):
    patient_id: str
    intervention: str  # "do(Drug_Dose = 50mg)"
    counterfactual_outcome: float
    confidence_interval: Tuple[float, float]
    refutation_status: RefutationStatus
    cate_estimates: Optional[List[float]] = Field(
        default=None, description="Conditional Average Treatment Effects per patient"
    )


class ExperimentProposal(BaseModel):
    target: str = Field(..., description="The target variable to intervene on (e.g., 'Gene_A')")
    action: str = Field(..., description="The action to perform (e.g., 'CRISPR_Knockout')")
    confidence_gain: str = Field(..., description="Expected confidence gain (e.g., 'High')")
    rationale: str = Field(..., description="Explanation for the experiment (e.g., 'Resolve direction A-B')")


class ProtocolRule(BaseModel):
    feature: str = Field(..., description="The feature name (e.g., 'Albumin')")
    operator: str = Field(..., description="The operator (e.g., '<', '>=')")
    value: float = Field(..., description="The threshold value")
    rationale: str = Field(..., description="Reason for this rule (e.g., 'High CATE driver')")


class OptimizationOutput(BaseModel):
    new_criteria: List[ProtocolRule]
    original_pos: float = Field(..., description="Baseline Probability of Success / Response Rate")
    optimized_pos: float = Field(..., description="Optimized Probability of Success in subgroup")
    safety_flags: List[str] = Field(default_factory=list, description="List of safety warnings")


class VirtualTrialResult(BaseModel):
    cohort_size: int = Field(..., description="Size of the synthetic cohort after filtering")
    safety_scan: List[str] = Field(default_factory=list, description="Detected safety risks")
    simulation_result: Optional[InterventionResult] = Field(
        default=None, description="Result of the virtual trial simulation"
    )
