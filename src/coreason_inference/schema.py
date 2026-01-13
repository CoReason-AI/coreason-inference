# Copyright (C) 2026 CoReason
# Licensed under the Prosperity Public License 3.0.0

from enum import Enum
from typing import List, Set, Tuple

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


class InterventionResult(BaseModel):
    patient_id: str
    intervention: str  # "do(Drug_Dose = 50mg)"
    counterfactual_outcome: float
    confidence_interval: Tuple[float, float]
    refutation_status: RefutationStatus
