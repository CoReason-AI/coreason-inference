# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from enum import Enum
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, model_validator


class LoopType(str, Enum):
    """
    Defines the type of feedback loop in the causal graph.
    """

    POSITIVE_FEEDBACK = "POSITIVE"  # Runaway (Cancer/Cytokine Storm)
    NEGATIVE_FEEDBACK = "NEGATIVE"  # Homeostasis
    NONE = "ACYCLIC"


class CausalNode(BaseModel):
    """
    Represents a node in the causal graph.
    """

    id: str  # "variable_alt_level"
    codex_concept_id: int  # Linked to Ontology
    is_latent: bool  # True if discovered by VAE


class CausalGraph(BaseModel):
    """
    Represents the entire causal graph structure, including nodes, edges, and dynamic loops.
    """

    nodes: List[CausalNode]
    edges: List[Tuple[str, str]]
    loop_dynamics: List[Dict[str, Any]]  # { "path": ["A","B","A"], "type": "NEGATIVE" }
    stability_score: float

    @model_validator(mode="after")
    def check_edges_reference_existing_nodes(self) -> "CausalGraph":
        node_ids = {node.id for node in self.nodes}
        for source, target in self.edges:
            if source not in node_ids:
                raise ValueError(f"Edge source '{source}' not found in nodes.")
            if target not in node_ids:
                raise ValueError(f"Edge target '{target}' not found in nodes.")
        return self


class InterventionResult(BaseModel):
    """
    Stores the result of a counterfactual intervention.
    """

    patient_id: str
    intervention: str  # "do(Drug_Dose = 50mg)"
    counterfactual_outcome: float
    confidence_interval: Tuple[float, float]
    refutation_status: str  # "PASSED" or "FAILED"
