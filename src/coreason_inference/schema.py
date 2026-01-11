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
    Enum representing the type of feedback loop in a causal graph.
    """

    POSITIVE_FEEDBACK = "POSITIVE"  # Runaway (Cancer/Cytokine Storm)
    NEGATIVE_FEEDBACK = "NEGATIVE"  # Homeostasis
    NONE = "ACYCLIC"


class CausalNode(BaseModel):
    """
    Represents a node in the causal graph.
    """

    id: str
    codex_concept_id: int  # Linked to Ontology
    is_latent: bool  # True if discovered by VAE


class CausalGraph(BaseModel):
    """
    Represents the causal graph structure including nodes, edges, and dynamics.
    """

    nodes: List[CausalNode]
    edges: List[Tuple[str, str]]
    loop_dynamics: List[Dict[str, Any]]  # { "path": ["A","B","A"], "type": "NEGATIVE" }
    stability_score: float

    @model_validator(mode="after")
    def check_graph_integrity(self) -> "CausalGraph":
        """
        Validates the integrity of the graph structure.
        Ensures all nodes have unique IDs and that edges/loops refer to existing nodes.
        """
        node_ids = {node.id for node in self.nodes}

        # Check for duplicates
        if len(node_ids) != len(self.nodes):
            raise ValueError("Duplicate node IDs found in graph.")

        # Check edges
        for u, v in self.edges:
            if u not in node_ids:
                raise ValueError(f"Edge source '{u}' not found in nodes.")
            if v not in node_ids:
                raise ValueError(f"Edge target '{v}' not found in nodes.")

        # Check loop dynamics
        for loop in self.loop_dynamics:
            path = loop.get("path", [])
            if not isinstance(path, list):
                raise ValueError("Loop path must be a list.")
            for node_id in path:
                if node_id not in node_ids:
                    raise ValueError(f"Loop path node '{node_id}' not found in nodes.")

        return self


class InterventionResult(BaseModel):
    """
    Represents the result of a causal intervention simulation.
    """

    patient_id: str
    intervention: str  # "do(Drug_Dose = 50mg)"
    counterfactual_outcome: float
    confidence_interval: Tuple[float, float]
    refutation_status: str  # "PASSED" or "FAILED"
