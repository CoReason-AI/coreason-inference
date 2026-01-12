# Copyright (C) 2026 CoReason
# Licensed under the Prosperity Public License 3.0.0

from enum import Enum
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, model_validator


class LoopType(str, Enum):
    POSITIVE_FEEDBACK = "POSITIVE"  # Runaway (Cancer/Cytokine Storm)
    NEGATIVE_FEEDBACK = "NEGATIVE"  # Homeostasis
    NONE = "ACYCLIC"


class CausalNode(BaseModel):
    id: str  # "variable_alt_level"
    codex_concept_id: int  # Linked to Ontology
    is_latent: bool  # True if discovered by VAE


class CausalGraph(BaseModel):
    nodes: List[CausalNode]
    edges: List[Tuple[str, str]]
    loop_dynamics: List[Dict[str, Any]]  # { "path": ["A","B","A"], "type": "NEGATIVE" }
    stability_score: float

    @model_validator(mode="after")
    def check_edges_refer_to_existing_nodes(self) -> "CausalGraph":
        node_ids = {node.id for node in self.nodes}
        for u, v in self.edges:
            if u not in node_ids:
                raise ValueError(f"Edge source node '{u}' not found in nodes.")
            if v not in node_ids:
                raise ValueError(f"Edge target node '{v}' not found in nodes.")
        return self


class InterventionResult(BaseModel):
    patient_id: str
    intervention: str  # "do(Drug_Dose = 50mg)"
    counterfactual_outcome: float
    confidence_interval: Tuple[float, float]
    refutation_status: str  # "PASSED" or "FAILED"
