# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from enum import Enum
from typing import Any, List, Tuple

from pydantic import BaseModel, Field, model_validator


class LoopType(str, Enum):
    POSITIVE_FEEDBACK = "POSITIVE"  # Runaway (Cancer/Cytokine Storm)
    NEGATIVE_FEEDBACK = "NEGATIVE"  # Homeostasis
    NONE = "ACYCLIC"


class RefutationStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"


class CausalNode(BaseModel):
    id: str
    codex_concept_id: int  # Linked to Ontology
    is_latent: bool = False


class LoopDynamics(BaseModel):
    path: List[str]
    type: LoopType

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


class CausalGraph(BaseModel):
    nodes: List[CausalNode]
    edges: List[Tuple[str, str]]
    loop_dynamics: List[LoopDynamics] = Field(default_factory=list)
    stability_score: float

    @model_validator(mode="after")
    def validate_graph_integrity(self) -> "CausalGraph":
        node_ids = {node.id for node in self.nodes}

        # Validate unique node IDs
        if len(node_ids) != len(self.nodes):
            raise ValueError("Node IDs must be unique")

        # Validate edges
        for source, target in self.edges:
            if source not in node_ids:
                raise ValueError(f"Edge source '{source}' not found in nodes")
            if target not in node_ids:
                raise ValueError(f"Edge target '{target}' not found in nodes")

        # Validate loop dynamics
        for loop in self.loop_dynamics:
            for node_id in loop.path:
                if node_id not in node_ids:
                    raise ValueError(f"Loop node '{node_id}' not found in nodes")

        return self


class InterventionResult(BaseModel):
    patient_id: str
    intervention: str  # "do(Drug_Dose = 50mg)"
    counterfactual_outcome: float
    confidence_interval: Tuple[float, float]
    refutation_status: RefutationStatus
