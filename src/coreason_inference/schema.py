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

from pydantic import BaseModel


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


class InterventionResult(BaseModel):
    """
    Represents the result of a causal intervention simulation.
    """

    patient_id: str
    intervention: str  # "do(Drug_Dose = 50mg)"
    counterfactual_outcome: float
    confidence_interval: Tuple[float, float]
    refutation_status: str  # "PASSED" or "FAILED"
