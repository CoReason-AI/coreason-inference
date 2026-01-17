# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from coreason_inference.analysis.active_scientist import ActiveScientist
from coreason_inference.analysis.dynamics import DynamicsEngine
from coreason_inference.analysis.estimator import CausalEstimator
from coreason_inference.analysis.latent import LatentMiner
from coreason_inference.schema import CausalGraph, ExperimentProposal, InterventionResult
from coreason_inference.utils.logger import logger


class InferenceResult(BaseModel):
    """
    Container for the results of the full causal inference pipeline.
    """

    graph: CausalGraph
    latents: pd.DataFrame = Field(..., description="Discovered latent variables (Z)")
    proposals: List[ExperimentProposal] = Field(default_factory=list)
    augmented_data: pd.DataFrame = Field(..., description="Original data + Latents")

    class Config:
        arbitrary_types_allowed = True


class InferenceEngine:
    """
    The 'Principal Investigator' / Mechanism Engine.
    Orchestrates the Discover-Represent-Simulate-Act loop.
    """

    def __init__(self) -> None:
        self.dynamics_engine = DynamicsEngine()
        self.latent_miner = LatentMiner()
        self.active_scientist = ActiveScientist()
        # Estimator is instantiated per query usually, but we can keep a reference if needed.
        self.estimator: Optional[CausalEstimator] = None

        # State
        self.graph: Optional[CausalGraph] = None
        self.latents: Optional[pd.DataFrame] = None
        self.augmented_data: Optional[pd.DataFrame] = None

    def analyze(
        self,
        data: pd.DataFrame,
        time_col: str,
        variable_cols: List[str],
        estimate_effect_for: Optional[tuple[str, str]] = None,
    ) -> InferenceResult:
        """
        Executes the full causal discovery pipeline.

        Args:
            data: Input dataframe containing time-series data.
            time_col: Name of the time column.
            variable_cols: List of variable columns to analyze.
            estimate_effect_for: Optional tuple (treatment, outcome) to run estimation for.

        Returns:
            InferenceResult: The consolidated results.
        """
        logger.info("Starting Inference Engine Pipeline...")

        # 1. Discover (Dynamics)
        logger.info("Step 1: Discover (Dynamics & Loops)")
        self.dynamics_engine.fit(data, time_col, variable_cols)
        self.graph = self.dynamics_engine.discover_loops()
        logger.info(
            f"Discovered Graph with {len(self.graph.edges)} edges and stability score {self.graph.stability_score}"
        )

        # 2. Represent (Latent Learning)
        logger.info("Step 2: Represent (Latent Mining)")
        # We use the variable columns for latent discovery
        # (Assuming latents explain the variance in these observed vars)
        observation_data = data[variable_cols]
        self.latent_miner.fit(observation_data)
        self.latents = self.latent_miner.discover_latents(observation_data)

        # Augment Data
        # We merge on index. Ensure indices align.
        self.augmented_data = pd.concat([data, self.latents], axis=1)
        logger.info(f"Discovered {self.latents.shape[1]} latent variables. Data augmented.")

        # 3. Act (Active Experimentation)
        logger.info("Step 3: Act (Active Scientist)")
        # The Active Scientist works on the Observational Data (augmented?)
        # Ideally, it uses the augmented data to find ambiguity in the full system.
        # However, our ActiveScientist implementation (PC algorithm) might struggle with too many vars.
        # Let's pass the augmented data (Variables + Latents).

        # We need to filter only numeric columns that are relevant
        analysis_cols = variable_cols + list(self.latents.columns)
        analysis_data = self.augmented_data[analysis_cols]

        self.active_scientist.fit(analysis_data)
        proposals = self.active_scientist.propose_experiments()
        logger.info(f"Generated {len(proposals)} experiment proposals.")

        # 4. Simulate (Estimation) - Optional Trigger
        if estimate_effect_for:
            treatment, outcome = estimate_effect_for
            logger.info(f"Step 4: Simulate (Estimating effect of {treatment} on {outcome})")
            self.estimator = CausalEstimator(self.augmented_data)

            # Use all other variables as potential confounders (excluding time)
            # This is a naive selection; usually, we use the graph to select the adjustment set.
            # For this atomic unit, we pass other variables as confounders.
            confounders = [c for c in analysis_cols if c not in [treatment, outcome]]

            # Check if valid
            if treatment not in self.augmented_data.columns or outcome not in self.augmented_data.columns:
                logger.warning(f"Skipping estimation: {treatment} or {outcome} not in data.")
            else:
                try:
                    result = self.estimator.estimate_effect(treatment, outcome, confounders)
                    logger.info(
                        f"Effect Result: {result.counterfactual_outcome} (Refutation: {result.refutation_status})"
                    )
                except Exception as e:
                    logger.error(f"Estimation failed during pipeline: {e}")

        # Assemble Result
        result_obj = InferenceResult(
            graph=self.graph,
            latents=self.latents,
            proposals=proposals,
            augmented_data=self.augmented_data,
        )

        logger.info("Inference Pipeline Completed.")
        return result_obj

    def explain_latents(self) -> pd.DataFrame:
        """
        Returns the interpretation of the latent variables (SHAP values).
        Uses the data stored in the latent miner (if any) or requires state.
        Currently returns an empty DataFrame as placeholder for future atomic unit.
        """
        if self.latent_miner.model is None:
            raise ValueError("Pipeline not run or latent miner not fitted.")

        # In a real implementation, we would store the input columns used for fit
        # and pass the corresponding data from augmented_data to interpret_latents.
        # For this atomic unit, we defer specific interpretation wiring.
        return pd.DataFrame()

    def estimate_effect(self, treatment: str, outcome: str, confounders: List[str]) -> InterventionResult:
        """
        Direct access to the CausalEstimator (Simulate).
        """
        if self.augmented_data is None:
            raise ValueError("Data not available. Run analyze() first.")

        estimator = CausalEstimator(self.augmented_data)
        return estimator.estimate_effect(treatment, outcome, confounders)
