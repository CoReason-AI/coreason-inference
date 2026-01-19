# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from coreason_inference.analysis.active_scientist import ActiveScientist
from coreason_inference.analysis.dynamics import DynamicsEngine
from coreason_inference.analysis.estimator import CausalEstimator
from coreason_inference.analysis.latent import LatentMiner
from coreason_inference.analysis.rule_inductor import RuleInductor
from coreason_inference.analysis.virtual_simulator import VirtualSimulator
from coreason_inference.schema import (
    CausalGraph,
    ExperimentProposal,
    InterventionResult,
    OptimizationOutput,
    VirtualTrialResult,
)
from coreason_inference.utils.logger import logger


class InferenceResult(BaseModel):
    """
    Container for the results of the full causal inference pipeline.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: CausalGraph
    latents: pd.DataFrame = Field(..., description="Discovered latent variables (Z)")
    proposals: List[ExperimentProposal] = Field(default_factory=list)
    augmented_data: pd.DataFrame = Field(..., description="Original data + Latents")


class InferenceEngine:
    """
    The 'Principal Investigator' / Mechanism Engine.
    Orchestrates the Discover-Represent-Simulate-Act loop.
    """

    def __init__(
        self,
        dynamics_engine: Optional[DynamicsEngine] = None,
        latent_miner: Optional[LatentMiner] = None,
        active_scientist: Optional[ActiveScientist] = None,
        virtual_simulator: Optional[VirtualSimulator] = None,
        rule_inductor: Optional[RuleInductor] = None,
    ) -> None:
        self.dynamics_engine = dynamics_engine or DynamicsEngine()
        self.latent_miner = latent_miner or LatentMiner()
        self.active_scientist = active_scientist or ActiveScientist()
        self.virtual_simulator = virtual_simulator or VirtualSimulator()
        self.rule_inductor = rule_inductor or RuleInductor()

        # Estimator is instantiated per query usually, but we can keep a reference if needed.
        self.estimator: Optional[CausalEstimator] = None

        # State
        self.graph: Optional[CausalGraph] = None
        self.latents: Optional[pd.DataFrame] = None
        self.augmented_data: Optional[pd.DataFrame] = None
        self.cate_estimates: Optional[pd.Series] = None
        self._last_analysis_meta: Dict[str, str] = {}

    @property
    def _estimator(self) -> CausalEstimator:
        if self.augmented_data is None:
            raise ValueError("Data not available. Run analyze() first.")
        return CausalEstimator(self.augmented_data)

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
            # We set self.estimator for backward compatibility or exposure, though usually transient
            self.estimator = self._estimator

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
        return self._estimator.estimate_effect(treatment, outcome, confounders)

    def analyze_heterogeneity(self, treatment: str, outcome: str, confounders: List[str]) -> InterventionResult:
        """
        Estimates Heterogeneous Treatment Effects (CATE) using Causal Forests.
        Stores the estimates for subsequent rule induction.
        """
        logger.info(f"Analyzing Heterogeneity for {treatment} -> {outcome}")

        if self.augmented_data is None:
            raise ValueError("Data not available. Run analyze() first.")

        # Run estimation with 'forest' method
        result = self._estimator.estimate_effect(
            treatment=treatment, outcome=outcome, confounders=confounders, method="forest"
        )

        # Update metadata for rule induction context
        self._last_analysis_meta = {"treatment": treatment, "outcome": outcome}

        # Store CATE estimates
        if result.cate_estimates:
            self.cate_estimates = pd.Series(
                result.cate_estimates, index=self.augmented_data.index, name=f"CATE_{treatment}_{outcome}"
            )
            logger.info(f"Stored {len(result.cate_estimates)} CATE estimates.")
        else:
            logger.warning("No CATE estimates returned from Causal Forest.")
            self.cate_estimates = None

        return result

    def induce_rules(self, feature_cols: Optional[List[str]] = None) -> OptimizationOutput:
        """
        Induces rules to identify Super-Responders based on stored CATE estimates.

        Args:
            feature_cols: Optional list of columns to use as features for rule induction.
                          If None, uses all numeric columns from augmented_data (excluding metadata).
        """
        if self.cate_estimates is None:
            raise ValueError("No CATE estimates found. Run analyze_heterogeneity() first.")

        if self.augmented_data is None:
            raise ValueError("Data not available.")

        # Determine features
        if feature_cols:
            features = self.augmented_data[feature_cols]
        else:
            # Select numeric types, exclude potential metadata/targets
            # This is a heuristic. Ideally user provides features.
            features = self.augmented_data.select_dtypes(include=["number"])

            # Prevent Data Leakage: Exclude treatment and outcome if known
            exclusions = []
            if "treatment" in self._last_analysis_meta:
                exclusions.append(self._last_analysis_meta["treatment"])
            if "outcome" in self._last_analysis_meta:
                exclusions.append(self._last_analysis_meta["outcome"])

            features = features.drop(columns=[c for c in exclusions if c in features.columns])

        logger.info(f"Inducing rules using {features.shape[1]} features (excluded: {self._last_analysis_meta}).")

        self.rule_inductor.fit(features, self.cate_estimates)
        return self.rule_inductor.induce_rules_with_data(features, self.cate_estimates)

    def run_virtual_trial(
        self,
        optimization_result: OptimizationOutput,
        treatment: str,
        outcome: str,
        confounders: List[str],
        n_samples: int = 1000,
        adverse_outcomes: List[str] | None = None,
    ) -> VirtualTrialResult:
        """
        Runs a virtual trial: Generates synthetic cohort based on optimized rules,
        scans for safety risks, and simulates the treatment effect.

        Args:
            optimization_result: Output from RuleInductor containing new_criteria.
            treatment: Treatment variable name.
            outcome: Outcome variable name.
            confounders: List of confounder names.
            n_samples: Number of digital twins to generate.
            adverse_outcomes: List of adverse outcome names for safety scanning.

        Returns:
            VirtualTrialResult: Result containing cohort size, safety flags, and effect estimate.
        """
        if self.latent_miner.model is None or self.graph is None:
            raise ValueError("Model not fitted. Run analyze() first.")

        logger.info("Running Virtual Phase 3 Trial...")

        # 1. Generate Synthetic Cohort
        cohort = self.virtual_simulator.generate_synthetic_cohort(
            miner=self.latent_miner,
            n_samples=n_samples,
            rules=optimization_result.new_criteria,
        )

        if cohort.empty:
            logger.warning("Virtual trial aborted: Cohort is empty after filtering.")
            return VirtualTrialResult(cohort_size=0, safety_scan=[], simulation_result=None)

        # 2. Safety Scan
        safety_flags = []
        if adverse_outcomes:
            safety_flags = self.virtual_simulator.scan_safety(
                graph=self.graph, treatment=treatment, adverse_outcomes=adverse_outcomes
            )

        # 3. Simulate Effect
        try:
            sim_result = self.virtual_simulator.simulate_trial(
                cohort=cohort,
                treatment=treatment,
                outcome=outcome,
                confounders=confounders,
            )
        except Exception as e:
            logger.error(f"Virtual trial simulation failed: {e}")
            sim_result = None

        return VirtualTrialResult(cohort_size=len(cohort), safety_scan=safety_flags, simulation_result=sim_result)
