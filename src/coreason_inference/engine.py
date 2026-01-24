# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import Any, Dict, List, Optional, cast

import anyio
import httpx
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
    """Container for the results of the full causal inference pipeline.

    Attributes:
        graph: The discovered causal graph (including loops).
        latents: Discovered latent variables (Z).
        proposals: List of experiment proposals from Active Scientist.
        augmented_data: Original data augmented with latent variables.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: CausalGraph
    latents: pd.DataFrame = Field(..., description="Discovered latent variables (Z)")
    proposals: List[ExperimentProposal] = Field(default_factory=list)
    augmented_data: pd.DataFrame = Field(..., description="Original data + Latents")


class InferenceEngineAsync:
    """The 'Principal Investigator' / Mechanism Engine (Async).

    Orchestrates the Discover-Represent-Simulate-Act loop.

    This engine integrates:
    1. DynamicsEngine: For feedback loop discovery.
    2. LatentMiner: For representation learning (confounders).
    3. ActiveScientist: For experimental design.
    4. CausalEstimator: For effect estimation (ATE/CATE).
    5. RuleInductor: For subgroup optimization (TPP).
    6. VirtualSimulator: For in-silico trials.
    """

    def __init__(
        self,
        client: Optional[httpx.AsyncClient] = None,
        dynamics_engine: Optional[DynamicsEngine] = None,
        latent_miner: Optional[LatentMiner] = None,
        active_scientist: Optional[ActiveScientist] = None,
        virtual_simulator: Optional[VirtualSimulator] = None,
        rule_inductor: Optional[RuleInductor] = None,
    ) -> None:
        """Initializes the InferenceEngineAsync with its component engines.

        Args:
            client: Optional HTTPX AsyncClient for network operations.
            dynamics_engine: Engine for discovering system dynamics and loops.
            latent_miner: Engine for discovering latent confounders.
            active_scientist: Engine for proposing experiments.
            virtual_simulator: Engine for running virtual trials.
            rule_inductor: Engine for inducing optimization rules.
        """
        self._internal_client = client is None
        self._client = client or httpx.AsyncClient()

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
        self._latent_features: List[str] = []

    async def __aenter__(self) -> "InferenceEngineAsync":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._internal_client:
            await self._client.aclose()

    @property
    def _estimator(self) -> CausalEstimator:
        if self.augmented_data is None:
            raise ValueError("Data not available. Run analyze() first.")
        return CausalEstimator(self.augmented_data)

    async def analyze(
        self,
        data: pd.DataFrame,
        time_col: str,
        variable_cols: List[str],
        estimate_effect_for: Optional[tuple[str, str]] = None,
    ) -> InferenceResult:
        """Executes the full causal discovery pipeline (Discover-Represent-Act-Simulate).

        Args:
            data: Input dataframe containing time-series data.
            time_col: Name of the time column.
            variable_cols: List of variable columns to analyze.
            estimate_effect_for: Optional tuple (treatment, outcome) to run estimation for.

        Returns:
            InferenceResult: The consolidated results containing the graph, latents, proposals,
            and augmented data.

        Raises:
            ValueError: If input data is empty or invalid.
        """
        logger.info("Starting Inference Engine Pipeline...")

        # 1. Discover (Dynamics)
        logger.info("Step 1: Discover (Dynamics & Loops)")
        await anyio.to_thread.run_sync(self.dynamics_engine.fit, data, time_col, variable_cols)

        self.graph = await anyio.to_thread.run_sync(self.dynamics_engine.discover_loops)

        logger.info(
            f"Discovered Graph with {len(self.graph.edges)} edges and stability score {self.graph.stability_score}"
        )

        # 2. Represent (Latent Learning)
        logger.info("Step 2: Represent (Latent Mining)")
        self._latent_features = variable_cols
        observation_data = data[self._latent_features]

        await anyio.to_thread.run_sync(self.latent_miner.fit, observation_data)
        self.latents = await anyio.to_thread.run_sync(self.latent_miner.discover_latents, observation_data)

        # Augment Data
        # We merge on index. Ensure indices align.
        self.augmented_data = pd.concat([data, self.latents], axis=1)
        logger.info(f"Discovered {self.latents.shape[1]} latent variables. Data augmented.")

        # 3. Act (Active Experimentation)
        logger.info("Step 3: Act (Active Scientist)")
        # We need to filter only numeric columns that are relevant
        analysis_cols = variable_cols + list(self.latents.columns)
        analysis_data = self.augmented_data[analysis_cols]

        await anyio.to_thread.run_sync(self.active_scientist.fit, analysis_data)
        proposals = await anyio.to_thread.run_sync(self.active_scientist.propose_experiments)
        logger.info(f"Generated {len(proposals)} experiment proposals.")

        # 4. Simulate (Estimation) - Optional Trigger
        if estimate_effect_for:
            treatment, outcome = estimate_effect_for
            logger.info(f"Step 4: Simulate (Estimating effect of {treatment} on {outcome})")
            # We set self.estimator for backward compatibility or exposure, though usually transient
            self.estimator = self._estimator

            # Use all other variables as potential confounders (excluding time)
            confounders = [c for c in analysis_cols if c not in [treatment, outcome]]

            # Check if valid
            if treatment not in self.augmented_data.columns or outcome not in self.augmented_data.columns:
                logger.warning(f"Skipping estimation: {treatment} or {outcome} not in data.")
            else:
                try:

                    def _run_est() -> InterventionResult:
                        if self.estimator is None:
                            raise ValueError("Estimator not initialized")
                        return self.estimator.estimate_effect(treatment, outcome, confounders)

                    # estimate_effect can be slow (refutation, model training)
                    result = await anyio.to_thread.run_sync(_run_est)
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

    async def explain_latents(self, background_samples: int = 100) -> pd.DataFrame:
        """Returns the interpretation of the latent variables (SHAP values).

        Calculates global feature importance (Mean Absolute SHAP) for each latent dimension.

        Args:
            background_samples: Number of samples to use for the background dataset (SHAP optimization).

        Returns:
            pd.DataFrame: A matrix where rows are Latent Variables (Z) and columns are Input Features.

        Raises:
            ValueError: If the pipeline hasn't been run or data is missing.
        """
        if self.latent_miner.model is None:
            raise ValueError("Pipeline not run or latent miner not fitted.")

        if self.augmented_data is None or not self._latent_features:
            raise ValueError("Data not available. Run analyze() first.")

        logger.info(f"Explaining Latents using SHAP (background samples: {background_samples})...")

        # Extract the original input features used for training the VAE
        # self.augmented_data contains both original columns and latents.
        # We must only pass the original input features to the explainer.
        try:
            explanation_data = self.augmented_data[self._latent_features]
        except KeyError as e:
            # Fallback if columns were dropped or renamed (unlikely in current flow)
            raise ValueError(f"Could not retrieve original features for explanation: {e}") from e

        # SHAP calculation is heavy
        return await anyio.to_thread.run_sync(self.latent_miner.interpret_latents, explanation_data, background_samples)

    async def estimate_effect(self, treatment: str, outcome: str, confounders: List[str]) -> InterventionResult:
        """Direct access to the CausalEstimator (Simulate).

        Args:
            treatment: Treatment variable name.
            outcome: Outcome variable name.
            confounders: List of confounders.

        Returns:
            InterventionResult: The result of the intervention estimation.

        Raises:
            ValueError: If data is not available.
        """
        return cast(
            InterventionResult,
            await anyio.to_thread.run_sync(self._estimator.estimate_effect, treatment, outcome, confounders),
        )

    async def analyze_heterogeneity(self, treatment: str, outcome: str, confounders: List[str]) -> InterventionResult:
        """Estimates Heterogeneous Treatment Effects (CATE) using Causal Forests.

        Stores the estimates for subsequent rule induction.

        Args:
            treatment: Treatment variable name.
            outcome: Outcome variable name.
            confounders: List of confounders.

        Returns:
            InterventionResult: Result containing ATE and CATE estimates.

        Raises:
            ValueError: If data is not available.
        """
        logger.info(f"Analyzing Heterogeneity for {treatment} -> {outcome}")

        if self.augmented_data is None:
            raise ValueError("Data not available. Run analyze() first.")

        # Run estimation with 'forest' method
        def _estimate() -> InterventionResult:
            return self._estimator.estimate_effect(
                treatment=treatment, outcome=outcome, confounders=confounders, method="forest"
            )

        result = cast(InterventionResult, await anyio.to_thread.run_sync(_estimate))

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

    async def induce_rules(self, feature_cols: Optional[List[str]] = None) -> OptimizationOutput:
        """Induces rules to identify Super-Responders based on stored CATE estimates.

        Args:
            feature_cols: Optional list of columns to use as features for rule induction.
                          If None, uses all numeric columns from augmented_data (excluding metadata).

        Returns:
            OptimizationOutput: The optimization rules and projected uplift.

        Raises:
            ValueError: If CATE estimates are missing or data is unavailable.
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

        await anyio.to_thread.run_sync(self.rule_inductor.fit, features, self.cate_estimates)
        return cast(
            OptimizationOutput,
            await anyio.to_thread.run_sync(self.rule_inductor.induce_rules_with_data, features, self.cate_estimates),
        )

    async def run_virtual_trial(
        self,
        optimization_result: OptimizationOutput,
        treatment: str,
        outcome: str,
        confounders: List[str],
        n_samples: int = 1000,
        adverse_outcomes: List[str] | None = None,
    ) -> VirtualTrialResult:
        """Runs a virtual trial: Generates synthetic cohort, scans safety, and simulates effect.

        Args:
            optimization_result: Output from RuleInductor containing new_criteria.
            treatment: Treatment variable name.
            outcome: Outcome variable name.
            confounders: List of confounder names.
            n_samples: Number of digital twins to generate.
            adverse_outcomes: List of adverse outcome names for safety scanning.

        Returns:
            VirtualTrialResult: Result containing cohort size, safety flags, and effect estimate.

        Raises:
            ValueError: If model is not fitted.
        """
        if self.latent_miner.model is None or self.graph is None:
            raise ValueError("Model not fitted. Run analyze() first.")

        logger.info("Running Virtual Phase 3 Trial...")

        # 1. Generate Synthetic Cohort
        def _generate() -> pd.DataFrame:
            return self.virtual_simulator.generate_synthetic_cohort(
                miner=self.latent_miner,
                n_samples=n_samples,
                rules=optimization_result.new_criteria,
            )

        cohort = await anyio.to_thread.run_sync(_generate)

        if cohort.empty:
            logger.warning("Virtual trial aborted: Cohort is empty after filtering.")
            return VirtualTrialResult(cohort_size=0, safety_scan=[], simulation_result=None)

        # 2. Safety Scan
        safety_flags = []
        if adverse_outcomes:

            def _scan() -> List[str]:
                # We know self.graph is not None from check above
                assert self.graph is not None
                return self.virtual_simulator.scan_safety(
                    graph=self.graph, treatment=treatment, adverse_outcomes=adverse_outcomes
                )

            safety_flags = await anyio.to_thread.run_sync(_scan)

        # 3. Simulate Effect
        sim_result = None
        try:

            def _simulate() -> InterventionResult:
                return self.virtual_simulator.simulate_trial(
                    cohort=cohort,
                    treatment=treatment,
                    outcome=outcome,
                    confounders=confounders,
                )

            sim_result = await anyio.to_thread.run_sync(_simulate)
        except Exception as e:
            logger.error(f"Virtual trial simulation failed: {e}")
            sim_result = None

        return VirtualTrialResult(cohort_size=len(cohort), safety_scan=safety_flags, simulation_result=sim_result)


class InferenceEngine:
    """The 'Principal Investigator' / Mechanism Engine (Sync Facade).

    Wraps InferenceEngineAsync to provide a synchronous interface.
    """

    def __init__(
        self,
        dynamics_engine: Optional[DynamicsEngine] = None,
        latent_miner: Optional[LatentMiner] = None,
        active_scientist: Optional[ActiveScientist] = None,
        virtual_simulator: Optional[VirtualSimulator] = None,
        rule_inductor: Optional[RuleInductor] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initializes the InferenceEngine facade."""
        self._async = InferenceEngineAsync(
            client=client,
            dynamics_engine=dynamics_engine,
            latent_miner=latent_miner,
            active_scientist=active_scientist,
            virtual_simulator=virtual_simulator,
            rule_inductor=rule_inductor,
        )

    def __enter__(self) -> "InferenceEngine":
        return self

    def __exit__(self, *args: Any) -> None:
        anyio.run(self._async.__aexit__, *args)

    @property
    def dynamics_engine(self) -> DynamicsEngine:
        return self._async.dynamics_engine

    @dynamics_engine.setter
    def dynamics_engine(self, value: DynamicsEngine) -> None:
        self._async.dynamics_engine = value

    @property
    def latent_miner(self) -> LatentMiner:
        return self._async.latent_miner

    @latent_miner.setter
    def latent_miner(self, value: LatentMiner) -> None:
        self._async.latent_miner = value

    @property
    def active_scientist(self) -> ActiveScientist:
        return self._async.active_scientist

    @active_scientist.setter
    def active_scientist(self, value: ActiveScientist) -> None:
        self._async.active_scientist = value

    @property
    def virtual_simulator(self) -> VirtualSimulator:
        return self._async.virtual_simulator

    @virtual_simulator.setter
    def virtual_simulator(self, value: VirtualSimulator) -> None:
        self._async.virtual_simulator = value

    @property
    def rule_inductor(self) -> RuleInductor:
        return self._async.rule_inductor

    @rule_inductor.setter
    def rule_inductor(self, value: RuleInductor) -> None:
        self._async.rule_inductor = value

    @property
    def graph(self) -> Optional[CausalGraph]:
        return self._async.graph

    @graph.setter
    def graph(self, value: Optional[CausalGraph]) -> None:
        self._async.graph = value

    @property
    def latents(self) -> Optional[pd.DataFrame]:
        return self._async.latents

    @latents.setter
    def latents(self, value: Optional[pd.DataFrame]) -> None:
        self._async.latents = value

    @property
    def augmented_data(self) -> Optional[pd.DataFrame]:
        return self._async.augmented_data

    @augmented_data.setter
    def augmented_data(self, value: Optional[pd.DataFrame]) -> None:
        self._async.augmented_data = value

    @property
    def cate_estimates(self) -> Optional[pd.Series]:
        return self._async.cate_estimates

    @cate_estimates.setter
    def cate_estimates(self, value: Optional[pd.Series]) -> None:
        self._async.cate_estimates = value

    @property
    def _last_analysis_meta(self) -> Dict[str, str]:
        return self._async._last_analysis_meta

    @_last_analysis_meta.setter
    def _last_analysis_meta(self, value: Dict[str, str]) -> None:
        self._async._last_analysis_meta = value

    @property
    def _latent_features(self) -> List[str]:
        return self._async._latent_features

    @_latent_features.setter
    def _latent_features(self, value: List[str]) -> None:
        self._async._latent_features = value

    def analyze(
        self,
        data: pd.DataFrame,
        time_col: str,
        variable_cols: List[str],
        estimate_effect_for: Optional[tuple[str, str]] = None,
    ) -> InferenceResult:
        return cast(InferenceResult, anyio.run(self._async.analyze, data, time_col, variable_cols, estimate_effect_for))

    def explain_latents(self, background_samples: int = 100) -> pd.DataFrame:
        return cast(pd.DataFrame, anyio.run(self._async.explain_latents, background_samples))

    def estimate_effect(self, treatment: str, outcome: str, confounders: List[str]) -> InterventionResult:
        return cast(InterventionResult, anyio.run(self._async.estimate_effect, treatment, outcome, confounders))

    def analyze_heterogeneity(self, treatment: str, outcome: str, confounders: List[str]) -> InterventionResult:
        return cast(InterventionResult, anyio.run(self._async.analyze_heterogeneity, treatment, outcome, confounders))

    def induce_rules(self, feature_cols: Optional[List[str]] = None) -> OptimizationOutput:
        return cast(OptimizationOutput, anyio.run(self._async.induce_rules, feature_cols))

    def run_virtual_trial(
        self,
        optimization_result: OptimizationOutput,
        treatment: str,
        outcome: str,
        confounders: List[str],
        n_samples: int = 1000,
        adverse_outcomes: List[str] | None = None,
    ) -> VirtualTrialResult:
        return cast(
            VirtualTrialResult,
            anyio.run(
                self._async.run_virtual_trial,
                optimization_result,
                treatment,
                outcome,
                confounders,
                n_samples,
                adverse_outcomes,
            ),
        )
