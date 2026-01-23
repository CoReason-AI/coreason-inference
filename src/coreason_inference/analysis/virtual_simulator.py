# Copyright (C) 2026 CoReason
# Licensed under the Prosperity Public License 3.0.0

import operator
from typing import Any, Callable, Dict, List

import networkx as nx
import pandas as pd
from coreason_inference.analysis.estimator import CausalEstimator
from coreason_inference.analysis.latent import LatentMiner
from coreason_inference.schema import CausalGraph, InterventionResult, ProtocolRule
from coreason_inference.utils.logger import logger

# Map string operators to functions
# e.g. ">" -> operator.gt(a, b)
# We use Any because pandas Series comparison returns Series[bool], not bool
OPERATOR_MAP: Dict[str, Callable[[Any, Any], Any]] = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


class VirtualSimulator:
    """
    "Re-runs" Phase 2 trials in silico with new inclusion/exclusion criteria.
    Generates synthetic cohorts ('Digital Twins') and simulates outcomes.
    """

    def __init__(self) -> None:
        pass

    def generate_synthetic_cohort(
        self,
        miner: LatentMiner,
        n_samples: int = 1000,
        rules: List[ProtocolRule] | None = None,
        max_retries: int = 10,
    ) -> pd.DataFrame:
        """
        Generates a synthetic cohort with Adaptive Sampling to ensure retention.

        Args:
            miner: Fitted LatentMiner instance.
            n_samples: Number of initial candidates to generate (pool size).
            rules: List of inclusion criteria to filter the cohort.
            max_retries: Maximum number of attempts to generate enough samples.

        Returns:
            pd.DataFrame: The filtered synthetic cohort (Digital Twins).
        """
        target_size = n_samples
        collected_cohorts = []
        total_collected = 0

        logger.info(f"Generating synthetic cohort. Target Size: {target_size}")

        for attempt in range(max_retries):
            # Heuristic: Generate 5x the target to account for strict filtering
            batch_size = target_size * 5

            try:
                batch = miner.generate(batch_size)
            except Exception as e:
                logger.error(f"Failed to generate synthetic data: {e}")
                raise e

            if batch.empty:
                continue

            # Filter by Rules
            if rules:
                batch = self._apply_rules(batch, rules)

            if not batch.empty:
                collected_cohorts.append(batch)
                total_collected += len(batch)
                logger.debug(f"Batch {attempt + 1}: Found {len(batch)} matching profiles.")

            if total_collected >= target_size:
                break

        if not collected_cohorts:
            logger.warning("LatentMiner failed to produce valid candidates after retries.")
            # Try to preserve columns from the miner if possible
            cols = getattr(miner, "feature_names", None)
            return pd.DataFrame(columns=cols) if cols else pd.DataFrame()

        # Combine all successful batches
        final_cohort = pd.concat(collected_cohorts, ignore_index=True)

        # Trim to exact requested size
        if len(final_cohort) > target_size:
            final_cohort = final_cohort.iloc[:target_size]

        logger.info(f"Final cohort size: {len(final_cohort)} (Generated after {len(collected_cohorts)} batches)")
        return final_cohort

    def scan_safety(self, graph: CausalGraph, treatment: str, adverse_outcomes: List[str]) -> List[str]:
        """
        Scans the causal graph for pathways leading from the treatment to adverse outcomes.

        Args:
            graph: The discovered CausalGraph.
            treatment: The treatment variable node ID.
            adverse_outcomes: List of adverse outcome node IDs.

        Returns:
            List[str]: A list of warning messages describing the risk pathways found.
        """
        logger.info(f"Starting Safety Scan for treatment '{treatment}' against {len(adverse_outcomes)} outcomes.")

        # Use centralized graph conversion
        G = graph.to_networkx()

        safety_flags = []

        if treatment not in G:
            logger.warning(f"Treatment '{treatment}' not found in graph. Skipping scan.")
            return []

        for adverse in adverse_outcomes:
            if adverse not in G:
                logger.debug(f"Adverse outcome '{adverse}' not in graph. No risk detected.")
                continue

            if nx.has_path(G, treatment, adverse):
                try:
                    path = nx.shortest_path(G, treatment, adverse)
                    path_str = " -> ".join(path)
                    msg = f"Risk path detected: {path_str}"
                    safety_flags.append(msg)
                    logger.warning(msg)
                except Exception as e:
                    logger.error(f"Error calculating path to {adverse}: {e}")

        return safety_flags

    def simulate_trial(
        self,
        cohort: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str],
        method: str = "forest",
    ) -> InterventionResult:
        """
        Simulates the trial outcome on the synthetic cohort using Causal Estimator.

        Args:
            cohort: The synthetic cohort DataFrame.
            treatment: Treatment column name.
            outcome: Outcome column name.
            confounders: List of confounders.
            method: Estimation method ('linear' or 'forest'). Defaults to 'forest' for heterogeneity.

        Returns:
            InterventionResult: The estimated effect.
        """
        if cohort.empty:
            raise ValueError("Cannot simulate trial on empty cohort.")

        if treatment not in cohort.columns:
            raise ValueError(f"Treatment '{treatment}' not found in cohort.")

        if outcome not in cohort.columns:
            raise ValueError(f"Outcome '{outcome}' not found in cohort.")

        logger.info(f"Simulating Virtual Trial on {len(cohort)} digital twins. Method: {method}")

        try:
            estimator = CausalEstimator(cohort)
            result = estimator.estimate_effect(
                treatment=treatment,
                outcome=outcome,
                confounders=confounders,
                method=method,
                num_simulations=5,  # Reduced for simulation speed, or make configurable
            )
            return result
        except Exception as e:
            logger.error(f"Virtual Trial Simulation failed: {e}")
            raise e

    def _apply_rules(self, data: pd.DataFrame, rules: List[ProtocolRule]) -> pd.DataFrame:
        """
        Filters the dataframe based on the list of ProtocolRules.
        """
        filtered_data = data.copy()

        for rule in rules:
            feature = rule.feature
            op = rule.operator
            val = rule.value

            if feature not in filtered_data.columns:
                logger.warning(f"Rule feature '{feature}' not found in generated data. Skipping rule.")
                continue

            op_func = OPERATOR_MAP.get(op)
            if not op_func:
                logger.warning(f"Unsupported operator '{op}' in rule for '{feature}'. Skipping.")
                continue

            try:
                # Apply filter
                # We use boolean indexing directly with the operator function
                mask = op_func(filtered_data[feature], val)
                filtered_data = filtered_data[mask]
            except Exception as e:
                logger.error(f"Error applying rule {rule}: {e}")

            if filtered_data.empty:
                logger.warning(f"Cohort empty after applying rule: {feature} {op} {val}")
                break

        return filtered_data
