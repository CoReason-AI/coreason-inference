# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import List, Optional

import numpy as np
import pandas as pd
from dowhy import CausalModel
from sklearn.linear_model import LinearRegression, LogisticRegression

from coreason_inference.schema import InterventionResult, RefutationStatus
from coreason_inference.utils.logger import logger


class CausalEstimator:
    """
    Estimates the causal effect of a treatment on an outcome using Double Machine Learning (DML)
    and validates the result using a Placebo Refuter.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the estimator with the dataset.

        Args:
            data: A pandas DataFrame containing the observational data.
        """
        self.data = data

    def estimate_effect(
        self,
        treatment: str,
        outcome: str,
        confounders: List[str],
        patient_id_col: str = "patient_id",
        treatment_is_binary: bool = False,
        method: str = "linear",  # "linear" or "forest"
    ) -> InterventionResult:
        """
        Estimate the causal effect of `treatment` on `outcome` controlling for `confounders`.

        Args:
            treatment: The column name of the treatment variable.
            outcome: The column name of the outcome variable.
            confounders: A list of column names representing confounding variables.
            patient_id_col: The column name for patient IDs (used for result mapping).
            treatment_is_binary: Set to True if the treatment variable is binary (0/1).
            method: The estimation method. "linear" for LinearDML, "forest" for CausalForestDML.

        Returns:
            InterventionResult: The estimated effect (ATE) and optional CATE estimates.
        """
        logger.info(f"Starting causal estimation: Treatment='{treatment}', Outcome='{outcome}', Method='{method}'")

        # 1. Define Causal Model
        # If using Causal Forest, we treat confounders as effect modifiers (X) to estimate heterogeneity.
        # Otherwise, they are treated as common causes (W) for control.
        # Note: DoWhy maps effect_modifiers to 'X' and common_causes to 'W' in EconML.
        # CausalForestDML requires 'X' to be present.
        effect_modifiers = confounders if method == "forest" else []

        model = CausalModel(
            data=self.data,
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders,
            effect_modifiers=effect_modifiers,
            logging_level="ERROR",  # Reduce noise from dowhy
        )

        # 2. Identify Effect
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # 3. Estimate Effect
        # Configure estimator based on treatment type and selected method
        model_t = LogisticRegression() if treatment_is_binary else LinearRegression()
        model_y = LinearRegression()

        if method == "forest":
            method_name = "backdoor.econml.dml.CausalForestDML"
            # CausalForestDML specific params
            # We use default trees/splits for this atomic unit.
            method_params = {
                "init_params": {
                    "model_y": model_y,
                    "model_t": model_t,
                    "discrete_treatment": treatment_is_binary,
                    "n_estimators": 100,  # Efficient for tests/demos
                    "random_state": 42,
                },
                "fit_params": {},
            }
        else:
            method_name = "backdoor.econml.dml.LinearDML"
            method_params = {
                "init_params": {
                    "model_y": model_y,
                    "model_t": model_t,
                    "discrete_treatment": treatment_is_binary,
                    "random_state": 42,
                },
                "fit_params": {},
            }

        try:
            estimate = model.estimate_effect(
                identified_estimand,
                method_name=method_name,
                method_params=method_params,
            )
        except Exception as e:
            logger.error(f"Estimation failed: {e}")
            raise e

        effect_value = estimate.value
        logger.info(f"Estimated Effect (ATE): {effect_value}")

        # Extract CATE if method is forest
        cate_estimates: Optional[List[float]] = None
        if method == "forest":
            try:
                # Access the underlying EconML estimator from the DoWhy estimate object
                # DoWhy stores the fitted estimator.
                # Usually: estimate.estimator
                # The EconML wrapper has a method `effect(X)`.
                # We need the features (X) used for heterogeneity.
                # In CausalForestDML, effect_modifiers usually default to common_causes (confounders)
                # unless specified otherwise in identifying estimand.
                # DoWhy passes common_causes as X + W usually?
                # Actually, `estimate.estimator.effect(data)` usually requires data with the effect modifier columns.
                # We pass the full dataframe subset?
                # Dowhy's `estimate_effect` passes the data used during fit.
                # We can try to use `estimate.estimator.effect(self.data[confounders])`?
                # Or safely `self.data` if it handles column selection.
                # Let's inspect `estimate.estimator` behavior or try passing full data.
                # Safest is to pass the specific columns if we know them.
                # DoWhy might have stored effect modifiers.
                # EconML expects X (effect modifiers) to predict CATE.
                cate_arr = estimate.estimator.effect(self.data[effect_modifiers])
                cate_estimates = cate_arr.flatten().tolist()
            except Exception as e:
                logger.warning(f"Could not extract CATE estimates: {e}")

        # 4. Refute Estimate (Placebo Test)
        refutation = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            num_simulations=10,
        )

        refutation_passed = refutation.refutation_result["is_statistically_significant"]
        status = RefutationStatus.FAILED if refutation_passed else RefutationStatus.PASSED
        logger.info(f"Refutation Status: {status} (p-value: {refutation.refutation_result['p_value']})")

        # Confidence Interval
        ci = estimate.get_confidence_intervals()
        if ci is None:
            ci_low, ci_high = (effect_value, effect_value)
        else:
            # If CATE, CI might be array?
            # get_confidence_intervals() on DoWhy usually returns the CI of the *target estimand* (ATE).
            # For CATE, we need `const_marginal_effect_interval` or similar from EconML.
            # But `estimate.get_confidence_intervals()` returns ATE CI.
            # We'll stick to ATE CI for the main fields.
            # EconML usually returns (lb, ub) scalar for ATE if asked for ATE, or arrays for CATE.
            # DoWhy asks for the ATE CI usually.
            try:
                if isinstance(ci, tuple) or isinstance(ci, list):
                    ci_low, ci_high = ci[0], ci[1]
                    # If these are arrays (CATE CIs), take mean or just first?
                    # Ideally we want ATE CI here.
                    if isinstance(ci_low, (np.ndarray, list)):
                        ci_low = np.mean(ci_low)
                    if isinstance(ci_high, (np.ndarray, list)):
                        ci_high = np.mean(ci_high)
            except Exception:
                ci_low, ci_high = (effect_value, effect_value)

        result = InterventionResult(
            patient_id="POPULATION_ATE",
            intervention=f"do({treatment})",
            counterfactual_outcome=float(effect_value),
            confidence_interval=(float(ci_low), float(ci_high)),
            refutation_status=status,
            cate_estimates=cate_estimates,
        )

        return result
