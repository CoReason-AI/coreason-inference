# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import List

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
    ) -> InterventionResult:
        """
        Estimate the causal effect of `treatment` on `outcome` controlling for `confounders`.

        Args:
            treatment: The column name of the treatment variable.
            outcome: The column name of the outcome variable.
            confounders: A list of column names representing confounding variables.
            patient_id_col: The column name for patient IDs (used for result mapping).
            treatment_is_binary: Set to True if the treatment variable is binary (0/1).

        Returns:
            InterventionResult: The estimated effect and refutation status.
        """
        logger.info(f"Starting causal estimation: Treatment='{treatment}', Outcome='{outcome}'")

        # 1. Define Causal Model
        model = CausalModel(
            data=self.data,
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders,
            logging_level="ERROR",  # Reduce noise from dowhy
        )

        # 2. Identify Effect
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # 3. Estimate Effect using Double Machine Learning (LinearDML) via EconML
        # Note: We use a linear model for the DML components for simplicity and robustness in this atomic unit.
        # EconML requires passing the estimator classes, not instances, for some args, or valid strings.
        # "backdoor.econml.dml.LinearDML" is the method name in dowhy.

        # Configure estimator based on treatment type
        # For discrete treatment, use LogisticRegression for model_t
        model_t = LogisticRegression() if treatment_is_binary else LinearRegression()

        try:
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.econml.dml.LinearDML",
                method_params={
                    "init_params": {
                        "model_y": LinearRegression(),
                        "model_t": model_t,
                        "discrete_treatment": treatment_is_binary,
                        "random_state": 42,
                    },
                    "fit_params": {},
                },
            )
        except Exception as e:
            logger.error(f"Estimation failed: {e}")
            raise e

        effect_value = estimate.value

        logger.info(f"Estimated Effect: {effect_value}")

        # 4. Refute Estimate (Placebo Test)
        refutation = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            num_simulations=10,  # Keep low for performance in atomic unit, increase for prod
        )

        refutation_passed = refutation.refutation_result["is_statistically_significant"]
        # In placebo test, we want the effect to be NOT significant (i.e., close to zero).
        # Dowhy's `is_statistically_significant` usually returns True if p-value < alpha (default 0.05).
        # So if it IS significant, the placebo test FAILED (we found an effect where there should be none).
        # If it is NOT significant, the placebo test PASSED.

        status = RefutationStatus.FAILED if refutation_passed else RefutationStatus.PASSED
        logger.info(f"Refutation Status: {status} (p-value: {refutation.refutation_result['p_value']})")

        # Construct InterventionResult
        # Note: InterventionResult expects a single patient_id, but this is a population estimate (ATE).
        # The PRD says "InterventionResult ... patient_id: str".
        # This implies we might need CATE (Conditional ATE) per patient, or we are just returning the ATE
        # and using a placeholder or the first patient?
        # The PRD "User Stories" mentions "What would Patient 001's tumor size be...".
        # However, for this atomic unit "De-Confounder", we are calculating "The true impact of a treatment".
        # Usually DML gives ATE or CATE.
        # If we use LinearDML, we can get CATE.
        # Let's see if we can get the effect for a specific target or just average.
        # For this implementation, to adhere to the return type `InterventionResult`,
        # I will return the Average Treatment Effect (ATE) and mark it as "POPULATION_ATE".

        # Also confidence interval. Dowhy estimate might provide it.
        # EconML estimates provide CIs. Dowhy wraps it.
        # `estimate.get_confidence_intervals()`
        ci = estimate.get_confidence_intervals()
        if ci is None:
            # Fallback if not computed
            ci_low, ci_high = (effect_value, effect_value)
        else:
            ci_low, ci_high = ci[0], ci[1]

        result = InterventionResult(
            patient_id="POPULATION_ATE",  # Returning ATE for now
            intervention=f"do({treatment})",
            # This field name suggests Y_cf, but here we have the *Effect* (dy/dx).
            # I will store the Effect Size in `counterfactual_outcome` for this unit.
            counterfactual_outcome=effect_value,
            confidence_interval=(float(ci_low), float(ci_high)),
            refutation_status=status,
        )

        return result
