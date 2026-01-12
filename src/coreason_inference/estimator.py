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
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression, LogisticRegression

from coreason_inference.utils.logger import logger


class EstimationResult(BaseModel):
    effect: float
    refutation_passed: bool
    refutation_p_value: float


class CausalEstimator:
    """
    Implements Double Machine Learning (DML) via EconML and Dowhy.
    Enforces mandatory Placebo Refutation.
    """

    def __init__(self, treatment_is_binary: bool = False):
        self.treatment_is_binary = treatment_is_binary

    def estimate_effect(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        confounders: List[str],
    ) -> EstimationResult:
        """
        Estimates the causal effect of treatment on outcome controlling for confounders.
        """
        if data.empty:
            raise ValueError("Data is empty.")

        missing_cols = [col for col in [treatment, outcome] + confounders if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        logger.info(f"Initializing CausalModel for treatment='{treatment}', outcome='{outcome}'")

        # 1. Define Causal Model (Graph)
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders,
            logging_level="ERROR",  # Reduce verbosity
        )

        # 2. Identify Estimand
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # 3. Estimate Effect using EconML LinearDML
        # We configure model_y (Outcome) and model_t (Treatment)
        # If treatment is binary, use LogisticRegression for model_t
        model_y = LinearRegression()
        model_t = LogisticRegression() if self.treatment_is_binary else LinearRegression()

        logger.info("Running LinearDML estimation...")
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.econml.dml.LinearDML",
            method_params={
                "init_params": {
                    "model_y": model_y,
                    "model_t": model_t,
                    "discrete_treatment": self.treatment_is_binary,
                    "random_state": 42,
                },
                "fit_params": {},
            },
        )

        effect_value = estimate.value
        logger.info(f"Estimated Effect: {effect_value}")

        # 4. Refute Estimate (Placebo Treatment)
        # "Refutation is Law"
        logger.info("Running Placebo Refutation...")
        refute = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            num_simulations=10,  # Keep low for unit tests, maybe config later
            random_state=42,
        )

        p_value = refute.refutation_result["p_value"]
        # If p_value < 0.05, it means the placebo *did* have an effect (significant),
        # which implies our model is capturing noise/confounders as signal.
        # Thus, high p-value is GOOD for placebo test (we want null hypothesis to hold).
        refutation_passed = p_value >= 0.05

        logger.info(f"Refutation p-value: {p_value}, Passed: {refutation_passed}")

        if not refutation_passed:
            logger.warning("Refutation Failed! The model found a causal link in placebo data.")

        return EstimationResult(effect=effect_value, refutation_passed=refutation_passed, refutation_p_value=p_value)
