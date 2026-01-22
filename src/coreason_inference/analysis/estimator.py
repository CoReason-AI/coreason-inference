# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dowhy import CausalModel
from sklearn.linear_model import LinearRegression, LogisticRegression

from coreason_inference.schema import InterventionResult, RefutationStatus
from coreason_inference.utils.logger import logger

METHOD_LINEAR = "linear"
METHOD_FOREST = "forest"

DML_LINEAR_BACKEND = "backdoor.econml.dml.LinearDML"
DML_FOREST_BACKEND = "backdoor.econml.dml.CausalForestDML"


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
        method: str = METHOD_LINEAR,
        num_simulations: int = 10,
        target_patient_id: Optional[str] = None,
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
            num_simulations: Number of simulations for placebo refutation.
            target_patient_id: Optional ID of a specific patient to retrieve individual CATE for.

        Returns:
            InterventionResult: The estimated effect (ATE or individual CATE) and optional CATE distribution.
        """
        logger.info(f"Starting causal estimation: Treatment='{treatment}', Outcome='{outcome}', Method='{method}'")

        # 1. Define & Fit Model
        model, identified_estimand, estimate = self._fit_and_estimate(
            treatment, outcome, confounders, method, treatment_is_binary
        )

        # 2. Extract Results (ATE & CATE)
        effect_modifiers = confounders if method == METHOD_FOREST else []
        cate_estimates = self._extract_cate_estimates(estimate, effect_modifiers) if method == METHOD_FOREST else None

        effect_value, result_patient_id = self._determine_effect_value(
            estimate, cate_estimates, patient_id_col, target_patient_id
        )

        # Early exit if estimation failed (value is None)
        if effect_value is None:
            return self._build_failure_result(treatment)

        logger.info(f"Estimated Effect ({result_patient_id}): {effect_value}")

        # 3. Refute Estimate
        status = self._refute_estimate(model, identified_estimand, estimate, num_simulations)

        # 4. Finalize Result
        # Invalidate if refutation failed
        final_effect = effect_value if status == RefutationStatus.PASSED else None
        final_cate = cate_estimates if status == RefutationStatus.PASSED else None

        if status == RefutationStatus.FAILED:
            logger.warning(f"Estimate invalidated due to failed refutation for {treatment}->{outcome}")

        ci_low, ci_high = self._extract_confidence_intervals(estimate, effect_value)

        return InterventionResult(
            patient_id=result_patient_id,
            intervention=f"do({treatment})",
            counterfactual_outcome=final_effect,
            confidence_interval=(ci_low, ci_high),
            refutation_status=status,
            cate_estimates=final_cate,
        )

    def _fit_and_estimate(
        self, treatment: str, outcome: str, confounders: List[str], method: str, treatment_is_binary: bool
    ) -> Tuple[CausalModel, Any, Any]:
        """
        Configures the CausalModel, identifies the effect, and runs the estimation.
        """
        effect_modifiers = confounders if method == METHOD_FOREST else []

        model = CausalModel(
            data=self.data,
            treatment=treatment,
            outcome=outcome,
            common_causes=confounders,
            effect_modifiers=effect_modifiers,
            logging_level="ERROR",
        )

        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        method_name, method_params = self._get_method_params(method, treatment_is_binary)

        try:
            estimate = model.estimate_effect(
                identified_estimand,
                method_name=method_name,
                method_params=method_params,
            )
        except Exception as e:
            logger.error(f"Estimation failed: {e}")
            raise e

        return model, identified_estimand, estimate

    def _determine_effect_value(
        self,
        estimate: Any,
        cate_estimates: Optional[List[float]],
        patient_id_col: str,
        target_patient_id: Optional[str],
    ) -> Tuple[Optional[float], str]:
        """
        Determines whether to return the Population ATE or a Personalized CATE.
        Returns (effect_value, result_label).
        """
        if target_patient_id and cate_estimates:
            # Personalized Inference
            if patient_id_col not in self.data.columns:
                raise ValueError(f"Patient ID column '{patient_id_col}' not found in data.")

            mask = (self.data[patient_id_col] == target_patient_id).values
            locs = np.where(mask)[0]

            if len(locs) == 0:
                raise ValueError(f"Patient ID '{target_patient_id}' not found in data.")

            patient_idx = locs[0]
            effect_value = float(cate_estimates[patient_idx])
            return effect_value, target_patient_id

        # Population ATE
        if estimate.value is None:
            logger.error("DoWhy returned None for estimate.value.")
            return None, "ERROR"

        return float(estimate.value), "POPULATION_ATE"

    def _refute_estimate(
        self, model: CausalModel, identified_estimand: Any, estimate: Any, num_simulations: int
    ) -> RefutationStatus:
        """
        Runs the Placebo Treatment Refuter.
        """
        try:
            refutation = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute",
                num_simulations=num_simulations,
            )

            if refutation.refutation_result["is_statistically_significant"]:
                return RefutationStatus.FAILED
            return RefutationStatus.PASSED

        except Exception as e:
            logger.warning(f"Refutation process failed: {e}")
            # If refutation crashes, do we fail the result?
            # Usually strict safety says YES, fail it if we can't verify it.
            # But prompt request was to handle crash.
            # I will return FAILED to be safe.
            return RefutationStatus.FAILED

    def _build_failure_result(self, treatment: str) -> InterventionResult:
        return InterventionResult(
            patient_id="ERROR",
            intervention=f"do({treatment})",
            counterfactual_outcome=None,
            confidence_interval=(0.0, 0.0),
            refutation_status=RefutationStatus.FAILED,
            cate_estimates=None,
        )

    def _get_method_params(self, method: str, treatment_is_binary: bool) -> Tuple[str, Dict[str, Any]]:
        """
        Constructs the method name and parameters for the EconML estimator.
        """
        model_t = LogisticRegression() if treatment_is_binary else LinearRegression()
        model_y = LinearRegression()

        init_params = {
            "model_y": model_y,
            "model_t": model_t,
            "discrete_treatment": treatment_is_binary,
            "random_state": 42,
        }

        if method == METHOD_FOREST:
            method_name = DML_FOREST_BACKEND
            init_params["n_estimators"] = 100
        else:
            method_name = DML_LINEAR_BACKEND

        return method_name, {"init_params": init_params, "fit_params": {}}

    def _extract_cate_estimates(self, estimate: Any, effect_modifiers: List[str]) -> Optional[List[float]]:
        """
        Extracts Conditional Average Treatment Effects (CATE) from the fitted estimator.
        """
        try:
            # EconML expects X (effect modifiers) to predict CATE.
            cate_arr = estimate.estimator.effect(self.data[effect_modifiers])
            return list(cate_arr.flatten().tolist())
        except Exception as e:
            logger.warning(f"Could not extract CATE estimates: {e}")
            return None

    def _extract_confidence_intervals(self, estimate: Any, default_value: float) -> Tuple[float, float]:
        """
        Safely extracts confidence intervals from the estimate.
        """
        try:
            ci = estimate.get_confidence_intervals()
            if ci is None:
                return default_value, default_value

            if isinstance(ci, (tuple, list)):
                ci_low, ci_high = ci[0], ci[1]

                if isinstance(ci_low, (np.ndarray, list)):
                    ci_low = float(np.mean(ci_low))
                if isinstance(ci_high, (np.ndarray, list)):
                    ci_high = float(np.mean(ci_high))

                return float(ci_low), float(ci_high)
        except Exception:
            pass

        return default_value, default_value
