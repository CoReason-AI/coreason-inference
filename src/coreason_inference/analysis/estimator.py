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

        # 1. Define Causal Model
        effect_modifiers = confounders if method == METHOD_FOREST else []

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

        # Extract CATE if method is forest
        cate_estimates = self._extract_cate_estimates(estimate, effect_modifiers) if method == METHOD_FOREST else None

        # Determine Primary Outcome Value (ATE or Personalized CATE)
        if target_patient_id and cate_estimates:
            # Personalized Inference
            if patient_id_col not in self.data.columns:
                raise ValueError(f"Patient ID column '{patient_id_col}' not found in data.")

            # Find patient index
            patient_indices = self.data.index[self.data[patient_id_col] == target_patient_id].tolist()
            if not patient_indices:
                raise ValueError(f"Patient ID '{target_patient_id}' not found in data.")

            # Use the first occurrence (assuming unique ID per row or taking first match)
            # cate_estimates is a list matching data order.
            # We need the integer position (iloc) corresponding to the patient.
            # self.data might have arbitrary index.
            # cate_estimates is generated from self.data[effect_modifiers] in original order.
            # So we need integer location.

            # Get integer location of the patient
            # pd.Index.get_loc returns integer or slice or boolean mask.
            # Safer to find integer index manually or assume unique.
            # Let's use boolean mask on the column, then np.where
            mask = (self.data[patient_id_col] == target_patient_id).values
            locs = np.where(mask)[0]
            if len(locs) == 0:
                raise ValueError(f"Patient ID '{target_patient_id}' not found.")

            patient_idx = locs[0]
            effect_value = float(cate_estimates[patient_idx])
            result_patient_id = target_patient_id
            logger.info(f"Personalized Effect for {target_patient_id}: {effect_value}")

        else:
            # Population ATE
            effect_value = float(estimate.value)
            result_patient_id = "POPULATION_ATE"
            logger.info(f"Estimated Effect (ATE): {effect_value}")

        # 4. Refute Estimate (Placebo Test)
        refutation = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            num_simulations=num_simulations,
        )

        refutation_passed = refutation.refutation_result["is_statistically_significant"]
        status = RefutationStatus.FAILED if refutation_passed else RefutationStatus.PASSED
        logger.info(f"Refutation Status: {status} (p-value: {refutation.refutation_result['p_value']})")

        # Confidence Interval
        ci_low, ci_high = self._extract_confidence_intervals(estimate, effect_value)

        result = InterventionResult(
            patient_id=result_patient_id,
            intervention=f"do({treatment})",
            counterfactual_outcome=effect_value,
            confidence_interval=(ci_low, ci_high),
            refutation_status=status,
            cate_estimates=cate_estimates,
        )

        return result

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
        ci = estimate.get_confidence_intervals()
        if ci is None:
            return default_value, default_value

        try:
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
