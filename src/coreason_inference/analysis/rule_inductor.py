# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, _tree

from coreason_inference.schema import OptimizationOutput, ProtocolRule
from coreason_inference.utils.logger import logger


class RuleInductor:
    """
    Translates CATE scores into human-readable Clinical Protocols using Decision Trees.
    Optimizes Phase 3 Probability of Success (PoS) by identifying Super-Responders.
    """

    def __init__(self, max_depth: int = 3, min_samples_leaf: int = 20):
        """
        Args:
            max_depth: Maximum depth of the decision tree (interpretable rules).
            min_samples_leaf: Minimum samples in a leaf to consider it a valid subgroup.
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree_model: Optional[DecisionTreeRegressor] = None
        self.feature_names: List[str] = []

    def fit(self, features: pd.DataFrame, cate_scores: pd.Series | np.ndarray) -> None:
        """
        Fits a Decision Tree Regressor to predict CATE scores from features.

        Args:
            features: DataFrame of patient features (Covariates).
            cate_scores: Estimated Individual Treatment Effects (CATE).
        """
        if features.empty or len(cate_scores) == 0:
            raise ValueError("Input features or CATE scores are empty.")

        if len(features) != len(cate_scores):
            raise ValueError("Features and CATE scores must have the same length.")

        # Ensure features are numeric for Decision Tree
        # Note: We assume encoding is handled upstream or features are numeric.
        # If not, we basic check.
        # In this atomic unit, we assume numeric.
        if not np.issubdtype(features.values.dtype, np.number):
            # Simple check, might fail for object cols that are actually numbers
            # We assume caller provides preprocessed data for now.
            pass

        self.feature_names = list(features.columns)

        self.tree_model = DecisionTreeRegressor(
            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=42
        )
        self.tree_model.fit(features, cate_scores)
        logger.info("RuleInductor tree fitted.")

    def induce_rules(self, cate_scores: pd.Series | np.ndarray) -> OptimizationOutput:
        """
        Extracts optimized inclusion criteria from the fitted tree.

        Args:
            cate_scores: The original CATE scores (needed to calculate baseline PoS).
                         In a real scenario, this might be 'outcomes', but usually we optimize for CATE.
                         Wait, PoS (Probability of Success) usually relates to the Outcome (Y), not CATE.
                         However, the PRD says: "Uses PRIM or pruned Decision Trees on CATE scores."
                         And "Result: Drug works in 85% of patients... Phase 3 PoS increases".

                         If we have CATE, a "Success" might be defined as CATE > threshold (e.g. > 0).
                         Let's assume PoS = Proportion of patients with CATE > 0 (Responders).

        Returns:
            OptimizationOutput: The recommended rules and stats.
        """
        if self.tree_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # 1. Calculate Baseline PoS (Original)
        # Definition of "Success": CATE > 0 (Treatment Benefit)
        # Or if we had the actual binary outcome Y, we would use that.
        # Given we are "inducing rules on CATE scores", let's define PoS as P(CATE > 0).
        # Or simply average CATE if the metric is continuous improvement.
        # PRD example: "original_pos: 0.30".

        # Let's convert CATE to a binary 'Responder' status for PoS calculation if possible,
        # or just use mean CATE if that's the metric.
        # The PRD mentions "PoS increases from 30% to 75%". This implies a probability/proportion.
        # So we define Responder = CATE > 0 (or some epsilon).

        y_true = np.array(cate_scores)
        baseline_pos = np.mean(y_true > 0)

        # 2. Find the best leaf node
        # We want the leaf with the highest average CATE.
        # (Alternatively, the highest density of Responders).
        # DecisionTreeRegressor predicts the Mean of the target (CATE) in the leaf.

        tree = self.tree_model.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value  # shape (n_nodes, 1, n_outputs=1) -> Mean CATE

        # Identify leaf nodes
        leaf_indices = [i for i in range(n_nodes) if children_left[i] == _tree.TREE_LEAF]

        # Find leaf with max value (Mean CATE)
        # Filter for leaves with "enough" samples is handled by min_samples_leaf during fit,
        # but we might want to enforce "meaningful uplift".

        best_leaf_idx = -1
        max_cate_mean = -np.inf

        for idx in leaf_indices:
            # value[idx][0][0] is the mean CATE
            mean_cate = value[idx][0][0]
            if mean_cate > max_cate_mean:
                max_cate_mean = mean_cate
                best_leaf_idx = idx

        if best_leaf_idx == -1:
            # Should not happen unless tree is empty
            return OptimizationOutput(
                new_criteria=[],
                original_pos=float(baseline_pos),
                optimized_pos=float(baseline_pos),
                safety_flags=["No valid subgroup found."],
            )

        # 3. Extract Rules for the best leaf
        # We need to traverse from root to this leaf.
        # We can do this by traversing up if we had parent pointers, or searching down.
        # Since we have the index, we can search down or pre-compute parents.
        # Let's compute parents map.

        parents: dict[int, Optional[int]] = {}
        node_stack: List[Tuple[int, Optional[int]]] = [(0, None)]  # (node_idx, parent_idx)
        while node_stack:
            node_idx, parent_idx = node_stack.pop()
            parents[node_idx] = parent_idx
            if children_left[node_idx] != _tree.TREE_LEAF:
                node_stack.append((children_left[node_idx], node_idx))
                node_stack.append((children_right[node_idx], node_idx))

        # Trace back from best_leaf_idx to root
        rules = []
        curr = best_leaf_idx
        while curr != 0:
            parent = parents[curr]
            # Determine if we went left or right
            if children_left[parent] == curr:
                # Left child: feature <= threshold
                op = "<="
            else:
                # Right child: feature > threshold
                op = ">"

            # Mypy might complain about parent being Optional[int] used as int, but loop condition ensures != 0
            # which implies parent is not None.
            assert parent is not None

            feat_idx = feature[parent]
            feat_name = self.feature_names[feat_idx]
            feat_thresh = threshold[parent]

            rules.append(
                ProtocolRule(
                    feature=feat_name,
                    operator=op,
                    value=float(feat_thresh),
                    rationale=f"Directs towards high CATE (Leaf Mean: {max_cate_mean:.2f})",
                )
            )

            curr = parent

        # Rules are in reverse order (Leaf -> Root), reverse them
        rules.reverse()

        # Note: In this method, we can't calculate optimized_pos accurately without features.
        # So we return 0.0 or raise warning.
        # For this implementation, we will assume this method is less preferred than induce_rules_with_data.
        return OptimizationOutput(
            new_criteria=rules,
            original_pos=float(baseline_pos),
            optimized_pos=float(baseline_pos),  # Placeholder
            safety_flags=["Optimization inaccurate without feature data. Use induce_rules_with_data."],
        )

    def induce_rules_with_data(self, features: pd.DataFrame, cate_scores: pd.Series | np.ndarray) -> OptimizationOutput:
        """
        Extracts optimized inclusion criteria.
        Calculates stats based on the provided data.
        """
        if self.tree_model is None:
            # Auto-fit if not fitted? No, let's raise error.
            # Or if fit was called, maybe we can use that?
            # Let's assume user calls fit then induce, or we just combine them.
            # But separating allows fitting on Train and inducing on Validation.
            raise ValueError("Model not fitted.")

        y_true = np.array(cate_scores)
        baseline_pos = np.mean(y_true > 0)

        # Get leaf indices for all samples
        leaf_ids = self.tree_model.apply(features)

        # Find leaf with highest PoS (not just mean CATE, though likely correlated)
        unique_leaves = np.unique(leaf_ids)

        best_leaf = -1
        max_pos = -1.0

        # We only consider leaves with sufficient samples
        # (already enforced by min_samples_leaf in fit, but good to check)

        for leaf in unique_leaves:
            mask = leaf_ids == leaf
            leaf_y = y_true[mask]

            # PoS = Proportion of responders
            pos = np.mean(leaf_y > 0)

            if pos > max_pos:
                max_pos = pos
                best_leaf = leaf

        if best_leaf == -1:
            return OptimizationOutput(
                new_criteria=[],
                original_pos=float(baseline_pos),
                optimized_pos=float(baseline_pos),
                safety_flags=["No subgroup improves PoS."],
            )

        # Extract rules for best_leaf
        # (Logic duplicated from above, let's extract helper if needed, or just run it here)
        tree = self.tree_model.tree_
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold

        parents: dict[int, Optional[int]] = {}
        node_stack: List[Tuple[int, Optional[int]]] = [(0, None)]
        while node_stack:
            node_idx, parent_idx = node_stack.pop()
            parents[node_idx] = parent_idx
            if children_left[node_idx] != _tree.TREE_LEAF:
                node_stack.append((children_left[node_idx], node_idx))
                node_stack.append((children_right[node_idx], node_idx))

        rules = []
        curr = best_leaf
        while curr != 0:
            parent = parents[curr]
            op = "<=" if children_left[parent] == curr else ">"

            assert parent is not None

            feat_idx = feature[parent]
            feat_name = self.feature_names[feat_idx]
            feat_thresh = threshold[parent]

            rules.append(
                ProtocolRule(
                    feature=feat_name, operator=op, value=float(feat_thresh), rationale="Optimizes Responder Rate"
                )
            )
            curr = parent

        rules.reverse()

        return OptimizationOutput(
            new_criteria=rules, original_pos=float(baseline_pos), optimized_pos=float(max_pos), safety_flags=[]
        )
