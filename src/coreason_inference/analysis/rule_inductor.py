# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_inference

from typing import Any, List, Optional, Tuple

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

    def fit(self, features: pd.DataFrame, cate_scores: pd.Series | np.ndarray[Any, Any]) -> None:
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
        self.feature_names = list(features.columns)

        self.tree_model = DecisionTreeRegressor(
            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, random_state=42
        )
        self.tree_model.fit(features, cate_scores)
        logger.info("RuleInductor tree fitted.")

    def _extract_rules(self, leaf_idx: int) -> List[ProtocolRule]:
        """
        Helper method to extract rules (path) from root to a specific leaf.

        Args:
            leaf_idx: The index of the leaf node in the tree.

        Returns:
            List[ProtocolRule]: List of rules defining the path to the leaf.
        """
        if self.tree_model is None:  # pragma: no cover
            raise ValueError("Model not fitted.")

        tree = self.tree_model.tree_
        children_left = tree.children_left
        children_right = tree.children_right
        feature = tree.feature
        threshold = tree.threshold

        # Build parent map to traverse upwards
        # Efficiently we could do this once per tree, but trees are small (shallow)
        # so rebuilding per call is negligible and simpler.
        parents: dict[int, Optional[int]] = {}
        node_stack: List[Tuple[int, Optional[int]]] = [(0, None)]  # (node_idx, parent_idx)

        while node_stack:
            node_idx, parent_idx = node_stack.pop()
            parents[node_idx] = parent_idx
            if children_left[node_idx] != _tree.TREE_LEAF:
                node_stack.append((children_left[node_idx], node_idx))
                node_stack.append((children_right[node_idx], node_idx))

        rules = []
        curr = leaf_idx
        # Traverse up from leaf to root
        while curr != 0:
            parent = parents[curr]
            assert parent is not None

            # Determine if we went left or right
            if children_left[parent] == curr:
                op = "<="
            else:
                op = ">"

            feat_idx = feature[parent]
            feat_name = self.feature_names[feat_idx]
            feat_thresh = threshold[parent]

            rules.append(
                ProtocolRule(feature=feat_name, operator=op, value=float(feat_thresh), rationale="Optimizes Response")
            )
            curr = parent

        # Rules are collected from Leaf -> Root, so reverse them to be Root -> Leaf
        rules.reverse()
        return rules

    def induce_rules(self, cate_scores: pd.Series | np.ndarray[Any, Any]) -> OptimizationOutput:
        """
        Extracts optimized inclusion criteria from the fitted tree using internal tree statistics (Mean CATE).
        """
        if self.tree_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        y_true = np.array(cate_scores)
        baseline_pos = np.mean(y_true > 0)

        tree = self.tree_model.tree_
        n_nodes = tree.node_count
        children_left = tree.children_left
        value = tree.value  # shape (n_nodes, 1, n_outputs=1) -> Mean CATE

        # Identify leaf nodes
        leaf_indices = [i for i in range(n_nodes) if children_left[i] == _tree.TREE_LEAF]

        # Find leaf with max value (Mean CATE)
        best_leaf_idx = -1
        max_cate_mean = -np.inf

        for idx in leaf_indices:
            mean_cate = value[idx][0][0]
            if mean_cate > max_cate_mean:
                max_cate_mean = mean_cate
                best_leaf_idx = idx

        if best_leaf_idx == -1:  # pragma: no cover
            return OptimizationOutput(
                new_criteria=[],
                original_pos=float(baseline_pos),
                optimized_pos=float(baseline_pos),
                safety_flags=["No valid subgroup found."],
            )

        rules = self._extract_rules(best_leaf_idx)

        # Add context to rationale
        for r in rules:
            r.rationale = f"Directs towards high CATE (Leaf Mean: {max_cate_mean:.2f})"

        return OptimizationOutput(
            new_criteria=rules,
            original_pos=float(baseline_pos),
            optimized_pos=float(baseline_pos),  # Placeholder as we can't calc actual PoS without features
            safety_flags=["Optimization inaccurate without feature data. Use induce_rules_with_data."],
        )

    def induce_rules_with_data(self, features: pd.DataFrame, cate_scores: pd.Series | np.ndarray[Any, Any]) -> OptimizationOutput:
        """
        Extracts optimized inclusion criteria calculating stats based on the provided data.
        """
        if self.tree_model is None:
            raise ValueError("Model not fitted.")

        y_true = np.array(cate_scores)
        baseline_pos = np.mean(y_true > 0)

        # Get leaf indices for all samples
        leaf_ids = self.tree_model.apply(features)

        # Find leaf with highest PoS
        unique_leaves = np.unique(leaf_ids)
        best_leaf = -1
        max_pos = -1.0

        for leaf in unique_leaves:
            mask = leaf_ids == leaf
            leaf_y = y_true[mask]
            # PoS = Proportion of responders
            pos = np.mean(leaf_y > 0)
            if pos > max_pos:
                max_pos = pos
                best_leaf = leaf

        if best_leaf == -1:  # pragma: no cover
            return OptimizationOutput(
                new_criteria=[],
                original_pos=float(baseline_pos),
                optimized_pos=float(baseline_pos),
                safety_flags=["No subgroup improves PoS."],
            )

        rules = self._extract_rules(best_leaf)

        # Add context to rationale
        for r in rules:
            r.rationale = "Optimizes Responder Rate"

        return OptimizationOutput(
            new_criteria=rules, original_pos=float(baseline_pos), optimized_pos=float(max_pos), safety_flags=[]
        )
