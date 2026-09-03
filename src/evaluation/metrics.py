"""
Evaluation metrics for recommendation systems.

This module provides implementations of standard recommendation metrics
including Recall@K, NDCG@K, MRR@K, and Coverage@K with vectorized
computations for efficiency.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


def recall_at_k(
    predictions: Union[List[List[int]], np.ndarray],
    ground_truth: Union[List[int], np.ndarray],
    k: int,
) -> float:
    """
    Compute Recall@K for recommendation lists.
    
    Args:
        predictions: List of recommendation lists or 2D array.
        ground_truth: List of true item IDs or 1D array.
        k: Number of top recommendations to consider.
        
    Returns:
        Recall@K score (0.0 to 1.0).
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    hits = 0
    total = len(predictions)
    
    for pred_list, true_item in zip(predictions, ground_truth):
        if isinstance(pred_list, (list, tuple)):
            top_k = list(pred_list)[:k]
        else:
            top_k = pred_list[:k].tolist()
        
        if true_item in top_k:
            hits += 1
    
    return hits / max(total, 1)


def ndcg_at_k(
    predictions: Union[List[List[int]], np.ndarray],
    ground_truth: Union[List[int], np.ndarray],
    k: int,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K.
    
    Args:
        predictions: List of recommendation lists or 2D array.
        ground_truth: List of true item IDs or 1D array.
        k: Number of top recommendations to consider.
        
    Returns:
        NDCG@K score (0.0 to 1.0).
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    total_ndcg = 0.0
    total_queries = len(predictions)
    
    for pred_list, true_item in zip(predictions, ground_truth):
        if isinstance(pred_list, (list, tuple)):
            top_k = list(pred_list)[:k]
        else:
            top_k = pred_list[:k].tolist()
        
        # Find position of true item (1-indexed)
        try:
            position = top_k.index(true_item) + 1
            # DCG = 1 / log2(position + 1)
            dcg = 1.0 / math.log2(position + 1)
        except ValueError:
            # True item not in top-k
            dcg = 0.0
        
        # IDCG = 1 / log2(2) = 1 (since we have only one relevant item)
        idcg = 1.0
        
        # NDCG = DCG / IDCG
        ndcg = dcg / idcg
        total_ndcg += ndcg
    
    return total_ndcg / max(total_queries, 1)


def mrr_at_k(
    predictions: Union[List[List[int]], np.ndarray],
    ground_truth: Union[List[int], np.ndarray],
    k: int,
) -> float:
    """
    Compute Mean Reciprocal Rank at K.
    
    Args:
        predictions: List of recommendation lists or 2D array.
        ground_truth: List of true item IDs or 1D array.
        k: Number of top recommendations to consider.
        
    Returns:
        MRR@K score (0.0 to 1.0).
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    total_rr = 0.0
    total_queries = len(predictions)
    
    for pred_list, true_item in zip(predictions, ground_truth):
        if isinstance(pred_list, (list, tuple)):
            top_k = list(pred_list)[:k]
        else:
            top_k = pred_list[:k].tolist()
        
        # Find position of true item (1-indexed)
        try:
            position = top_k.index(true_item) + 1
            reciprocal_rank = 1.0 / position
        except ValueError:
            # True item not in top-k
            reciprocal_rank = 0.0
        
        total_rr += reciprocal_rank
    
    return total_rr / max(total_queries, 1)


def coverage_at_k(
    predictions: Union[List[List[int]], np.ndarray],
    total_items: int,
    k: int,
) -> float:
    """
    Compute Coverage@K (catalog coverage).
    
    Args:
        predictions: List of recommendation lists or 2D array.
        total_items: Total number of items in the catalog.
        k: Number of top recommendations to consider.
        
    Returns:
        Coverage@K score (0.0 to 1.0).
    """
    recommended_items = set()
    
    for pred_list in predictions:
        if isinstance(pred_list, (list, tuple)):
            top_k = list(pred_list)[:k]
        else:
            top_k = pred_list[:k].tolist()
        
        recommended_items.update(top_k)
    
    return len(recommended_items) / max(total_items, 1)


def diversity_at_k(
    predictions: Union[List[List[int]], np.ndarray],
    item_features: Optional[np.ndarray] = None,
    k: int = 10,
) -> float:
    """
    Compute intra-list diversity at K.
    
    Args:
        predictions: List of recommendation lists or 2D array.
        item_features: Optional item feature matrix for similarity computation.
        k: Number of top recommendations to consider.
        
    Returns:
        Average intra-list diversity score (0.0 to 1.0).
    """
    if item_features is None:
        # Use simple item ID-based diversity (Jaccard distance)
        return _compute_id_diversity(predictions, k)
    else:
        # Use feature-based diversity (cosine distance)
        return _compute_feature_diversity(predictions, item_features, k)


def _compute_id_diversity(predictions: List[List[int]], k: int) -> float:
    """Compute diversity based on unique item IDs."""
    total_diversity = 0.0
    valid_lists = 0
    
    for pred_list in predictions:
        if isinstance(pred_list, (list, tuple)):
            top_k = list(pred_list)[:k]
        else:
            top_k = pred_list[:k].tolist()
        
        if len(top_k) <= 1:
            continue
        
        # Diversity = number of unique items / total items
        diversity = len(set(top_k)) / len(top_k)
        total_diversity += diversity
        valid_lists += 1
    
    return total_diversity / max(valid_lists, 1)


def _compute_feature_diversity(
    predictions: List[List[int]],
    item_features: np.ndarray,
    k: int,
) -> float:
    """Compute diversity based on item feature similarities."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    total_diversity = 0.0
    valid_lists = 0
    
    for pred_list in predictions:
        if isinstance(pred_list, (list, tuple)):
            top_k = list(pred_list)[:k]
        else:
            top_k = pred_list[:k].tolist()
        
        if len(top_k) <= 1:
            continue
        
        # Get features for recommended items
        try:
            list_features = item_features[top_k]
        except IndexError:
            # Some items not in feature matrix, skip
            continue
        
        # Compute pairwise similarities
        similarities = cosine_similarity(list_features)
        
        # Remove diagonal (self-similarity)
        n_items = len(top_k)
        mask = ~np.eye(n_items, dtype=bool)
        pairwise_similarities = similarities[mask]
        
        # Diversity = 1 - average similarity
        diversity = 1.0 - np.mean(pairwise_similarities)
        total_diversity += diversity
        valid_lists += 1
    
    return total_diversity / max(valid_lists, 1)


def compute_all_metrics(
    predictions: Union[List[List[int]], np.ndarray],
    ground_truth: Union[List[int], np.ndarray],
    k_values: List[int],
    total_items: Optional[int] = None,
    item_features: Optional[np.ndarray] = None,
) -> Dict[str, Dict[int, float]]:
    """
    Compute all recommendation metrics for multiple K values.
    
    Args:
        predictions: List of recommendation lists or 2D array.
        ground_truth: List of true item IDs or 1D array.
        k_values: List of K values to evaluate.
        total_items: Total number of items for coverage computation.
        item_features: Optional item features for diversity computation.
        
    Returns:
        Dictionary with metric names as keys and K-value dictionaries as values.
    """
    logger.info(f"Computing metrics for {len(predictions)} predictions")
    
    metrics = {}
    
    # Compute ranking metrics for each K
    for k in k_values:
        logger.debug(f"Computing metrics for K={k}")
        
        if 'Recall' not in metrics:
            metrics['Recall'] = {}
        metrics['Recall'][k] = recall_at_k(predictions, ground_truth, k)
        
        if 'NDCG' not in metrics:
            metrics['NDCG'] = {}
        metrics['NDCG'][k] = ndcg_at_k(predictions, ground_truth, k)
        
        if 'MRR' not in metrics:
            metrics['MRR'] = {}
        metrics['MRR'][k] = mrr_at_k(predictions, ground_truth, k)
        
        # Coverage metrics
        if total_items is not None:
            if 'Coverage' not in metrics:
                metrics['Coverage'] = {}
            metrics['Coverage'][k] = coverage_at_k(predictions, total_items, k)
        
        # Diversity metrics
        if item_features is not None:
            if 'Diversity' not in metrics:
                metrics['Diversity'] = {}
            metrics['Diversity'][k] = diversity_at_k(predictions, item_features, k)
    
    # Log summary
    for metric_name, k_values_dict in metrics.items():
        for k, value in k_values_dict.items():
            logger.info(f"{metric_name}@{k}: {value:.4f}")
    
    return metrics


def evaluate_baseline_models(
    models: Dict[str, object],
    test_interactions: List[Tuple[int, int]],
    k_values: List[int],
    total_items: Optional[int] = None,
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    Evaluate multiple baseline models on test interactions.
    
    Args:
        models: Dictionary of model name to model object.
        test_interactions: List of (user_id, true_item_id) tuples.
        k_values: List of K values to evaluate.
        total_items: Total number of items for coverage computation.
        
    Returns:
        Nested dictionary with model results.
    """
    logger.info(f"Evaluating {len(models)} baseline models")
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}")
        
        predictions = []
        ground_truth = []
        
        for user_id, true_item_id in test_interactions:
            # Get recommendations from model
            if hasattr(model, 'recommend'):
                # Assume model has recommend method
                max_k = max(k_values)
                rec_list = model.recommend(user_id, k=max_k)
                predictions.append(rec_list)
                ground_truth.append(true_item_id)
            else:
                logger.warning(f"Model {model_name} has no recommend method")
                continue
        
        if predictions:
            # Compute metrics
            model_metrics = compute_all_metrics(
                predictions, ground_truth, k_values, total_items
            )
            results[model_name] = model_metrics
        else:
            logger.warning(f"No predictions generated for {model_name}")
    
    return results


class MetricsTracker:
    """
    Utility class for tracking metrics during training and evaluation.
    
    Provides methods to accumulate metrics across batches and compute
    running averages for monitoring training progress.
    """
    
    def __init__(self):
        """Initialize the metrics tracker."""
        self.metrics = {}
        self.counts = {}
    
    def update(self, metric_name: str, value: float, count: int = 1) -> None:
        """
        Update a metric with a new value.
        
        Args:
            metric_name: Name of the metric.
            value: New metric value.
            count: Number of samples this value represents.
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = 0.0
            self.counts[metric_name] = 0
        
        self.metrics[metric_name] += value * count
        self.counts[metric_name] += count
    
    def get_average(self, metric_name: str) -> float:
        """
        Get the running average of a metric.
        
        Args:
            metric_name: Name of the metric.
            
        Returns:
            Average metric value.
        """
        if metric_name not in self.metrics or self.counts[metric_name] == 0:
            return 0.0
        
        return self.metrics[metric_name] / self.counts[metric_name]
    
    def get_all_averages(self) -> Dict[str, float]:
        """
        Get running averages of all tracked metrics.
        
        Returns:
            Dictionary mapping metric names to average values.
        """
        return {
            name: self.get_average(name)
            for name in self.metrics.keys()
        }
    
    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.metrics.clear()
        self.counts.clear()
    
    def summary(self) -> str:
        """
        Get a formatted summary of all metrics.
        
        Returns:
            Formatted string with metric summaries.
        """
        averages = self.get_all_averages()
        
        if not averages:
            return "No metrics tracked"
        
        lines = []
        for name, value in sorted(averages.items()):
            lines.append(f"{name}: {value:.4f}")
        
        return " | ".join(lines)
