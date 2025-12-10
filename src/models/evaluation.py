"""
Model Evaluation Module

This module contains functions for evaluating model performance
using various metrics and visualizations.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    roc_auc_score
)
from sklearn.inspection import permutation_importance


def evaluate_classifier(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    average: str = 'weighted'
) -> Dict[str, float]:
    """
    Evaluate classifier with comprehensive metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        average: Averaging method for multi-class ('weighted', 'macro', 'micro')
        
    Returns:
        Dictionary containing accuracy, precision, recall, F1, and optionally AUC
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # Add AUC if probabilities are provided
    if y_proba is not None:
        try:
            # Binary classification
            if len(np.unique(y_true)) == 2:
                # Use probabilities for positive class
                if y_proba.ndim > 1 and y_proba.shape[1] == 2:
                    metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['auc'] = roc_auc_score(y_true, y_proba)
            else:
                # Multi-class AUC with ovr (one-vs-rest)
                metrics['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average=average)
        except Exception as e:
            # If AUC calculation fails, skip it
            pass
    
    return metrics


def get_confusion_matrix(y_true, y_pred) -> np.ndarray:
    """
    Generate confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true,
    y_pred,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate classification report as dictionary.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional list of class names
        
    Returns:
        Classification report dictionary
    """
    return classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)


def calculate_roc_auc(y_true, y_proba, multi_class: str = 'ovr', average: str = 'weighted') -> float:
    """
    Calculate ROC-AUC score for binary or multi-class classification.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        multi_class: Strategy for multi-class ('ovr' or 'ovo')
        average: Averaging method for multi-class
        
    Returns:
        ROC-AUC score
    """
    try:
        # Binary classification
        if len(np.unique(y_true)) == 2:
            # Use probabilities for positive class
            if y_proba.ndim > 1 and y_proba.shape[1] == 2:
                return roc_auc_score(y_true, y_proba[:, 1])
            else:
                return roc_auc_score(y_true, y_proba)
        else:
            # Multi-class
            return roc_auc_score(y_true, y_proba, multi_class=multi_class, average=average)
    except Exception as e:
        print(f"Error calculating ROC-AUC: {e}")
        return 0.0


def get_feature_importance(
    model,
    feature_names: List[str],
    model_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Extract feature importances from a model.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_type: Type of model (optional, auto-detected)
        
    Returns:
        DataFrame with features and their importances
    """
    importances = None
    
    # Try to get feature importances from tree-based models
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    # Try to get coefficients from linear models
    elif hasattr(model, 'coef_'):
        # For multi-class, take mean of absolute coefficients
        coef = model.coef_
        if coef.ndim > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)
    else:
        raise ValueError(f"Model does not have feature_importances_ or coef_ attributes")
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    return importance_df


def get_permutation_importance(
    model,
    X,
    y,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Calculate permutation importance for any model.
    
    Args:
        model: Trained model
        X: Feature data
        y: Target labels
        feature_names: List of feature names
        n_repeats: Number of times to permute each feature
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame with features and their permutation importances
    """
    perm_importance = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
    
    return importance_df


def compare_models(results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create comparison DataFrame from results dictionary.
    
    Args:
        results_dict: Dictionary mapping model names to their metrics
                     Format: {'model_name': {'accuracy': 0.9, 'f1': 0.85, ...}}
        
    Returns:
        DataFrame comparing all models
    """
    comparison_df = pd.DataFrame(results_dict).T
    
    # Sort by F1 score if available
    if 'f1' in comparison_df.columns:
        comparison_df = comparison_df.sort_values('f1', ascending=False)
    
    return comparison_df


def get_best_model(
    results_dict: Dict[str, Dict[str, float]],
    metric: str = 'f1'
) -> str:
    """
    Get the name of the best performing model based on a metric.
    
    Args:
        results_dict: Dictionary mapping model names to their metrics
        metric: Metric to use for comparison (default: 'f1')
        
    Returns:
        Name of the best model
    """
    comparison_df = compare_models(results_dict)
    
    if metric not in comparison_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results")
    
    best_model_name = comparison_df[metric].idxmax()
    return best_model_name


def calculate_clustering_metrics(X, labels) -> Dict[str, float]:
    """
    Calculate clustering evaluation metrics (silhouette score, Davies-Bouldin index).
    
    Args:
        X: Feature data
        labels: Cluster labels
        
    Returns:
        Dictionary containing clustering metrics
        
    TODO: Implement clustering metrics
    """
    pass


def perform_cross_validation(model, X, y, cv: int = 5, scoring: str = "accuracy"):
    """
    Perform k-fold cross-validation.
    
    Args:
        model: Scikit-learn model
        X: Features
        y: Labels
        cv: Number of folds
        scoring: Scoring metric
        
    Returns:
        Cross-validation scores
        
    TODO: Implement cross-validation
    """
    pass
