"""
Data Splitting Module

This module contains functions for splitting data into train/validation/test sets.
"""

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        
    TODO: Implement stratified splitting to maintain class distribution
    """
    pass


def create_stratified_folds(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
    random_state: int = 42
):
    """
    Create stratified K-fold splits for cross-validation.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        n_folds: Number of folds
        random_state: Random seed for reproducibility
        
    Returns:
        Generator yielding (train_idx, val_idx) for each fold
        
    TODO: Implement using StratifiedKFold from sklearn
    """
    pass


def balance_dataset(X: pd.DataFrame, y: pd.Series, method: str = "oversample"):
    """
    Balance the dataset to handle class imbalance.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        method: Balancing method ("oversample", "undersample", "smote")
        
    Returns:
        Balanced X and y
        
    TODO: Implement balancing using imbalanced-learn library
    """
    pass
