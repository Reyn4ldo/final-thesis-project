"""
Data Splitting Module

This module contains functions for splitting data into train/validation/test sets.
"""

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets with stratification.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        train_size: Proportion of data for training set (default 0.7)
        val_size: Proportion of data for validation set (default 0.2)
        test_size: Proportion of data for test set (default 0.1)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Verify proportions sum to 1
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"train_size, val_size, and test_size must sum to 1.0, got {total}")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    # Second split: separate train and validation
    # Adjust val_size relative to remaining data
    val_size_adjusted = val_size / (train_size + val_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
