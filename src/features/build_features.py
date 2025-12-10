"""
Feature Engineering Module

This module contains functions for feature extraction and engineering,
including MAR (Multiple Antibiotic Resistance) index calculation.
"""

from typing import List
import pandas as pd
import numpy as np


def extract_resistance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only encoded resistance interpretation columns.
    
    Args:
        df: DataFrame containing encoded resistance data
        
    Returns:
        DataFrame with only encoded resistance features
    """
    # Find all encoded columns (ending with _encoded)
    encoded_cols = [col for col in df.columns if col.endswith('_encoded')]
    
    if not encoded_cols:
        raise ValueError("No encoded resistance columns found. Run encode_resistance first.")
    
    return df[encoded_cols].copy()


def create_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create final feature matrix for machine learning.
    
    Args:
        df: DataFrame containing encoded resistance data
        
    Returns:
        DataFrame with feature matrix ready for ML models
    """
    # Extract resistance features
    feature_matrix = extract_resistance_features(df)
    
    # Fill any remaining NaN values with -1 to indicate not tested
    # (alternative: could drop rows with NaN or use imputation)
    feature_matrix = feature_matrix.fillna(-1)
    
    return feature_matrix
