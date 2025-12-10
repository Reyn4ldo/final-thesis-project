"""
Data Preprocessing Module

This module contains functions for cleaning and encoding the AMR dataset.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw data from CSV file with proper handling of special characters.
    
    Args:
        filepath: Path to the raw data CSV file
        
    Returns:
        DataFrame containing the raw data
    """
    df = pd.read_csv(filepath, encoding='utf-8')
    return df


def identify_antibiotic_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify antibiotic interpretation columns (ending with '_int').
    
    Args:
        df: DataFrame containing antibiotic data
        
    Returns:
        List of interpretation column names
    """
    return [col for col in df.columns if col.endswith('_int')]


def clean_interpretation_values(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Standardize antibiotic interpretation values (s, i, r).
    
    Handles values like 's', 'S', '*s', 'r', 'R', '*r', 'i', 'I', '*i'
    and converts them to lowercase 's', 'i', 'r'.
    
    Args:
        df: DataFrame with interpretation columns
        columns: List of interpretation column names
        
    Returns:
        DataFrame with standardized interpretation values
    """
    df = df.copy()
    
    for col in tqdm(columns, desc="Cleaning interpretation values"):
        if col in df.columns:
            # Convert to string, strip whitespace, remove *, convert to lowercase
            df[col] = df[col].astype(str).str.strip().str.replace('*', '', regex=False).str.lower()
            # Replace 'nan' string back to actual NaN and keep only valid values
            df[col] = df[col].apply(lambda x: x if x in ['s', 'i', 'r'] else np.nan)
    
    return df


def handle_missing_values(df: pd.DataFrame, columns: List[str], strategy: str = "drop") -> pd.DataFrame:
    """
    Handle missing values in antibiotic interpretation columns.
    
    Args:
        df: DataFrame with potential missing values
        columns: List of columns to handle
        strategy: Strategy for handling missing values ("drop" or "keep")
        
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    if strategy == "drop":
        # Drop rows where ALL antibiotic columns are missing
        df = df.dropna(subset=columns, how='all')
    
    return df


def encode_resistance(df: pd.DataFrame, columns: List[str], method: str = 'ordinal') -> pd.DataFrame:
    """
    Encode antibiotic resistance interpretations as numerical values.
    
    Encoding: s=0 (susceptible), i=1 (intermediate), r=2 (resistant)
    
    Args:
        df: DataFrame with interpretation columns
        columns: List of interpretation column names
        method: Encoding method ('ordinal' for s=0, i=1, r=2)
        
    Returns:
        DataFrame with encoded resistance values
    """
    df = df.copy()
    
    encoding_map = {'s': 0, 'i': 1, 'r': 2}
    
    for col in tqdm(columns, desc="Encoding resistance values"):
        if col in df.columns:
            # Create new column with _encoded suffix
            encoded_col = col.replace('_int', '_encoded')
            df[encoded_col] = df[col].map(encoding_map)
    
    return df


def standardize_species_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize bacterial_species column.
    
    Args:
        df: DataFrame with bacterial_species column
        
    Returns:
        DataFrame with standardized species labels
    """
    df = df.copy()
    
    if 'bacterial_species' in df.columns:
        # Convert to lowercase and strip whitespace
        df['bacterial_species'] = df['bacterial_species'].astype(str).str.strip().str.lower()
        # Replace underscores with spaces for consistency (optional)
        # df['bacterial_species'] = df['bacterial_species'].str.replace('_', ' ')
    
    return df


def calculate_mar_index(df: pd.DataFrame, resistance_columns: List[str]) -> pd.Series:
    """
    Calculate Multiple Antibiotic Resistance (MAR) index for each isolate.
    
    MAR index = Number of resistant antibiotics / Total antibiotics tested
    
    Args:
        df: DataFrame containing encoded resistance data
        resistance_columns: List of encoded resistance column names
        
    Returns:
        Series containing MAR index for each sample
    """
    # Count resistant (value = 2) for each row
    resistant_count = (df[resistance_columns] == 2).sum(axis=1)
    
    # Count non-missing values (tested antibiotics) for each row
    tested_count = df[resistance_columns].notna().sum(axis=1)
    
    # Calculate MAR index (avoid division by zero)
    mar_index = resistant_count / tested_count.replace(0, np.nan)
    
    return mar_index


def create_mar_target(df: pd.DataFrame, threshold: float = 0.2) -> pd.Series:
    """
    Create binary High_MAR target variable.
    
    Args:
        df: DataFrame with MAR_index column
        threshold: Threshold for classifying as high MAR (default 0.2)
        
    Returns:
        Series with binary classification (1 if MAR > threshold, else 0)
    """
    if 'MAR_index' not in df.columns:
        raise ValueError("MAR_index column not found in DataFrame")
    
    high_mar = (df['MAR_index'] > threshold).astype(int)
    
    return high_mar


def prepare_species_target(df: pd.DataFrame, min_samples: int = 10) -> pd.DataFrame:
    """
    Prepare species classification target by merging rare species.
    
    Args:
        df: DataFrame with bacterial_species column
        min_samples: Minimum number of samples for a species to be kept separate
        
    Returns:
        DataFrame with processed species column
    """
    df = df.copy()
    
    if 'bacterial_species' not in df.columns:
        raise ValueError("bacterial_species column not found in DataFrame")
    
    # Count samples per species
    species_counts = df['bacterial_species'].value_counts()
    
    # Identify rare species
    rare_species = species_counts[species_counts < min_samples].index
    
    # Replace rare species with 'Other'
    df['species_target'] = df['bacterial_species'].apply(
        lambda x: 'other' if x in rare_species else x
    )
    
    return df
