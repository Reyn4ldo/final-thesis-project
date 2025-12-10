"""
Association Rule Mining Module

This module contains functions for mining co-resistance patterns
using association rule learning algorithms.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


def prepare_binary_resistance(df: pd.DataFrame, resistance_cols: List[str]) -> pd.DataFrame:
    """
    Convert resistance data to binary format for association rule mining.
    
    Args:
        df: DataFrame with encoded resistance columns (s=0, i=1, r=2)
        resistance_cols: List of resistance column names (e.g., ['ampicillin_encoded', ...])
        
    Returns:
        DataFrame with binary columns (True=resistant, False=not resistant)
    """
    df_binary = pd.DataFrame()
    
    for col in resistance_cols:
        if col in df.columns:
            # Create binary column: True if resistant (value == 2), False otherwise
            # Also handle NaN values by treating them as False
            antibiotic_name = col.replace('_encoded', '')
            df_binary[antibiotic_name] = (df[col] == 2).fillna(False).astype(bool)
    
    return df_binary


def mine_frequent_itemsets(df_binary: pd.DataFrame, min_support: float = 0.02, 
                           use_fpgrowth: bool = False) -> pd.DataFrame:
    """
    Mine frequent itemsets using Apriori or FP-Growth algorithm.
    
    Args:
        df_binary: Binary DataFrame of resistance patterns
        min_support: Minimum support threshold (proportion of samples)
        use_fpgrowth: If True, use FP-Growth instead of Apriori (faster for large datasets)
        
    Returns:
        DataFrame with frequent itemsets and their support values
    """
    if df_binary.empty or len(df_binary.columns) == 0:
        return pd.DataFrame(columns=['support', 'itemsets'])
    
    if use_fpgrowth:
        frequent_itemsets = fpgrowth(df_binary, min_support=min_support, use_colnames=True)
    else:
        frequent_itemsets = apriori(df_binary, min_support=min_support, use_colnames=True)
    
    return frequent_itemsets


def generate_association_rules(frequent_itemsets: pd.DataFrame, 
                               min_confidence: float = 0.6, 
                               min_lift: float = 1.0,
                               metric: str = 'confidence') -> pd.DataFrame:
    """
    Generate association rules from frequent itemsets.
    
    Args:
        frequent_itemsets: DataFrame from mine_frequent_itemsets
        min_confidence: Minimum confidence threshold
        min_lift: Minimum lift threshold
        metric: Metric to use for filtering ('confidence', 'lift', 'support')
        
    Returns:
        DataFrame containing association rules with metrics
    """
    if frequent_itemsets.empty or len(frequent_itemsets) == 0:
        return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 
                                    'confidence', 'lift'])
    
    # Generate rules using the specified metric
    rules = association_rules(frequent_itemsets, metric=metric, 
                             min_threshold=min_confidence)
    
    # Filter by lift if specified
    if min_lift > 0:
        rules = rules[rules['lift'] >= min_lift]
    
    return rules


def filter_top_rules(rules: pd.DataFrame, n: int = 20, 
                     sort_by: str = 'lift') -> pd.DataFrame:
    """
    Filter and return top N association rules.
    
    Args:
        rules: DataFrame of association rules
        n: Number of top rules to return
        sort_by: Column to sort by ('lift', 'confidence', 'support')
        
    Returns:
        DataFrame with top N rules
    """
    if rules.empty:
        return rules
    
    # Ensure sort_by column exists
    if sort_by not in rules.columns:
        sort_by = 'lift'
    
    # Sort and get top N
    top_rules = rules.sort_values(by=sort_by, ascending=False).head(n)
    
    return top_rules


def interpret_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """
    Add human-readable interpretation to association rules.
    
    Args:
        rules: DataFrame of association rules
        
    Returns:
        DataFrame with added 'interpretation' column
    """
    if rules.empty:
        return rules
    
    rules_copy = rules.copy()
    interpretations = []
    
    for idx, row in rules_copy.iterrows():
        # Convert frozensets to readable strings
        antecedents = ', '.join(list(row['antecedents']))
        consequents = ', '.join(list(row['consequents']))
        
        # Create interpretation
        interpretation = (
            f"If resistant to {antecedents} → "
            f"then resistant to {consequents} "
            f"(confidence: {row['confidence']:.2f}, lift: {row['lift']:.2f})"
        )
        interpretations.append(interpretation)
    
    rules_copy['interpretation'] = interpretations
    
    # Add readable versions of antecedents and consequents
    rules_copy['antecedents_str'] = rules_copy['antecedents'].apply(
        lambda x: ', '.join(list(x))
    )
    rules_copy['consequents_str'] = rules_copy['consequents'].apply(
        lambda x: ', '.join(list(x))
    )
    
    return rules_copy


def get_resistance_frequency(df_binary: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate resistance frequency for each antibiotic.
    
    Args:
        df_binary: Binary DataFrame of resistance patterns
        
    Returns:
        DataFrame with antibiotic names and resistance frequencies
    """
    frequencies = []
    
    for col in df_binary.columns:
        freq = df_binary[col].sum() / len(df_binary)
        frequencies.append({
            'antibiotic': col,
            'resistance_frequency': freq,
            'resistant_count': df_binary[col].sum(),
            'total_count': len(df_binary)
        })
    
    freq_df = pd.DataFrame(frequencies)
    freq_df = freq_df.sort_values('resistance_frequency', ascending=False)
    
    return freq_df
