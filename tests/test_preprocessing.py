"""
Unit tests for preprocessing functions

Tests for data cleaning, encoding, and preprocessing functions.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import (
    load_raw_data,
    identify_antibiotic_columns,
    clean_interpretation_values,
    handle_missing_values,
    encode_resistance,
    standardize_species_labels,
    calculate_mar_index,
    create_mar_target,
    prepare_species_target
)
from src.data.splitting import stratified_split
from src.features.build_features import extract_resistance_features, create_feature_matrix


class TestPreprocessing:
    """Test cases for preprocessing module."""
    
    def test_load_raw_data(self):
        """Test loading raw data from CSV."""
        df = load_raw_data('data/raw/rawdata.csv')
        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0
        assert 'bacterial_species' in df.columns
    
    def test_identify_antibiotic_columns(self):
        """Test identifying antibiotic interpretation columns."""
        df = load_raw_data('data/raw/rawdata.csv')
        int_cols = identify_antibiotic_columns(df)
        assert len(int_cols) > 0
        assert all(col.endswith('_int') for col in int_cols)
    
    def test_clean_interpretation_values(self):
        """Test cleaning interpretation values."""
        # Create test dataframe
        df = pd.DataFrame({
            'test_int': ['s', 'S', '*r', 'R', '*i', 'nan', None]
        })
        cleaned = clean_interpretation_values(df, ['test_int'])
        
        # Check that values are standardized
        valid_values = cleaned['test_int'].dropna().unique()
        assert all(val in ['s', 'i', 'r'] for val in valid_values)
    
    def test_handle_missing_values(self):
        """Test handling missing values."""
        df = pd.DataFrame({
            'col1_int': [np.nan, 's', 'r'],
            'col2_int': [np.nan, np.nan, 'i'],
            'other': [1, 2, 3]
        })
        cols = ['col1_int', 'col2_int']
        
        # Test drop strategy
        result = handle_missing_values(df, cols, strategy='drop')
        assert len(result) == 2  # First row should be dropped (all NaN)
    
    def test_encode_resistance(self):
        """Test resistance encoding."""
        df = pd.DataFrame({
            'test_int': ['s', 'i', 'r', np.nan]
        })
        encoded = encode_resistance(df, ['test_int'])
        
        # Check encoding
        assert 'test_encoded' in encoded.columns
        assert encoded['test_encoded'].iloc[0] == 0  # s = 0
        assert encoded['test_encoded'].iloc[1] == 1  # i = 1
        assert encoded['test_encoded'].iloc[2] == 2  # r = 2
        assert pd.isna(encoded['test_encoded'].iloc[3])  # nan stays nan
    
    def test_standardize_species_labels(self):
        """Test species label standardization."""
        df = pd.DataFrame({
            'bacterial_species': ['E. Coli', 'e. coli', '  Klebsiella  ']
        })
        result = standardize_species_labels(df)
        
        # All should be lowercase and stripped
        assert result['bacterial_species'].iloc[0] == 'e. coli'
        assert result['bacterial_species'].iloc[1] == 'e. coli'
        assert result['bacterial_species'].iloc[2] == 'klebsiella'
    
    def test_calculate_mar_index(self):
        """Test MAR index calculation."""
        df = pd.DataFrame({
            'ab1_encoded': [2, 0, 2, np.nan],  # resistant, susceptible, resistant, missing
            'ab2_encoded': [2, 0, 0, 2],       # resistant, susceptible, susceptible, resistant
            'ab3_encoded': [0, 0, 2, 0]        # susceptible, susceptible, resistant, susceptible
        })
        
        mar_index = calculate_mar_index(df, ['ab1_encoded', 'ab2_encoded', 'ab3_encoded'])
        
        # Row 0: 2/3 = 0.667
        assert abs(mar_index.iloc[0] - 2/3) < 0.01
        # Row 1: 0/3 = 0
        assert mar_index.iloc[1] == 0
        # Row 2: 2/3 = 0.667 (ab1 and ab3 are resistant, ab2 is susceptible)
        assert abs(mar_index.iloc[2] - 2/3) < 0.01
        # Row 3: 1/2 = 0.5 (one missing, one resistant, one susceptible)
        assert abs(mar_index.iloc[3] - 0.5) < 0.01
    
    def test_create_mar_target(self):
        """Test MAR target creation."""
        df = pd.DataFrame({
            'MAR_index': [0.1, 0.25, 0.5, 0.2]
        })
        
        high_mar = create_mar_target(df, threshold=0.2)
        
        assert high_mar.iloc[0] == 0  # 0.1 <= 0.2
        assert high_mar.iloc[1] == 1  # 0.25 > 0.2
        assert high_mar.iloc[2] == 1  # 0.5 > 0.2
        assert high_mar.iloc[3] == 0  # 0.2 <= 0.2
    
    def test_prepare_species_target(self):
        """Test species target preparation."""
        df = pd.DataFrame({
            'bacterial_species': ['sp1'] * 15 + ['sp2'] * 8 + ['sp3'] * 20
        })
        
        result = prepare_species_target(df, min_samples=10)
        
        assert 'species_target' in result.columns
        # sp1 and sp3 should be kept (>= 10 samples)
        # sp2 should be merged to 'other' (< 10 samples)
        assert 'sp1' in result['species_target'].values
        assert 'sp3' in result['species_target'].values
        assert 'other' in result['species_target'].values
        assert 'sp2' not in result['species_target'].values


class TestSplitting:
    """Test cases for data splitting module."""
    
    def test_stratified_split(self):
        """Test stratified data splitting."""
        # Create sample data
        X = pd.DataFrame({
            'feat1': range(100),
            'feat2': range(100, 200)
        })
        y = pd.Series([0] * 70 + [1] * 30)
        
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
            X, y, train_size=0.7, val_size=0.2, test_size=0.1, random_state=42
        )
        
        # Check sizes (allow 1 sample tolerance due to stratification rounding)
        assert abs(len(X_train) - 70) <= 1
        assert abs(len(X_val) - 20) <= 1
        assert abs(len(X_test) - 10) <= 1
        
        # Check all samples are accounted for
        assert len(X_train) + len(X_val) + len(X_test) == 100
        
        # Check stratification (proportions should be similar)
        train_prop = y_train.mean()
        val_prop = y_val.mean()
        test_prop = y_test.mean()
        original_prop = y.mean()
        
        # Allow 10% tolerance
        assert abs(train_prop - original_prop) < 0.1
        assert abs(val_prop - original_prop) < 0.1
        assert abs(test_prop - original_prop) < 0.15  # More tolerance for smaller set


class TestFeatureBuilding:
    """Test cases for feature building module."""
    
    def test_extract_resistance_features(self):
        """Test extracting resistance features."""
        df = pd.DataFrame({
            'ab1_encoded': [0, 1, 2],
            'ab2_encoded': [2, 1, 0],
            'other_col': [1, 2, 3]
        })
        
        features = extract_resistance_features(df)
        
        assert 'ab1_encoded' in features.columns
        assert 'ab2_encoded' in features.columns
        assert 'other_col' not in features.columns
    
    def test_create_feature_matrix(self):
        """Test creating feature matrix."""
        df = pd.DataFrame({
            'ab1_encoded': [0, np.nan, 2],
            'ab2_encoded': [2, 1, np.nan],
            'other_col': [1, 2, 3]
        })
        
        X = create_feature_matrix(df)
        
        # Should fill NaN with -1
        assert X['ab1_encoded'].iloc[1] == -1
        assert X['ab2_encoded'].iloc[2] == -1
        # Should not contain non-encoded columns
        assert 'other_col' not in X.columns


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
