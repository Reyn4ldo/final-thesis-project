"""
Unit tests for unsupervised learning and association rules modules

Tests for clustering, dimensionality reduction, and association rule mining.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.unsupervised import (
    perform_kmeans, perform_hierarchical, perform_dbscan,
    find_optimal_clusters, get_cluster_summary,
    perform_pca, perform_tsne, perform_umap, get_pca_loadings
)
from src.models.association_rules import (
    prepare_binary_resistance, mine_frequent_itemsets,
    generate_association_rules, filter_top_rules, interpret_rules,
    get_resistance_frequency
)


class TestClustering:
    """Test cases for clustering functions."""
    
    def test_perform_kmeans(self):
        """Test K-means clustering."""
        X = np.random.randn(100, 10)
        labels, model = perform_kmeans(X, n_clusters=3, random_state=42)
        
        assert len(labels) == 100
        assert len(np.unique(labels)) <= 3
        assert model is not None
    
    def test_perform_hierarchical(self):
        """Test hierarchical clustering."""
        X = np.random.randn(100, 10)
        labels, model = perform_hierarchical(X, n_clusters=3, linkage='ward')
        
        assert len(labels) == 100
        assert len(np.unique(labels)) == 3
        assert model is not None
    
    def test_perform_dbscan(self):
        """Test DBSCAN clustering."""
        X = np.random.randn(100, 10)
        labels, model = perform_dbscan(X, eps=0.5, min_samples=5)
        
        assert len(labels) == 100
        assert model is not None
        # DBSCAN may have outliers labeled as -1
        assert -1 in labels or len(np.unique(labels)) > 0
    
    def test_find_optimal_clusters(self):
        """Test finding optimal number of clusters."""
        X = np.random.randn(100, 10)
        k_range, inertias, silhouette_scores = find_optimal_clusters(X, max_k=5)
        
        assert len(k_range) == 4  # 2 to 5
        assert len(inertias) == 4
        assert len(silhouette_scores) == 4
        assert all(isinstance(k, int) for k in k_range)
        assert all(i > 0 for i in inertias)
        assert all(-1 <= s <= 1 for s in silhouette_scores)
    
    def test_get_cluster_summary(self):
        """Test cluster summary generation."""
        df = pd.DataFrame({
            'bacterial_species': ['sp1'] * 50 + ['sp2'] * 50,
            'MAR_index': np.random.rand(100),
            'administrative_region': ['region1'] * 100,
            'feat1_encoded': np.random.choice([0, 1, 2], 100),
            'feat2_encoded': np.random.choice([0, 1, 2], 100)
        })
        labels = np.array([0] * 50 + [1] * 50)
        resistance_cols = ['feat1_encoded', 'feat2_encoded']
        
        summary = get_cluster_summary(df, labels, resistance_cols)
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2  # 2 clusters
        assert 'Cluster' in summary.columns
        assert 'N' in summary.columns
        assert 'Top_Species' in summary.columns


class TestDimensionalityReduction:
    """Test cases for dimensionality reduction functions."""
    
    def test_perform_pca(self):
        """Test PCA."""
        X = np.random.randn(100, 20)
        X_transformed, model = perform_pca(X, n_components=2)
        
        assert X_transformed.shape == (100, 2)
        assert model is not None
        assert hasattr(model, 'explained_variance_ratio_')
        assert len(model.explained_variance_ratio_) == 2
    
    def test_perform_tsne(self):
        """Test t-SNE."""
        X = np.random.randn(50, 10)
        X_transformed = perform_tsne(X, n_components=2, perplexity=10, random_state=42)
        
        assert X_transformed.shape == (50, 2)
    
    def test_perform_umap(self):
        """Test UMAP."""
        X = np.random.randn(50, 10)
        X_transformed = perform_umap(X, n_components=2, n_neighbors=10, random_state=42)
        
        assert X_transformed.shape == (50, 2)
    
    def test_get_pca_loadings(self):
        """Test PCA loadings extraction."""
        X = np.random.randn(100, 10)
        X_transformed, pca_model = perform_pca(X, n_components=3)
        
        feature_names = [f'feat_{i}' for i in range(10)]
        loadings = get_pca_loadings(pca_model, feature_names)
        
        assert isinstance(loadings, pd.DataFrame)
        assert loadings.shape == (10, 3)
        assert list(loadings.columns) == ['PC1', 'PC2', 'PC3']
        assert list(loadings.index) == feature_names


class TestAssociationRules:
    """Test cases for association rule mining functions."""
    
    def test_prepare_binary_resistance(self):
        """Test binary resistance conversion."""
        df = pd.DataFrame({
            'ab1_encoded': [0, 1, 2, 2, np.nan],
            'ab2_encoded': [2, 0, 2, 1, 0],
            'ab3_encoded': [0, 2, 0, 2, 2]
        })
        resistance_cols = ['ab1_encoded', 'ab2_encoded', 'ab3_encoded']
        
        df_binary = prepare_binary_resistance(df, resistance_cols)
        
        assert isinstance(df_binary, pd.DataFrame)
        assert df_binary.shape[0] == 5
        assert all(col.replace('_encoded', '') in df_binary.columns for col in resistance_cols)
        assert df_binary.dtypes.all() == bool or all(df_binary.dtypes == bool)
        
        # Check conversion: resistant (2) -> True, else -> False
        assert df_binary['ab1'].iloc[2] == True  # 2 -> True
        assert df_binary['ab1'].iloc[0] == False  # 0 -> False
        assert df_binary['ab1'].iloc[1] == False  # 1 -> False
        assert df_binary['ab1'].iloc[4] == False  # NaN -> False
    
    def test_mine_frequent_itemsets(self):
        """Test frequent itemset mining."""
        df_binary = pd.DataFrame({
            'ab1': [True, True, True, False, False],
            'ab2': [True, True, False, True, False],
            'ab3': [False, True, True, True, True]
        })
        
        itemsets = mine_frequent_itemsets(df_binary, min_support=0.4)
        
        assert isinstance(itemsets, pd.DataFrame)
        assert 'support' in itemsets.columns
        assert 'itemsets' in itemsets.columns
        assert all(itemsets['support'] >= 0.4)
    
    def test_generate_association_rules(self):
        """Test association rule generation."""
        df_binary = pd.DataFrame({
            'ab1': [True, True, True, True, False],
            'ab2': [True, True, True, False, False],
            'ab3': [True, False, True, True, True]
        })
        
        itemsets = mine_frequent_itemsets(df_binary, min_support=0.2)
        
        if len(itemsets) > 1:  # Need at least 2 itemsets to generate rules
            rules = generate_association_rules(itemsets, min_confidence=0.5, min_lift=1.0)
            
            assert isinstance(rules, pd.DataFrame)
            if len(rules) > 0:
                assert 'antecedents' in rules.columns
                assert 'consequents' in rules.columns
                assert 'confidence' in rules.columns
                assert 'lift' in rules.columns
                assert all(rules['confidence'] >= 0.5)
                assert all(rules['lift'] >= 1.0)
    
    def test_filter_top_rules(self):
        """Test rule filtering."""
        # Create sample rules dataframe
        rules = pd.DataFrame({
            'antecedents': [frozenset(['ab1']), frozenset(['ab2']), frozenset(['ab3'])],
            'consequents': [frozenset(['ab2']), frozenset(['ab3']), frozenset(['ab1'])],
            'support': [0.5, 0.6, 0.7],
            'confidence': [0.8, 0.9, 0.85],
            'lift': [1.5, 2.0, 1.8]
        })
        
        top_rules = filter_top_rules(rules, n=2, sort_by='lift')
        
        assert len(top_rules) == 2
        assert top_rules.iloc[0]['lift'] >= top_rules.iloc[1]['lift']
    
    def test_interpret_rules(self):
        """Test rule interpretation."""
        rules = pd.DataFrame({
            'antecedents': [frozenset(['ab1'])],
            'consequents': [frozenset(['ab2'])],
            'support': [0.5],
            'confidence': [0.8],
            'lift': [1.5]
        })
        
        interpreted = interpret_rules(rules)
        
        assert 'interpretation' in interpreted.columns
        assert 'antecedents_str' in interpreted.columns
        assert 'consequents_str' in interpreted.columns
        assert 'ab1' in interpreted['interpretation'].iloc[0]
        assert 'ab2' in interpreted['interpretation'].iloc[0]
    
    def test_get_resistance_frequency(self):
        """Test resistance frequency calculation."""
        df_binary = pd.DataFrame({
            'ab1': [True, True, False, False],
            'ab2': [True, False, False, False],
            'ab3': [True, True, True, True]
        })
        
        freq_df = get_resistance_frequency(df_binary)
        
        assert isinstance(freq_df, pd.DataFrame)
        assert 'antibiotic' in freq_df.columns
        assert 'resistance_frequency' in freq_df.columns
        assert len(freq_df) == 3
        assert freq_df['resistance_frequency'].iloc[0] >= freq_df['resistance_frequency'].iloc[1]


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
