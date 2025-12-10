"""
Unsupervised Learning Module

This module contains implementations of unsupervised learning algorithms
for clustering and dimensionality reduction.
"""

from typing import Tuple, Optional, Dict, List
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap


# ============================================================================
# Clustering Functions
# ============================================================================

def perform_kmeans(X: np.ndarray, n_clusters: int = 3, random_state: int = 42) -> Tuple[np.ndarray, object]:
    """
    Perform K-Means clustering.
    
    Args:
        X: Feature array or DataFrame
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (cluster labels, fitted KMeans model)
    """
    X_array = np.array(X) if isinstance(X, pd.DataFrame) else X
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_array)
    return labels, model


def perform_hierarchical(X: np.ndarray, n_clusters: int = 3, linkage: str = 'ward') -> Tuple[np.ndarray, object]:
    """
    Perform Hierarchical (Agglomerative) clustering.
    
    Args:
        X: Feature array or DataFrame
        n_clusters: Number of clusters
        linkage: Linkage criterion ("ward", "complete", "average", "single")
        
    Returns:
        Tuple of (cluster labels, fitted AgglomerativeClustering model)
    """
    X_array = np.array(X) if isinstance(X, pd.DataFrame) else X
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X_array)
    return labels, model


def perform_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, object]:
    """
    Perform DBSCAN clustering for outlier detection.
    
    Args:
        X: Feature array or DataFrame
        eps: Maximum distance between samples in neighborhood
        min_samples: Minimum samples in a neighborhood
        
    Returns:
        Tuple of (cluster labels, fitted DBSCAN model)
        Note: Outliers are labeled as -1
    """
    X_array = np.array(X) if isinstance(X, pd.DataFrame) else X
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_array)
    return labels, model


def find_optimal_clusters(X: np.ndarray, max_k: int = 10) -> Tuple[List[int], List[float], List[float]]:
    """
    Find optimal number of clusters using elbow method and silhouette scores.
    
    Args:
        X: Feature array or DataFrame
        max_k: Maximum number of clusters to test
        
    Returns:
        Tuple of (k_range, inertias, silhouette_scores)
    """
    X_array = np.array(X) if isinstance(X, pd.DataFrame) else X
    k_range = list(range(2, max_k + 1))
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_array)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_array, labels))
    
    return k_range, inertias, silhouette_scores


def get_cluster_summary(df: pd.DataFrame, labels: np.ndarray, feature_cols: List[str]) -> pd.DataFrame:
    """
    Summarize clusters by species, MAR, and resistance patterns.
    
    Args:
        df: Original dataframe with metadata
        labels: Cluster labels
        feature_cols: List of resistance feature column names
        
    Returns:
        DataFrame with cluster summary statistics
    """
    df_copy = df.copy()
    df_copy['cluster'] = labels
    
    summaries = []
    for cluster_id in sorted(df_copy['cluster'].unique()):
        if cluster_id == -1:  # DBSCAN outliers
            cluster_name = 'Outliers'
        else:
            cluster_name = f'Cluster {cluster_id}'
        
        cluster_data = df_copy[df_copy['cluster'] == cluster_id]
        
        # Basic stats
        n_samples = len(cluster_data)
        
        # Species distribution
        if 'bacterial_species' in df.columns:
            top_species = cluster_data['bacterial_species'].mode()
            top_species = top_species[0] if len(top_species) > 0 else 'N/A'
        else:
            top_species = 'N/A'
        
        # MAR index
        if 'MAR_index' in df.columns:
            avg_mar = cluster_data['MAR_index'].mean()
        else:
            avg_mar = np.nan
        
        # Key resistances - antibiotics with >50% resistance in cluster
        key_resistances = []
        for col in feature_cols:
            if col in cluster_data.columns:
                # Check if mostly resistant (encoded value 2)
                resistant_pct = (cluster_data[col] == 2).sum() / n_samples
                if resistant_pct > 0.5:
                    # Extract antibiotic name
                    ab_name = col.replace('_encoded', '')
                    key_resistances.append(ab_name)
        
        # Region distribution
        if 'administrative_region' in df.columns:
            top_region = cluster_data['administrative_region'].mode()
            top_region = top_region[0] if len(top_region) > 0 else 'N/A'
        else:
            top_region = 'N/A'
        
        summaries.append({
            'Cluster': cluster_name,
            'N': n_samples,
            'Top_Species': top_species,
            'Avg_MAR': avg_mar,
            'Key_Resistances': ', '.join(key_resistances[:5]) if key_resistances else 'None',
            'Top_Region': top_region
        })
    
    return pd.DataFrame(summaries)


# ============================================================================
# Dimensionality Reduction Functions
# ============================================================================

def perform_pca(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, object]:
    """
    Perform Principal Component Analysis with variance explained.
    
    Args:
        X: Feature array or DataFrame
        n_components: Number of principal components
        
    Returns:
        Tuple of (transformed data, fitted PCA model)
    """
    X_array = np.array(X) if isinstance(X, pd.DataFrame) else X
    model = PCA(n_components=n_components)
    X_transformed = model.fit_transform(X_array)
    return X_transformed, model


def perform_tsne(X: np.ndarray, n_components: int = 2, perplexity: int = 30, 
                 random_state: int = 42) -> np.ndarray:
    """
    Perform t-SNE dimensionality reduction.
    
    Args:
        X: Feature array or DataFrame
        n_components: Number of dimensions (typically 2 or 3)
        perplexity: Perplexity parameter for t-SNE
        random_state: Random seed for reproducibility
        
    Returns:
        Transformed data array
    """
    X_array = np.array(X) if isinstance(X, pd.DataFrame) else X
    model = TSNE(n_components=n_components, perplexity=perplexity, 
                 random_state=random_state, max_iter=1000)
    X_transformed = model.fit_transform(X_array)
    return X_transformed


def perform_umap(X: np.ndarray, n_components: int = 2, n_neighbors: int = 15, 
                 min_dist: float = 0.1, random_state: int = 42) -> np.ndarray:
    """
    Perform UMAP dimensionality reduction.
    
    Args:
        X: Feature array or DataFrame
        n_components: Number of dimensions
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        random_state: Random seed for reproducibility
        
    Returns:
        Transformed data array
    """
    X_array = np.array(X) if isinstance(X, pd.DataFrame) else X
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=random_state)
    X_transformed = reducer.fit_transform(X_array)
    return X_transformed


def get_pca_loadings(pca_model: PCA, feature_names: List[str]) -> pd.DataFrame:
    """
    Get feature loadings for PCA components.
    
    Args:
        pca_model: Fitted PCA model
        feature_names: List of feature names
        
    Returns:
        DataFrame with loadings for each component
    """
    loadings = pd.DataFrame(
        pca_model.components_.T,
        columns=[f'PC{i+1}' for i in range(pca_model.n_components_)],
        index=feature_names
    )
    return loadings


# ============================================================================
# Legacy Functions (kept for backward compatibility)
# ============================================================================

def perform_kmeans_clustering(X: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    """Legacy wrapper for perform_kmeans."""
    return perform_kmeans(X, n_clusters, random_state)


def perform_hierarchical_clustering(X: pd.DataFrame, n_clusters: int = 3, linkage: str = "ward"):
    """Legacy wrapper for perform_hierarchical."""
    return perform_hierarchical(X, n_clusters, linkage)


def perform_dbscan_clustering(X: pd.DataFrame, eps: float = 0.5, min_samples: int = 5):
    """Legacy wrapper for perform_dbscan."""
    return perform_dbscan(X, eps, min_samples)
