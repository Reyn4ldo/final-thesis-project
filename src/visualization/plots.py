"""
Visualization Module

This module contains functions for creating plots and visualizations
for data exploration and model results.
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.graph_objects as go
import plotly.express as px


def plot_data_distribution(df: pd.DataFrame, column: str, figsize: tuple = (10, 6)):
    """
    Plot distribution of a single column.
    
    Args:
        df: DataFrame containing the data
        column: Column name to plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
        
    TODO: Implement distribution plots (histogram, KDE)
    """
    pass


def plot_correlation_matrix(df: pd.DataFrame, figsize: tuple = (12, 10)):
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: DataFrame containing numerical features
        figsize: Figure size
        
    Returns:
        Matplotlib figure
        
    TODO: Implement correlation heatmap
    """
    pass


def plot_cluster_visualization(X_2d: np.ndarray, labels: np.ndarray, title: str = "Cluster Visualization"):
    """
    Visualize clusters in 2D space.
    
    Args:
        X_2d: 2D array of data points
        labels: Cluster labels
        title: Plot title
        
    Returns:
        Matplotlib figure
        
    TODO: Implement scatter plot with cluster colors
    """
    pass


def plot_dimensionality_reduction(
    X_reduced: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "PCA",
    figsize: tuple = (10, 8)
):
    """
    Plot dimensionality reduction results (PCA, t-SNE, UMAP).
    
    Args:
        X_reduced: Reduced dimensional data
        labels: Optional labels for coloring points
        method: Name of the dimensionality reduction method
        figsize: Figure size
        
    Returns:
        Matplotlib figure
        
    TODO: Implement visualization for dimensionality reduction
    """
    pass


def plot_feature_importance(feature_names: List[str], importances: np.ndarray, top_n: int = 20):
    """
    Plot feature importance from tree-based models.
    
    Args:
        feature_names: List of feature names
        importances: Array of feature importances
        top_n: Number of top features to display
        
    Returns:
        Matplotlib figure
        
    TODO: Implement horizontal bar plot of feature importances
    """
    pass


def plot_learning_curve(train_scores: List[float], val_scores: List[float], metric: str = "Accuracy"):
    """
    Plot learning curves showing training and validation scores.
    
    Args:
        train_scores: Training scores over epochs/iterations
        val_scores: Validation scores over epochs/iterations
        metric: Name of the metric being plotted
        
    Returns:
        Matplotlib figure
        
    TODO: Implement learning curve visualization
    """
    pass


def plot_mar_index_distribution(mar_indices: pd.Series, figsize: tuple = (10, 6)):
    """
    Plot distribution of MAR (Multiple Antibiotic Resistance) indices.
    
    Args:
        mar_indices: Series containing MAR index values
        figsize: Figure size
        
    Returns:
        Matplotlib figure
        
    TODO: Implement MAR index distribution plot
    """
    pass


def create_interactive_plot(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None):
    """
    Create interactive plot using Plotly.
    
    Args:
        df: DataFrame containing the data
        x: Column name for x-axis
        y: Column name for y-axis
        color: Optional column name for color coding
        
    Returns:
        Plotly figure
        
    TODO: Implement interactive scatter plot
    """
    pass


# ============================================================================
# Clustering Visualizations
# ============================================================================

def plot_elbow_curve(k_range: List[int], inertias: List[float], 
                     figsize: tuple = (10, 6), save_path: Optional[str] = None):
    """
    Plot elbow curve for optimal K selection.
    
    Args:
        k_range: Range of K values
        inertias: Inertia values for each K
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(k_range, inertias, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    ax.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_silhouette_scores(k_range: List[int], scores: List[float], 
                           figsize: tuple = (10, 6), save_path: Optional[str] = None):
    """
    Plot silhouette scores for different K values.
    
    Args:
        k_range: Range of K values
        scores: Silhouette scores for each K
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(k_range, scores, marker='o', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Mark the maximum
    max_idx = np.argmax(scores)
    ax.axvline(k_range[max_idx], color='red', linestyle='--', alpha=0.7, 
               label=f'Max at K={k_range[max_idx]}')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_dendrogram(X: np.ndarray, method: str = 'ward', 
                    truncate_mode: Optional[str] = 'level', p: int = 5,
                    figsize: tuple = (12, 6), save_path: Optional[str] = None):
    """
    Plot hierarchical clustering dendrogram.
    
    Args:
        X: Feature array
        method: Linkage method ('ward', 'complete', 'average', 'single')
        truncate_mode: Truncation mode for large datasets
        p: Truncation parameter
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Compute linkage
    Z = linkage(X, method=method)
    
    # Plot dendrogram
    dendrogram(Z, ax=ax, truncate_mode=truncate_mode, p=p)
    ax.set_xlabel('Sample Index or (Cluster Size)', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title(f'Hierarchical Clustering Dendrogram ({method.capitalize()} Linkage)', 
                 fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cluster_distribution(labels: np.ndarray, figsize: tuple = (10, 6),
                              save_path: Optional[str] = None):
    """
    Plot distribution of cluster sizes.
    
    Args:
        labels: Cluster labels
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Handle DBSCAN outliers (-1 label)
    cluster_names = []
    for label in unique_labels:
        if label == -1:
            cluster_names.append('Outliers')
        else:
            cluster_names.append(f'Cluster {label}')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    ax.bar(cluster_names, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (name, count) in enumerate(zip(cluster_names, counts)):
        ax.text(i, count, str(count), ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# Dimensionality Reduction Visualizations
# ============================================================================

def plot_2d_scatter(X_reduced: np.ndarray, labels: Optional[np.ndarray] = None, 
                    title: str = '2D Visualization', palette: str = 'viridis',
                    figsize: tuple = (10, 8), save_path: Optional[str] = None,
                    alpha: float = 0.7):
    """
    Plot 2D scatter plot with optional coloring.
    
    Args:
        X_reduced: 2D array of reduced dimensions
        labels: Optional labels for coloring points
        title: Plot title
        palette: Color palette
        figsize: Figure size
        save_path: Optional path to save figure
        alpha: Point transparency
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is not None:
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)
        
        # Use discrete colors if labels are categorical
        if isinstance(labels[0], (str, np.str_)):
            colors = plt.cm.get_cmap(palette)(np.linspace(0, 1, n_labels))
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                          label=label, alpha=alpha, s=50, color=colors[i], edgecolors='black', linewidth=0.5)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        else:
            scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                               c=labels, cmap=palette, alpha=alpha, s=50, 
                               edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, ax=ax, label='Label')
    else:
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=alpha, s=50,
                  edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_3d_scatter(X_reduced: np.ndarray, labels: Optional[np.ndarray] = None, 
                    title: str = '3D Visualization', save_path: Optional[str] = None):
    """
    Plot 3D interactive scatter plot using Plotly.
    
    Args:
        X_reduced: 3D array of reduced dimensions
        labels: Optional labels for coloring points
        title: Plot title
        save_path: Optional path to save figure (as HTML)
        
    Returns:
        Plotly figure
    """
    if X_reduced.shape[1] < 3:
        raise ValueError("X_reduced must have at least 3 dimensions for 3D plotting")
    
    df = pd.DataFrame({
        'x': X_reduced[:, 0],
        'y': X_reduced[:, 1],
        'z': X_reduced[:, 2]
    })
    
    if labels is not None:
        df['label'] = labels
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='label',
                           title=title, opacity=0.7)
    else:
        fig = px.scatter_3d(df, x='x', y='y', z='z',
                           title=title, opacity=0.7)
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3'
        )
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_pca_variance(pca_model, figsize: tuple = (10, 6), 
                     save_path: Optional[str] = None):
    """
    Plot cumulative variance explained by PCA components.
    
    Args:
        pca_model: Fitted PCA model
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    explained_var = pca_model.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    x = range(1, len(explained_var) + 1)
    
    # Plot individual variance
    ax.bar(x, explained_var, alpha=0.6, label='Individual', color='steelblue')
    
    # Plot cumulative variance
    ax.plot(x, cumulative_var, marker='o', color='red', linewidth=2, 
            label='Cumulative', markersize=6)
    
    # Add horizontal lines at common thresholds
    ax.axhline(0.8, color='green', linestyle='--', alpha=0.5, label='80% variance')
    ax.axhline(0.9, color='orange', linestyle='--', alpha=0.5, label='90% variance')
    
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Variance Explained', fontsize=12)
    ax.set_title('PCA Variance Explained', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_pca_loadings_heatmap(loadings: pd.DataFrame, feature_names: List[str], 
                              n_components: int = 5, figsize: tuple = (12, 10),
                              save_path: Optional[str] = None):
    """
    Plot heatmap of PCA feature loadings.
    
    Args:
        loadings: DataFrame with PCA loadings
        feature_names: List of feature names
        n_components: Number of components to show
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Select top n_components
    loadings_subset = loadings.iloc[:, :n_components]
    
    # Plot heatmap
    sns.heatmap(loadings_subset, annot=False, cmap='RdBu_r', center=0,
               cbar_kws={'label': 'Loading'}, ax=ax, linewidths=0.5)
    
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('PCA Feature Loadings Heatmap', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ============================================================================
# Multi-panel Visualizations
# ============================================================================

def plot_clustering_comparison(X: np.ndarray, labels_dict: Dict[str, np.ndarray],
                               figsize: tuple = (18, 6), save_path: Optional[str] = None):
    """
    Compare multiple clustering methods side by side.
    
    Args:
        X: 2D reduced feature array
        labels_dict: Dictionary mapping method names to cluster labels
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_methods = len(labels_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    
    if n_methods == 1:
        axes = [axes]
    
    for ax, (method_name, labels) in zip(axes, labels_dict.items()):
        unique_labels = np.unique(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            label_name = 'Outliers' if label == -1 else f'Cluster {label}'
            ax.scatter(X[mask, 0], X[mask, 1], label=label_name,
                      alpha=0.7, s=50, color=colors[i], edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Dimension 1', fontsize=10)
        ax.set_ylabel('Dimension 2', fontsize=10)
        ax.set_title(method_name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_reduction_comparison(X_pca: np.ndarray, X_tsne: np.ndarray, 
                              X_umap: np.ndarray, labels: Optional[np.ndarray] = None,
                              figsize: tuple = (18, 6), save_path: Optional[str] = None):
    """
    Compare PCA, t-SNE, and UMAP side by side.
    
    Args:
        X_pca: PCA reduced data
        X_tsne: t-SNE reduced data
        X_umap: UMAP reduced data
        labels: Optional labels for coloring
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    methods = [
        ('PCA', X_pca),
        ('t-SNE', X_tsne),
        ('UMAP', X_umap)
    ]
    
    for ax, (method_name, X_reduced) in zip(axes, methods):
        if labels is not None:
            unique_labels = np.unique(labels)
            
            # Check if labels are categorical
            if isinstance(labels[0], (str, np.str_)):
                colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                             label=label, alpha=0.7, s=50, color=colors[i],
                             edgecolors='black', linewidth=0.5)
                ax.legend(fontsize=8, loc='best')
            else:
                scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                   c=labels, cmap='viridis', alpha=0.7, s=50,
                                   edgecolors='black', linewidth=0.5)
                plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.7, s=50,
                      edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Dimension 1', fontsize=10)
        ax.set_ylabel('Dimension 2', fontsize=10)
        ax.set_title(method_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
