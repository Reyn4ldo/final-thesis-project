# Phase 1: Unsupervised Pattern Recognition

This directory contains the complete implementation of Phase 1: Unsupervised Pattern Recognition for the AMR (Antimicrobial Resistance) Pattern Recognition thesis project.

## Overview

Phase 1 focuses on discovering hidden patterns in antimicrobial resistance data using unsupervised learning techniques:

1. **Clustering Analysis** - Identify groups of isolates with similar resistance patterns
2. **Dimensionality Reduction** - Visualize high-dimensional resistance data in 2D/3D space
3. **Association Rule Mining** - Discover co-resistance patterns between antibiotics

## Implementation

### Core Modules

#### `src/models/unsupervised.py`
Contains clustering and dimensionality reduction functions:
- **Clustering**: `perform_kmeans()`, `perform_hierarchical()`, `perform_dbscan()`
- **Cluster Analysis**: `find_optimal_clusters()`, `get_cluster_summary()`
- **Dimensionality Reduction**: `perform_pca()`, `perform_tsne()`, `perform_umap()`
- **PCA Analysis**: `get_pca_loadings()`

#### `src/models/association_rules.py`
Contains association rule mining functions:
- **Data Preparation**: `prepare_binary_resistance()`
- **Mining**: `mine_frequent_itemsets()`, `generate_association_rules()`
- **Analysis**: `filter_top_rules()`, `interpret_rules()`, `get_resistance_frequency()`

#### `src/visualization/plots.py`
Contains visualization functions:
- **Clustering**: `plot_elbow_curve()`, `plot_silhouette_scores()`, `plot_dendrogram()`, `plot_cluster_distribution()`
- **Dimensionality Reduction**: `plot_2d_scatter()`, `plot_3d_scatter()`, `plot_pca_variance()`, `plot_pca_loadings_heatmap()`
- **Comparisons**: `plot_clustering_comparison()`, `plot_reduction_comparison()`

### Jupyter Notebook

**`notebooks/02_unsupervised_learning.ipynb`**

Complete analysis workflow including:
1. Data loading and preparation
2. Optimal cluster determination (elbow method, silhouette scores)
3. K-Means, Hierarchical, and DBSCAN clustering
4. PCA, t-SNE, and UMAP dimensionality reduction
5. Association rule mining for co-resistance patterns
6. Cluster characterization and interpretation
7. Comprehensive visualizations
8. Results export

## Usage

### Running the Notebook

```bash
cd notebooks/
jupyter notebook 02_unsupervised_learning.ipynb
```

Execute all cells to:
- Perform complete unsupervised analysis
- Generate all plots in `reports/figures/`
- Save results to `reports/results/`

### Using the Functions Programmatically

```python
from src.models.unsupervised import *
from src.models.association_rules import *
from src.visualization.plots import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('../data/processed/cleaned_data.csv')
resistance_cols = [col for col in df.columns if col.endswith('_encoded')]
X = df[resistance_cols].fillna(-1)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
k_range, inertias, sil_scores = find_optimal_clusters(X_scaled, max_k=10)
optimal_k = k_range[np.argmax(sil_scores)]
labels, model = perform_kmeans(X_scaled, n_clusters=optimal_k)

# Dimensionality Reduction
X_pca, pca_model = perform_pca(X_scaled, n_components=2)
X_tsne = perform_tsne(X_scaled, n_components=2, perplexity=30)
X_umap = perform_umap(X_scaled, n_components=2)

# Association Rules
df_binary = prepare_binary_resistance(df, resistance_cols)
itemsets = mine_frequent_itemsets(df_binary, min_support=0.02)
rules = generate_association_rules(itemsets, min_confidence=0.6, min_lift=1.0)
top_rules = filter_top_rules(rules, n=20, sort_by='lift')
```

## Outputs

### Results Files (`reports/results/`)

- `cluster_labels.csv` - Cluster assignments for each isolate
- `cluster_summary.xlsx` - Summary statistics per cluster (multiple sheets)
- `pca_embeddings.csv` - 2D PCA coordinates
- `tsne_embeddings.csv` - 2D t-SNE coordinates
- `umap_embeddings.csv` - 2D UMAP coordinates
- `pca_loadings.csv` - Feature contributions to principal components
- `association_rules.csv` - Top co-resistance patterns

### Figures (`reports/figures/`)

- `elbow_plot.png` - Elbow curve for optimal K selection
- `silhouette_plot.png` - Silhouette scores for different K values
- `dendrogram.png` - Hierarchical clustering dendrogram
- `kmeans_clusters_pca.png` - K-Means clusters visualized in PCA space
- `pca_variance_explained.png` - Cumulative variance explained by PCs
- `pca_by_species.png` - PCA colored by bacterial species
- `tsne_by_species.png` - t-SNE colored by bacterial species
- `umap_by_species.png` - UMAP colored by bacterial species
- `pca_loadings_heatmap.png` - Heatmap of feature loadings
- `reduction_comparison.png` - Side-by-side comparison of PCA, t-SNE, UMAP
- `clustering_comparison.png` - Comparison of clustering methods

## Testing

Run the test suite:

```bash
# Test unsupervised learning functions
python -m pytest tests/test_unsupervised.py -v

# Test all modules
python -m pytest tests/ -v
```

All tests should pass (15 tests for unsupervised learning).

## Technical Details

### Dependencies
- **scikit-learn** - Clustering, PCA, t-SNE
- **umap-learn** - UMAP dimensionality reduction
- **mlxtend** - Association rule mining (Apriori algorithm)
- **matplotlib, seaborn, plotly** - Visualizations
- **pandas, numpy** - Data manipulation

### Key Parameters

**Clustering:**
- K-Means: `n_clusters` (determined by elbow/silhouette), `random_state=42`
- Hierarchical: `n_clusters`, `linkage='ward'`
- DBSCAN: `eps` (distance threshold), `min_samples` (neighborhood size)

**Dimensionality Reduction:**
- PCA: `n_components` (number of components to keep)
- t-SNE: `n_components=2`, `perplexity=30` (neighborhood size)
- UMAP: `n_components=2`, `n_neighbors=15`, `min_dist=0.1`

**Association Rules:**
- `min_support=0.02` (2% of samples must have the pattern)
- `min_confidence=0.6` (60% confidence threshold)
- `min_lift=1.0` (association strength)

## Results Interpretation

### Clustering
- **Cluster 0**: Typically low-resistance isolates (low MAR index)
- **Cluster 1**: High-resistance isolates (high MAR index)
- **DBSCAN Outliers**: Isolates with unusual/rare resistance patterns

### Dimensionality Reduction
- **PCA**: Linear projection, interpretable through loadings
- **t-SNE**: Non-linear, preserves local structure, good for visualization
- **UMAP**: Non-linear, preserves both local and global structure

### Association Rules
- **Support**: Frequency of pattern in dataset
- **Confidence**: Likelihood of consequent given antecedent
- **Lift**: Strength of association (>1 = positive association)

## Next Steps

After completing Phase 1, proceed to:
- **Phase 2**: Supervised Learning (MAR prediction, species classification)
- **Phase 3**: Model Comparison and Deployment

## References

- Van der Maaten & Hinton (2008) - t-SNE
- McInnes et al. (2018) - UMAP
- Agrawal & Srikant (1994) - Apriori Algorithm
