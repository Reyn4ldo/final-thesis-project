"""
Data Explorer Page

Visualize resistance patterns, clusters, and UMAP embeddings.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config
from utils import load_feature_names, create_umap_plot
from components import display_page_header, create_sidebar_info

# Page configuration
st.set_page_config(
    page_title=f"{config.PAGE_TITLE} - Data Explorer",
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state=config.INITIAL_SIDEBAR_STATE
)

# Page header
display_page_header(
    "Data Explorer",
    "Visualize resistance patterns and cluster structures",
    "🗺️"
)

# Main content
st.markdown("""
### About Data Explorer

This tool helps you visualize the landscape of antimicrobial resistance patterns using
dimensionality reduction techniques. Explore:

- **UMAP Projections**: 2D visualization of high-dimensional resistance data
- **Cluster Structures**: How isolates group based on resistance patterns
- **Species Distribution**: Where different species fall in resistance space
- **New Predictions**: See where your predictions fit in the overall landscape
""")

st.markdown("---")

# Check for available data
try:
    # Try to load preprocessed data
    if config.CLEANED_DATA_PATH.exists():
        st.success("✅ Dataset loaded successfully")
        df = pd.read_csv(config.CLEANED_DATA_PATH)
        data_available = True
    else:
        st.warning("⚠️ Cleaned dataset not found. Using sample data for visualization.")
        data_available = False
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    data_available = False

if data_available:
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Isolates", len(df))
    
    with col2:
        if 'bacterial_species' in df.columns:
            st.metric("Species", df['bacterial_species'].nunique())
    
    with col3:
        if 'High_MAR' in df.columns:
            high_mar_count = df['High_MAR'].sum()
            st.metric("High MAR", f"{high_mar_count} ({high_mar_count/len(df):.1%})")
    
    with col4:
        if 'MAR_index' in df.columns:
            st.metric("Avg MAR Index", f"{df['MAR_index'].mean():.3f}")
    
    st.markdown("---")
    
    # Visualization options
    st.subheader("🗺️ Visualization Options")
    
    # Check if embeddings exist
    umap_path = config.DATA_DIR / "umap_embeddings.csv"
    tsne_path = config.DATA_DIR / "tsne_embeddings.csv"
    pca_path = config.DATA_DIR / "pca_embeddings.csv"
    
    embeddings_available = umap_path.exists() or tsne_path.exists() or pca_path.exists()
    
    if embeddings_available:
        # Select reduction method
        available_methods = []
        if umap_path.exists():
            available_methods.append("UMAP")
        if tsne_path.exists():
            available_methods.append("t-SNE")
        if pca_path.exists():
            available_methods.append("PCA")
        
        reduction_method = st.selectbox(
            "Select dimensionality reduction method:",
            options=available_methods,
            help="Different methods highlight different aspects of the data structure"
        )
        
        # Load selected embeddings
        if reduction_method == "UMAP":
            embeddings_df = pd.read_csv(umap_path)
        elif reduction_method == "t-SNE":
            embeddings_df = pd.read_csv(tsne_path)
        else:
            embeddings_df = pd.read_csv(pca_path)
        
        # Color by options
        color_options = []
        if 'bacterial_species' in df.columns:
            color_options.append("Species")
        if 'High_MAR' in df.columns:
            color_options.append("MAR Status")
        if 'cluster_kmeans' in df.columns:
            color_options.append("K-Means Cluster")
        if 'sample_source' in df.columns:
            color_options.append("Sample Source")
        
        if color_options:
            color_by = st.selectbox(
                "Color points by:",
                options=color_options,
                help="Choose what to highlight in the visualization"
            )
        else:
            color_by = None
        
        # Create visualization
        st.markdown("---")
        st.subheader(f"📈 {reduction_method} Visualization")
        
        # Prepare data for plotting
        plot_df = embeddings_df.copy()
        
        # Add coloring information
        if color_by == "Species" and 'bacterial_species' in df.columns:
            plot_df['color'] = df['bacterial_species'].values[:len(plot_df)]
            color_col = 'color'
        elif color_by == "MAR Status" and 'High_MAR' in df.columns:
            plot_df['color'] = df['High_MAR'].map({0: 'Low MAR', 1: 'High MAR'}).values[:len(plot_df)]
            color_col = 'color'
        elif color_by == "K-Means Cluster" and 'cluster_kmeans' in df.columns:
            plot_df['color'] = df['cluster_kmeans'].astype(str).values[:len(plot_df)]
            color_col = 'color'
        elif color_by == "Sample Source" and 'sample_source' in df.columns:
            plot_df['color'] = df['sample_source'].values[:len(plot_df)]
            color_col = 'color'
        else:
            color_col = None
        
        # Create plot
        import plotly.express as px
        
        # Get column names (they might be named differently)
        x_col = plot_df.columns[0]
        y_col = plot_df.columns[1]
        
        if color_col:
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f"{reduction_method} Projection colored by {color_by}",
                opacity=0.6,
                height=600,
                hover_data={x_col: ':.3f', y_col: ':.3f'}
            )
        else:
            fig = px.scatter(
                plot_df,
                x=x_col,
                y=y_col,
                title=f"{reduction_method} Projection",
                opacity=0.6,
                height=600
            )
        
        fig.update_layout(
            xaxis_title=f"{reduction_method} Dimension 1",
            yaxis_title=f"{reduction_method} Dimension 2",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation
        with st.expander("📖 How to Interpret This Visualization", expanded=False):
            st.markdown(f"""
            **{reduction_method} Projection:**
            
            This plot shows each bacterial isolate as a point in 2D space, where:
            - **Proximity**: Points close together have similar resistance patterns
            - **Clusters**: Groups of points suggest common resistance profiles
            - **Outliers**: Isolated points have unusual resistance patterns
            
            **What the colors represent:**
            - Current coloring: **{color_by}**
            - Different colors show different groups or categories
            - Look for patterns in how colors are distributed
            
            **Clinical Insights:**
            - Species clustering suggests species-specific resistance patterns
            - MAR status separation indicates distinct resistance levels
            - Mixed clusters may represent resistance spread across species
            """)
    
    else:
        st.info("""
        📊 **Dimensionality reduction visualizations not yet available**
        
        To generate UMAP/t-SNE/PCA embeddings:
        1. Run the `02_unsupervised_learning.ipynb` notebook
        2. This will create embedding files in `data/processed/`
        3. Return here to explore the visualizations
        
        These embeddings help visualize high-dimensional resistance data in 2D space.
        """)
    
    st.markdown("---")
    
    # Cluster analysis
    st.subheader("🔍 Cluster Analysis")
    
    if 'cluster_kmeans' in df.columns:
        cluster_col = st.selectbox(
            "Select clustering method:",
            options=[col for col in df.columns if 'cluster' in col.lower()],
            help="Different clustering algorithms may identify different patterns"
        )
        
        # Cluster summary statistics
        cluster_summary = df.groupby(cluster_col).agg({
            'MAR_index': ['mean', 'std', 'min', 'max'],
            'bacterial_species': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Mixed'
        }).round(3)
        
        st.write("**Cluster Summary Statistics:**")
        st.dataframe(cluster_summary, use_container_width=True)
        
        # Cluster distribution
        col1, col2 = st.columns(2)
        
        with col1:
            import plotly.express as px
            cluster_counts = df[cluster_col].value_counts().sort_index()
            fig = px.bar(
                x=cluster_counts.index.astype(str),
                y=cluster_counts.values,
                labels={'x': 'Cluster', 'y': 'Number of Isolates'},
                title='Isolates per Cluster'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'bacterial_species' in df.columns:
                # Species distribution per cluster
                species_cluster = pd.crosstab(df[cluster_col], df['bacterial_species'])
                fig = px.bar(
                    species_cluster,
                    title='Species Distribution per Cluster',
                    labels={'value': 'Count', 'variable': 'Species'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("""
        Cluster labels not found in the dataset. 
        Run the unsupervised learning notebook to generate cluster assignments.
        """)
    
    st.markdown("---")
    
    # Data table explorer
    st.subheader("📋 Data Table Explorer")
    
    with st.expander("View Raw Data", expanded=False):
        # Select columns to display
        display_cols = st.multiselect(
            "Select columns to display:",
            options=df.columns.tolist(),
            default=[col for col in ['bacterial_species', 'MAR_index', 'High_MAR'] if col in df.columns][:5]
        )
        
        if display_cols:
            # Add filters
            if 'bacterial_species' in df.columns:
                species_filter = st.multiselect(
                    "Filter by species:",
                    options=df['bacterial_species'].unique().tolist(),
                    default=None
                )
                
                if species_filter:
                    filtered_df = df[df['bacterial_species'].isin(species_filter)]
                else:
                    filtered_df = df
            else:
                filtered_df = df
            
            st.dataframe(filtered_df[display_cols], use_container_width=True, height=400)
            
            # Download filtered data
            csv = filtered_df[display_cols].to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv,
                file_name="filtered_data.csv",
                mime="text/csv"
            )

else:
    # Show example visualization
    st.info("""
    📊 **Data Explorer requires the cleaned dataset**
    
    To use this feature:
    1. Ensure `data/processed/cleaned_data.csv` exists
    2. Run the preprocessing notebook if needed
    3. Return here to explore visualizations
    """)
    
    # Show example placeholder
    st.subheader("Example Visualization")
    
    st.write("""
    Once data is available, you'll be able to:
    - Explore UMAP/t-SNE/PCA projections
    - View cluster distributions
    - Analyze species patterns
    - Filter and download data
    """)

# Information section
st.markdown("---")
st.markdown("""
### 📖 About Dimensionality Reduction

**UMAP (Uniform Manifold Approximation and Projection):**
- Preserves both local and global structure
- Good for visualizing clusters
- Faster than t-SNE for large datasets

**t-SNE (t-Distributed Stochastic Neighbor Embedding):**
- Excellent for revealing local structure
- Creates tight, well-separated clusters
- Can be computationally intensive

**PCA (Principal Component Analysis):**
- Linear dimensionality reduction
- Preserves global variance
- Fast and interpretable

Each method highlights different aspects of the data structure.
""")

# Sidebar
create_sidebar_info()
