"""
Home Page - AMR Pattern Recognition System

Project overview, features, and quick start guide.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path to import config and components
sys.path.append(str(Path(__file__).parent.parent))
import config
from components import display_page_header, create_sidebar_info

# Page configuration
st.set_page_config(
    page_title=f"{config.PAGE_TITLE} - Home",
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state=config.INITIAL_SIDEBAR_STATE
)

# Page header
display_page_header(
    "AMR Pattern Recognition System",
    "Machine Learning-based Antimicrobial Resistance Prediction",
    "🏠"
)

# Introduction
st.markdown("""
## Welcome! 👋

This application uses advanced machine learning techniques to analyze and predict antimicrobial 
resistance (AMR) patterns in bacterial isolates. Built as part of a thesis project, it provides 
an interactive interface for healthcare professionals and researchers to:

- 🎯 Predict multi-drug resistance (MDR) status
- 🦠 Identify bacterial species from resistance patterns
- 📊 Analyze resistance profiles in batch
- 📈 Explore model insights and feature importances
- 🗺️ Visualize resistance patterns in 2D space
""")

st.markdown("---")

# Key Features Section
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 Core Features
    
    **MDR Prediction**
    - Upload resistance profiles
    - Get instant MDR predictions
    - View confidence scores
    - Understand MAR indices
    
    **Species Prediction**
    - Predict bacterial species
    - See top 3 candidates
    - Probability distributions
    - Visual confidence indicators
    
    **Batch Processing**
    - Upload CSV files
    - Process multiple isolates
    - Download results
    - Comprehensive reports
    """)

with col2:
    st.markdown("""
    ### 📊 Analytics & Insights
    
    **Model Performance**
    - Compare multiple algorithms
    - View accuracy metrics
    - Confusion matrices
    - ROC curves
    
    **Feature Importance**
    - Identify key antibiotics
    - Understand predictions
    - Clinical insights
    - Visualization tools
    
    **Data Exploration**
    - UMAP embeddings
    - Cluster visualization
    - Pattern discovery
    - Interactive plots
    """)

st.markdown("---")

# Quick Start Guide
st.markdown("""
## 🚀 Quick Start Guide

### For Single Predictions:

1. **Navigate** to the MDR or Species Prediction page using the sidebar
2. **Input** your resistance profile by:
   - Manually selecting S/I/R for each antibiotic, OR
   - Uploading a CSV file with resistance data
3. **Click** the "Predict" button
4. **Review** the results, confidence scores, and interpretations

### For Batch Predictions:

1. **Prepare** a CSV file with your resistance data
   - Each row = one bacterial isolate
   - Columns = antibiotic names
   - Values = 0 (S), 1 (I), or 2 (R)
2. **Go to** the Batch Prediction page
3. **Upload** your CSV file
4. **Select** prediction type (MDR or Species)
5. **Download** results as CSV

### To Explore Insights:

1. **Visit** the Model Insights page to see:
   - Model performance comparisons
   - Feature importance rankings
   - Which antibiotics matter most
2. **Check out** the Data Explorer to:
   - Visualize resistance patterns
   - See cluster structures
   - Explore UMAP projections
""")

st.markdown("---")

# System Requirements
st.markdown("""
## 🔧 System Information

### Models Used
- Random Forest
- XGBoost
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes

### Antibiotic Classes Analyzed
""")

# Display antibiotic categories
for category, antibiotics in config.ANTIBIOTIC_CATEGORIES.items():
    with st.expander(f"📌 {category} ({len(antibiotics)} antibiotics)"):
        st.write(", ".join([ab.replace('_', ' ').title() for ab in antibiotics]))

st.markdown("---")

# Statistics
st.markdown("### 📈 Dataset Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Antibiotics",
        value=len(config.ANTIBIOTICS),
        help="Number of antibiotics analyzed"
    )

with col2:
    st.metric(
        label="Species Classes",
        value=len(config.SPECIES_NAMES),
        help="Number of bacterial species in dataset"
    )

with col3:
    st.metric(
        label="ML Models",
        value=len(config.MODEL_TYPES),
        help="Number of machine learning models compared"
    )

with col4:
    st.metric(
        label="MAR Threshold",
        value=f"{config.MAR_THRESHOLD:.1%}",
        help="Threshold for multi-drug resistance classification"
    )

st.markdown("---")

# Important Notes
st.info("""
### ℹ️ Important Notes

**Model Availability:**
- If you see warnings about missing models, they need to be trained first
- Run the training notebooks (Phase 0-2) to generate model files
- The app will still allow you to explore the interface without trained models

**Data Privacy:**
- This is a demonstration application
- Do not upload sensitive patient data
- All data processing happens locally

**Clinical Use:**
- This tool is for research and educational purposes
- Predictions should be validated by laboratory testing
- Always consult with clinical experts for treatment decisions
""")

st.markdown("---")

# Footer
st.markdown("""
### 📚 About This Project

This application is part of a thesis project on antimicrobial resistance pattern recognition
using machine learning. It demonstrates how ML techniques can assist in:

- Early detection of multi-drug resistant bacteria
- Rapid species identification
- Pattern discovery in resistance data
- Evidence-based treatment planning

### 🔗 Navigation

Use the sidebar to navigate between different pages:
- 🏠 **Home** - This page
- 🦠 **MDR Prediction** - Predict multi-drug resistance
- 🔬 **Species Prediction** - Identify bacterial species
- 📊 **Batch Prediction** - Process multiple isolates
- 📈 **Model Insights** - View model performance
- 🗺️ **Data Explorer** - Visualize patterns

### 📞 Support

For questions or issues, please refer to the documentation or contact the project maintainer.
""")

# Sidebar
create_sidebar_info()
