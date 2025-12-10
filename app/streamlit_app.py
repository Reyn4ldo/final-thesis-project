"""
AMR Pattern Recognition - Streamlit Web Application

Main entry point for the multi-page Streamlit application.

This application provides an interactive interface for:
- Predicting multi-drug resistance (MDR) from resistance profiles
- Identifying bacterial species from resistance patterns
- Batch processing of multiple isolates
- Exploring model performance and insights
- Visualizing resistance patterns and clusters
"""

import streamlit as st
from pathlib import Path
import sys

# Add app directory to path for imports
app_dir = Path(__file__).parent
sys.path.append(str(app_dir))

import config
from components import create_sidebar_info

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state=config.INITIAL_SIDEBAR_STATE
)

# Custom CSS for improved styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Main page content
st.markdown('<h1 class="main-header">🦠 AMR Pattern Recognition System</h1>', unsafe_allow_html=True)
st.markdown("### Machine Learning-based Antimicrobial Resistance Prediction")
st.markdown("---")

# Welcome message
st.markdown("""
## Welcome! 👋

This is a comprehensive web application for analyzing and predicting antimicrobial resistance (AMR) 
patterns in bacterial isolates using advanced machine learning techniques.

### 🚀 Getting Started

Use the sidebar to navigate between different pages:

- **🏠 Home** - Project overview and quick start guide
- **🦠 MDR Prediction** - Predict multi-drug resistance status
- **🔬 Species Prediction** - Identify bacterial species
- **📊 Batch Prediction** - Process multiple isolates from CSV
- **📈 Model Insights** - View model performance and feature importance
- **🗺️ Data Explorer** - Visualize resistance patterns and clusters
""")

st.markdown("---")

# Quick overview
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 Key Features
    
    **For Healthcare Professionals:**
    - Quick MDR status predictions
    - Species identification support
    - Batch processing capability
    - Clinical interpretation guidance
    
    **For Researchers:**
    - Model performance metrics
    - Feature importance analysis
    - Pattern visualization
    - Data exploration tools
    """)

with col2:
    st.markdown("""
    ### 📊 What You Can Do
    
    **Single Predictions:**
    1. Input a resistance profile manually
    2. Get instant MDR/species prediction
    3. View confidence scores
    4. Understand the results
    
    **Batch Analysis:**
    1. Upload a CSV file
    2. Process multiple isolates
    3. Download results
    4. Visualize patterns
    """)

st.markdown("---")

# Statistics
st.markdown("### 📈 System Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Antibiotics Analyzed",
        value=len(config.ANTIBIOTICS),
        help="Number of different antibiotics in the resistance profiles"
    )

with col2:
    st.metric(
        label="Species Classes",
        value=len(config.SPECIES_NAMES),
        help="Number of bacterial species the system can identify"
    )

with col3:
    st.metric(
        label="ML Models",
        value=len(config.MODEL_TYPES),
        help="Number of machine learning algorithms compared"
    )

with col4:
    st.metric(
        label="MAR Threshold",
        value=f"{config.MAR_THRESHOLD:.0%}",
        help="Threshold for multi-drug resistance classification"
    )

st.markdown("---")

# Model status check
st.markdown("### 🔧 System Status")

mar_model_exists = config.MAR_MODEL_PATH.exists()
species_model_exists = config.SPECIES_MODEL_PATH.exists()
data_exists = config.CLEANED_DATA_PATH.exists()

col1, col2, col3 = st.columns(3)

with col1:
    if mar_model_exists:
        st.success("✅ MDR Model Available")
    else:
        st.warning("⚠️ MDR Model Not Found")

with col2:
    if species_model_exists:
        st.success("✅ Species Model Available")
    else:
        st.warning("⚠️ Species Model Not Found")

with col3:
    if data_exists:
        st.success("✅ Dataset Available")
    else:
        st.warning("⚠️ Dataset Not Found")

if not (mar_model_exists or species_model_exists):
    st.info("""
    **📝 Note:** Models need to be trained before making predictions.
    
    To train models:
    1. Run the data preprocessing notebook
    2. Run the supervised learning notebook  
    3. Models will be saved to the `models/` directory
    
    You can still explore the interface without trained models!
    """)

st.markdown("---")

# Quick links
st.markdown("""
### 🔗 Quick Links

**Start Predicting:**
- 🦠 [MDR Prediction](#) - Determine if an isolate is multi-drug resistant
- 🔬 [Species Prediction](#) - Identify the bacterial species

**Analyze Data:**
- 📊 [Batch Prediction](#) - Process multiple isolates at once
- 📈 [Model Insights](#) - See which antibiotics matter most
- 🗺️ [Data Explorer](#) - Visualize resistance landscapes

### 📚 Documentation

For detailed information about:
- How the models work
- Interpreting predictions
- Data format requirements
- Clinical considerations

Visit the individual pages using the sidebar navigation.
""")

st.markdown("---")

# Footer information
st.markdown("""
### ℹ️ About

This application is part of a thesis project on antimicrobial resistance pattern recognition.
It demonstrates how machine learning can assist in:

- Early detection of multi-drug resistant bacteria
- Rapid species identification from resistance patterns
- Pattern discovery in AMR surveillance data
- Evidence-based clinical decision support

**Disclaimer:** This tool is for research and educational purposes. Always validate 
predictions with laboratory testing and consult clinical experts for treatment decisions.
""")

# Sidebar
create_sidebar_info()
