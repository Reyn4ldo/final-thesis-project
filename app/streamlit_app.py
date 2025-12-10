"""
AMR Pattern Recognition - Streamlit Web Application

This application provides an interactive interface for:
- Uploading new AMR data
- Visualizing resistance patterns
- Making predictions using trained models
- Exploring model performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append('..')
from src.data.preprocessing import clean_data, encode_categorical_features
from src.features.build_features import calculate_mar_index
from src.models.evaluation import calculate_classification_metrics
from utils import load_model, preprocess_input, create_prediction_plot

# Page configuration
st.set_page_config(
    page_title="AMR Pattern Recognition",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🦠 AMR Pattern Recognition System</h1>', unsafe_allow_html=True)
st.markdown("### Machine Learning-based Antimicrobial Resistance Prediction")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Data Upload", "Make Predictions", "Model Performance", "About"]
)

# Home Page
if page == "Home":
    st.markdown('<h2 class="sub-header">Welcome to the AMR Pattern Recognition System</h2>', 
                unsafe_allow_html=True)
    
    st.write("""
    This application uses machine learning to analyze and predict antimicrobial resistance patterns.
    
    **Key Features:**
    - 📊 Interactive data visualization
    - 🤖 Multiple ML models for prediction
    - 📈 Real-time resistance pattern analysis
    - 🎯 High accuracy predictions
    
    **How to Use:**
    1. Upload your AMR dataset or use sample data
    2. Explore visualizations and patterns
    3. Get predictions from trained models
    4. View model performance metrics
    """)
    
    # Display sample statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Models Available", "6", help="RF, XGBoost, LR, SVM, KNN, NB")
    with col2:
        st.metric("Best Model Accuracy", "TBD", help="To be updated after training")
    with col3:
        st.metric("Features Used", "TBD", help="Antibiotic resistance profiles")

# Data Upload Page
elif page == "Data Upload":
    st.markdown('<h2 class="sub-header">📤 Upload AMR Data</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.success(f"Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10))
        
        # Display basic statistics
        st.subheader("Dataset Statistics")
        st.write(df.describe())
        
        # Data quality checks
        st.subheader("Data Quality")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.warning(f"Found {missing_data.sum()} missing values")
            st.write(missing_data[missing_data > 0])
        else:
            st.success("No missing values found!")

# Predictions Page
elif page == "Make Predictions":
    st.markdown('<h2 class="sub-header">🎯 Make Predictions</h2>', unsafe_allow_html=True)
    
    # TODO: Load trained models
    st.write("**Select a model for prediction:**")
    
    model_choice = st.selectbox(
        "Choose model",
        ["Random Forest", "XGBoost", "Logistic Regression", "SVM", "KNN", "Naive Bayes"]
    )
    
    st.write("""
    **Input Method:**
    """)
    
    input_method = st.radio("", ["Manual Input", "Upload CSV"])
    
    if input_method == "Manual Input":
        st.write("Enter antibiotic resistance data:")
        # TODO: Create form for manual input
        st.info("Manual input form to be implemented")
        
    else:
        uploaded_pred_file = st.file_uploader("Upload data for prediction", type=['csv'])
        if uploaded_pred_file is not None:
            pred_df = pd.read_csv(uploaded_pred_file)
            st.dataframe(pred_df.head())
            
            if st.button("Generate Predictions"):
                # TODO: Make predictions
                st.info("Prediction functionality to be implemented")

# Model Performance Page
elif page == "Model Performance":
    st.markdown('<h2 class="sub-header">📊 Model Performance</h2>', unsafe_allow_html=True)
    
    st.write("**Compare performance across all trained models:**")
    
    # TODO: Load and display model comparison results
    st.info("Model performance metrics will be displayed here after training")
    
    # Placeholder for performance visualization
    # st.plotly_chart(fig, use_container_width=True)

# About Page
elif page == "About":
    st.markdown('<h2 class="sub-header">ℹ️ About This Project</h2>', unsafe_allow_html=True)
    
    st.write("""
    ## AMR Pattern Recognition using Machine Learning
    
    This thesis project develops machine learning models to predict antimicrobial resistance patterns
    and identify critical resistance profiles in bacterial isolates.
    
    ### Methodology
    
    1. **Data Collection**: AMR test results from clinical isolates
    2. **Preprocessing**: Data cleaning, encoding, and feature engineering
    3. **Unsupervised Learning**: Clustering and pattern discovery
    4. **Supervised Learning**: Classification models for prediction
    5. **Evaluation**: Comprehensive model comparison and validation
    6. **Deployment**: Interactive web application
    
    ### Technologies Used
    
    - **Python**: Primary programming language
    - **Scikit-learn**: Machine learning algorithms
    - **XGBoost**: Gradient boosting framework
    - **Streamlit**: Web application framework
    - **Plotly**: Interactive visualizations
    
    ### Author
    
    [Your Name]  
    Final Thesis Project  
    [University Name]  
    [Year]
    
    ### Contact
    
    For questions or feedback, please contact: [your.email@example.com]
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
    **AMR Pattern Recognition v0.1.0**  
    Developed for thesis research  
    © 2024
""")
