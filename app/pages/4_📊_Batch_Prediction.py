"""
Batch Prediction Page

Interface for processing multiple isolates from CSV files.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from io import StringIO

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config
from utils import (
    load_model, load_feature_names, preprocess_input, predict_mar, predict_species,
    get_prediction_confidence, calculate_mar_index, validate_input
)
from components import (
    display_page_header, create_sidebar_info, file_uploader_component,
    show_model_not_found_alert
)

# Page configuration
st.set_page_config(
    page_title=f"{config.PAGE_TITLE} - Batch Prediction",
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state=config.INITIAL_SIDEBAR_STATE
)

# Page header
display_page_header(
    "Batch Prediction",
    "Process multiple bacterial isolates at once",
    "📊"
)

# Load models and feature names
mar_model = load_model(config.MAR_MODEL_PATH)
species_model = load_model(config.SPECIES_MODEL_PATH)
feature_names = load_feature_names(config.FEATURE_NAMES_PATH)

# Check if models are available
models_available = mar_model is not None or species_model is not None

if not models_available:
    show_model_not_found_alert()

# Main content
st.markdown("""
### About Batch Prediction

This tool allows you to process **multiple bacterial isolates** at once by uploading a CSV file.
You can predict:

- **MDR Status**: Multi-drug resistance classification for all isolates
- **Species**: Bacterial species identification for all isolates
- **Both**: Get both predictions in a single run

Results can be downloaded as a CSV file for further analysis.
""")

st.markdown("---")

# CSV Format Instructions
with st.expander("📋 CSV Format Requirements", expanded=False):
    st.markdown("""
    Your CSV file should have the following structure:
    
    - **Rows**: Each row represents one bacterial isolate
    - **Columns**: Each column represents one antibiotic
    - **Values**: Resistance values as 0 (S), 1 (I), or 2 (R)
    - **Optional**: Include an 'isolate_id' column for identification
    
    **Example:**
    
    | isolate_id | ampicillin | gentamicin | ciprofloxacin | ... |
    |------------|------------|------------|---------------|-----|
    | ISO001     | 2          | 0          | 1             | ... |
    | ISO002     | 0          | 0          | 0             | ... |
    | ISO003     | 2          | 2          | 2             | ... |
    
    **Accepted column names:**
    - Antibiotic names (with or without '_encoded' suffix)
    - Values: 0, 1, 2 (or S, I, R will be converted)
    """)
    
    # Show expected antibiotics
    st.write("**Expected antibiotics:**")
    st.write(", ".join([ab.replace('_', ' ').title() for ab in config.ANTIBIOTICS]))

st.markdown("---")

# File upload
uploaded_df = file_uploader_component(
    help_text="Upload a CSV file with resistance data for multiple isolates"
)

if uploaded_df is not None:
    # Data preprocessing
    st.subheader("🔧 Data Preprocessing")
    
    # Check if isolate_id column exists
    has_id = 'isolate_id' in uploaded_df.columns
    
    if not has_id:
        st.info("No 'isolate_id' column found. Creating sequential IDs...")
        uploaded_df.insert(0, 'isolate_id', [f"ISO{i:04d}" for i in range(len(uploaded_df))])
    
    # Convert S/I/R to numeric if present
    for col in uploaded_df.columns:
        if col != 'isolate_id' and uploaded_df[col].dtype == 'object':
            uploaded_df[col] = uploaded_df[col].str.upper().map({'S': 0, 'I': 1, 'R': 2})
    
    st.success(f"✅ Loaded {len(uploaded_df)} isolates")
    
    # Validate data
    is_valid, error_msg = validate_input(uploaded_df.drop(columns=['isolate_id']), feature_names)
    
    if not is_valid:
        st.warning(f"⚠️ Data validation warning: {error_msg}")
        st.info("The system will attempt to proceed with available features.")
    else:
        st.success("✅ Data format validated successfully")
    
    # Prediction type selection
    st.markdown("---")
    st.subheader("🎯 Select Prediction Type")
    
    prediction_type = st.radio(
        "What would you like to predict?",
        ["MDR Status Only", "Species Only", "Both MDR and Species"],
        horizontal=False
    )
    
    # Additional options
    col1, col2 = st.columns(2)
    with col1:
        include_confidence = st.checkbox("Include confidence scores", value=True)
    with col2:
        include_mar_index = st.checkbox("Include MAR index", value=True)
    
    # Predict button
    st.markdown("---")
    
    if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
        # Check model availability
        if prediction_type in ["MDR Status Only", "Both MDR and Species"] and mar_model is None:
            st.error("❌ MDR model not available. Please train the model first.")
        elif prediction_type in ["Species Only", "Both MDR and Species"] and species_model is None:
            st.error("❌ Species model not available. Please train the model first.")
        else:
            with st.spinner(f"Processing {len(uploaded_df)} isolates..."):
                try:
                    # Prepare results dataframe
                    results_df = uploaded_df[['isolate_id']].copy()
                    
                    # Prepare features
                    X = preprocess_input(uploaded_df.drop(columns=['isolate_id']), feature_names)
                    
                    # MDR Prediction
                    if prediction_type in ["MDR Status Only", "Both MDR and Species"]:
                        st.info("Predicting MDR status...")
                        predictions, probabilities = predict_mar(mar_model, X)
                        
                        results_df['MDR_Prediction'] = ['High MAR' if p == 1 else 'Low MAR' 
                                                        for p in predictions]
                        
                        if include_confidence:
                            results_df['MDR_Confidence'] = [get_prediction_confidence(prob) 
                                                           for prob in probabilities]
                        
                        if include_mar_index:
                            # Calculate MAR index for each row
                            mar_indices = []
                            for idx, row in uploaded_df.drop(columns=['isolate_id']).iterrows():
                                mar_indices.append(calculate_mar_index(row))
                            results_df['MAR_Index'] = mar_indices
                    
                    # Species Prediction
                    if prediction_type in ["Species Only", "Both MDR and Species"]:
                        st.info("Predicting species...")
                        predictions, probabilities = predict_species(species_model, X)
                        
                        results_df['Species_Prediction'] = [
                            config.SPECIES_DISPLAY_NAMES.get(config.SPECIES_NAMES[p], config.SPECIES_NAMES[p])
                            for p in predictions
                        ]
                        
                        if include_confidence:
                            results_df['Species_Confidence'] = [get_prediction_confidence(prob) 
                                                               for prob in probabilities]
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("✅ Results")
                    
                    st.success(f"Successfully processed {len(results_df)} isolates!")
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Isolates", len(results_df))
                    
                    if 'MDR_Prediction' in results_df.columns:
                        with col2:
                            num_mdr = (results_df['MDR_Prediction'] == 'High MAR').sum()
                            st.metric("High MAR (MDR)", f"{num_mdr} ({num_mdr/len(results_df):.1%})")
                    
                    if 'Species_Prediction' in results_df.columns:
                        with col3:
                            num_species = results_df['Species_Prediction'].nunique()
                            st.metric("Unique Species", num_species)
                    
                    # Display results table
                    st.markdown("---")
                    st.subheader("📊 Detailed Results")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Download button
                    st.markdown("---")
                    st.subheader("💾 Download Results")
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="batch_prediction_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Visualizations
                    st.markdown("---")
                    st.subheader("📈 Summary Visualizations")
                    
                    if 'MDR_Prediction' in results_df.columns:
                        import plotly.express as px
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # MDR distribution
                            mdr_counts = results_df['MDR_Prediction'].value_counts()
                            fig = px.pie(
                                values=mdr_counts.values,
                                names=mdr_counts.index,
                                title='MDR Status Distribution'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Confidence distribution
                            if 'MDR_Confidence' in results_df.columns:
                                fig = px.histogram(
                                    results_df,
                                    x='MDR_Confidence',
                                    nbins=20,
                                    title='MDR Prediction Confidence Distribution'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                    if 'Species_Prediction' in results_df.columns:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Species distribution
                            species_counts = results_df['Species_Prediction'].value_counts()
                            fig = px.bar(
                                x=species_counts.index,
                                y=species_counts.values,
                                labels={'x': 'Species', 'y': 'Count'},
                                title='Species Distribution'
                            )
                            fig.update_xaxis(tickangle=-45)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Confidence distribution
                            if 'Species_Confidence' in results_df.columns:
                                fig = px.histogram(
                                    results_df,
                                    x='Species_Confidence',
                                    nbins=20,
                                    title='Species Prediction Confidence Distribution'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error during batch prediction: {str(e)}")
                    st.exception(e)

else:
    st.info("👆 Please upload a CSV file to begin batch prediction.")

# Information section
st.markdown("---")
st.markdown("""
### 💡 Tips for Batch Prediction

- **Large files**: For files with > 1000 isolates, consider splitting into smaller batches
- **Missing data**: The system will handle missing values by treating them as susceptible (S)
- **Validation**: Review a sample of predictions manually to ensure quality
- **Performance**: Processing time increases with the number of isolates

### 📊 Output Columns

The results CSV will include:
- **isolate_id**: Unique identifier for each isolate
- **MDR_Prediction**: High MAR or Low MAR (if MDR prediction selected)
- **MDR_Confidence**: Confidence score 0-1 (if selected)
- **MAR_Index**: Calculated MAR index (if selected)
- **Species_Prediction**: Predicted bacterial species (if species prediction selected)
- **Species_Confidence**: Confidence score 0-1 (if selected)
""")

# Sidebar
create_sidebar_info()
