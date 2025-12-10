"""
MDR Prediction Page

Interface for predicting Multi-Drug Resistance (MDR) status from resistance profiles.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config
from utils import (
    load_model, load_feature_names, preprocess_input, predict_mar,
    get_prediction_confidence, calculate_mar_index, parse_resistance_input
)
from components import (
    display_page_header, create_sidebar_info, resistance_input_form,
    file_uploader_component, example_data_selector, display_prediction_result,
    show_model_not_found_alert
)

# Page configuration
st.set_page_config(
    page_title=f"{config.PAGE_TITLE} - MDR Prediction",
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state=config.INITIAL_SIDEBAR_STATE
)

# Page header
display_page_header(
    "MDR/MAR Prediction",
    "Predict Multi-Drug Resistance status from antibiotic resistance profiles",
    "🦠"
)

# Load model and feature names
mar_model = load_model(config.MAR_MODEL_PATH)
feature_names = load_feature_names(config.FEATURE_NAMES_PATH)

# Check if model is available
if mar_model is None:
    show_model_not_found_alert()
    st.info("You can still explore the interface by entering data below.")

# Main content
st.markdown("""
### About MDR Prediction

This tool predicts whether a bacterial isolate is **Multi-Drug Resistant (MDR)** based on 
its antibiotic resistance profile. The model analyzes resistance patterns across multiple 
antibiotics and calculates:

- **MAR Index**: Multiple Antibiotic Resistance index (ratio of resistant antibiotics)
- **MDR Status**: Classification as High MAR (MDR) or Low MAR
- **Confidence Score**: How certain the model is about its prediction

**Threshold**: Isolates with MAR index > {threshold:.1%} are classified as High MAR (MDR).
""".format(threshold=config.MAR_THRESHOLD))

st.markdown("---")

# Input method selection
st.subheader("📝 Input Method")
input_method = st.radio(
    "Choose how to input resistance data:",
    ["Manual Entry", "Upload CSV", "Use Example Data"],
    horizontal=True
)

resistance_profile = None

if input_method == "Manual Entry":
    st.markdown("---")
    resistance_profile = resistance_input_form(config.ANTIBIOTICS, use_categories=True)
    
elif input_method == "Upload CSV":
    st.markdown("---")
    uploaded_df = file_uploader_component()
    
    if uploaded_df is not None:
        st.info("Using the first row of uploaded data for prediction.")
        # Use first row
        if len(uploaded_df) > 0:
            resistance_profile = uploaded_df.iloc[0].to_dict()
        
elif input_method == "Use Example Data":
    st.markdown("---")
    resistance_profile = example_data_selector()

# Prediction button
st.markdown("---")

if resistance_profile is not None:
    # Display current profile summary
    with st.expander("📋 Current Resistance Profile", expanded=False):
        # Count resistances
        num_resistant = sum(1 for v in resistance_profile.values() if v == 2)
        num_intermediate = sum(1 for v in resistance_profile.values() if v == 1)
        num_susceptible = sum(1 for v in resistance_profile.values() if v == 0)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Resistant (R)", num_resistant)
        col2.metric("Intermediate (I)", num_intermediate)
        col3.metric("Susceptible (S)", num_susceptible)
        
        # Show profile as table
        profile_df = pd.DataFrame([
            {
                'Antibiotic': k.replace('_', ' ').title(),
                'Status': config.RESISTANCE_LABELS[v]
            }
            for k, v in resistance_profile.items()
        ])
        st.dataframe(profile_df, use_container_width=True, hide_index=True)
    
    # Predict button
    if st.button("🎯 Predict MDR Status", type="primary", use_container_width=True):
        if mar_model is None:
            st.error("⚠️ Model not available. Please train the model first.")
        else:
            with st.spinner("Analyzing resistance profile..."):
                try:
                    # Prepare input
                    encoded_profile = parse_resistance_input(resistance_profile)
                    X = preprocess_input(encoded_profile, feature_names)
                    
                    # Make prediction
                    predictions, probabilities = predict_mar(mar_model, X)
                    prediction = predictions[0]
                    confidence = get_prediction_confidence(probabilities[0])
                    
                    # Calculate MAR index
                    mar_idx = calculate_mar_index(resistance_profile)
                    
                    # Display results
                    st.markdown("---")
                    display_prediction_result(
                        prediction=prediction,
                        confidence=confidence,
                        is_mdr=True,
                        mar_index=mar_idx
                    )
                    
                    # Additional visualizations
                    st.markdown("---")
                    st.subheader("📊 Detailed Analysis")
                    
                    # Probability distribution
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Class Probabilities:**")
                        prob_df = pd.DataFrame({
                            'Class': ['Low MAR', 'High MAR'],
                            'Probability': probabilities[0]
                        })
                        st.dataframe(prob_df, hide_index=True, use_container_width=True)
                    
                    with col2:
                        st.write("**MAR Index Details:**")
                        st.metric("Calculated MAR", f"{mar_idx:.3f}")
                        st.metric("Threshold", f"{config.MAR_THRESHOLD:.3f}")
                        st.metric("Classification", "High MAR" if mar_idx > config.MAR_THRESHOLD else "Low MAR")
                    
                    # Resistance breakdown by category
                    st.markdown("---")
                    st.subheader("📌 Resistance Breakdown by Antibiotic Class")
                    
                    for category, antibiotics in config.ANTIBIOTIC_CATEGORIES.items():
                        category_resistant = sum(
                            1 for ab in antibiotics 
                            if ab in resistance_profile and resistance_profile[ab] == 2
                        )
                        category_total = len([ab for ab in antibiotics if ab in resistance_profile])
                        
                        if category_total > 0:
                            category_mar = category_resistant / category_total
                            st.write(f"**{category}**: {category_resistant}/{category_total} resistant ({category_mar:.1%})")
                            st.progress(float(category_mar))
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.exception(e)

else:
    st.info("👆 Please select an input method and provide resistance data to make a prediction.")

# Information section
st.markdown("---")
st.markdown("""
### 💡 How to Use

1. **Choose an input method** from the options above
2. **Provide resistance data** using one of three methods:
   - **Manual Entry**: Select S/I/R for each antibiotic
   - **Upload CSV**: Upload a file with resistance data
   - **Example Data**: Use a pre-defined profile
3. **Review your profile** in the summary section
4. **Click "Predict"** to get MDR status and confidence
5. **Interpret results** using the guidance provided

### 📖 Understanding Results

- **High MAR (MDR)**: Isolate is resistant to multiple antibiotics (MAR > 20%)
- **Low MAR**: Isolate has limited resistance
- **Confidence**: How sure the model is (High > 80%, Medium 60-80%, Low < 60%)
- **MAR Index**: Proportion of antibiotics the isolate is resistant to

### ⚠️ Clinical Considerations

- Use predictions to inform, not replace, clinical judgment
- Validate with laboratory testing when possible
- Consider local resistance patterns and treatment guidelines
- Consult infectious disease specialists for complex cases
""")

# Sidebar
create_sidebar_info()
