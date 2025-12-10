"""
Species Prediction Page

Interface for predicting bacterial species from resistance patterns.
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
    load_model, load_feature_names, preprocess_input, predict_species,
    parse_resistance_input
)
from components import (
    display_page_header, create_sidebar_info, resistance_input_form,
    file_uploader_component, example_data_selector, display_species_prediction,
    show_model_not_found_alert
)

# Page configuration
st.set_page_config(
    page_title=f"{config.PAGE_TITLE} - Species Prediction",
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state=config.INITIAL_SIDEBAR_STATE
)

# Page header
display_page_header(
    "Species Prediction",
    "Identify bacterial species from antibiotic resistance patterns",
    "🔬"
)

# Load model and feature names
species_model = load_model(config.SPECIES_MODEL_PATH)
feature_names = load_feature_names(config.FEATURE_NAMES_PATH)

# Check if model is available
if species_model is None:
    show_model_not_found_alert()
    st.info("You can still explore the interface by entering data below.")

# Main content
st.markdown("""
### About Species Prediction

This tool predicts the **bacterial species** of an isolate based on its antibiotic resistance 
profile. The model has been trained to recognize resistance patterns characteristic of different 
bacterial species.

**Species Classes:**
""")

# Display species in a nice format
species_cols = st.columns(2)
for idx, species in enumerate(config.SPECIES_NAMES):
    with species_cols[idx % 2]:
        st.write(f"• {config.SPECIES_DISPLAY_NAMES.get(species, species)}")

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
    if st.button("🔬 Predict Species", type="primary", use_container_width=True):
        if species_model is None:
            st.error("⚠️ Model not available. Please train the model first.")
        else:
            with st.spinner("Analyzing resistance profile..."):
                try:
                    # Prepare input
                    encoded_profile = parse_resistance_input(resistance_profile)
                    X = preprocess_input(encoded_profile, feature_names)
                    
                    # Make prediction
                    predictions, probabilities = predict_species(species_model, X)
                    prediction = predictions[0]
                    
                    # Display results
                    st.markdown("---")
                    display_species_prediction(
                        prediction=prediction,
                        probabilities=probabilities[0],
                        species_names=config.SPECIES_NAMES,
                        top_n=5
                    )
                    
                    # Additional information
                    st.markdown("---")
                    st.subheader("📋 Species Information")
                    
                    predicted_species = config.SPECIES_NAMES[prediction]
                    display_name = config.SPECIES_DISPLAY_NAMES.get(predicted_species, predicted_species)
                    
                    # Species-specific information (can be expanded)
                    species_info = {
                        'escherichia_coli': """
                        **Escherichia coli** is a common Gram-negative bacterium found in the gut.
                        Some strains can cause infections, particularly urinary tract infections.
                        Resistance patterns vary widely across strains.
                        """,
                        'klebsiella_pneumoniae_ssp_pneumoniae': """
                        **Klebsiella pneumoniae** is a Gram-negative bacterium that can cause
                        pneumonia, bloodstream infections, and wound infections. Often found in
                        healthcare settings, some strains are extensively drug-resistant.
                        """,
                        'pseudomonas_aeruginosa': """
                        **Pseudomonas aeruginosa** is an opportunistic Gram-negative pathogen
                        known for intrinsic resistance to many antibiotics. Common in hospital
                        infections, especially in immunocompromised patients.
                        """,
                        'enterobacter_cloacae_complex': """
                        **Enterobacter cloacae** complex are Gram-negative bacteria that can
                        cause various infections. Known for developing resistance during treatment,
                        particularly to cephalosporins.
                        """,
                        'enterobacter_aerogenes': """
                        **Enterobacter aerogenes** is a Gram-negative bacterium increasingly
                        recognized as an important healthcare-associated pathogen with growing
                        antimicrobial resistance.
                        """,
                        'salmonella_group': """
                        **Salmonella** species are Gram-negative bacteria that cause gastroenteritis
                        and typhoid fever. Resistance to fluoroquinolones and third-generation
                        cephalosporins is a growing concern.
                        """,
                        'vibrio_cholerae': """
                        **Vibrio cholerae** causes cholera, characterized by severe diarrhea.
                        Treatment typically involves rehydration, though antibiotics may be needed
                        in severe cases.
                        """
                    }
                    
                    if predicted_species in species_info:
                        st.info(species_info[predicted_species])
                    else:
                        st.info(f"Information about {display_name} is not yet available in this version.")
                    
                    # Resistance pattern comparison
                    st.markdown("---")
                    st.subheader("🔍 Resistance Pattern Analysis")
                    
                    st.write("""
                    The model identified this species based on its characteristic resistance pattern.
                    Different species often show distinct patterns due to:
                    
                    - **Intrinsic resistance**: Natural resistance mechanisms
                    - **Acquired resistance**: Resistance gained through mutations or gene transfer
                    - **Species-specific traits**: Unique metabolic or structural features
                    """)
                    
                    # Show which antibiotics were most informative (if model has feature importances)
                    if hasattr(species_model, 'feature_importances_'):
                        st.write("**Key antibiotics for this prediction:**")
                        importances = species_model.feature_importances_
                        top_idx = np.argsort(importances)[::-1][:10]
                        
                        top_antibiotics = [feature_names[i].replace('_encoded', '').replace('_', ' ').title() 
                                          for i in top_idx]
                        top_values = [resistance_profile.get(feature_names[i].replace('_encoded', ''), 0) 
                                     for i in top_idx]
                        
                        importance_df = pd.DataFrame({
                            'Antibiotic': top_antibiotics,
                            'Your Value': [config.RESISTANCE_LABELS[v] for v in top_values],
                            'Importance': [f"{importances[i]:.4f}" for i in top_idx]
                        })
                        st.dataframe(importance_df, hide_index=True, use_container_width=True)
                    
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
4. **Click "Predict Species"** to get species identification
5. **Interpret results** including top candidates and probabilities

### 📖 Understanding Results

- **Predicted Species**: The most likely bacterial species
- **Confidence**: Probability that this is the correct species
- **Top Predictions**: Alternative species that match the profile
- **Probability Distribution**: How confident the model is for each species

### ⚠️ Important Notes

- Resistance-based species prediction is complementary to traditional identification
- Some species may have overlapping resistance patterns
- Always confirm with molecular or biochemical identification when possible
- Low confidence predictions warrant additional testing
""")

# Sidebar
create_sidebar_info()
