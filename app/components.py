"""
Reusable UI Components for the Streamlit Application

Contains input forms, display components, and other reusable UI elements.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import config
from utils import calculate_mar_index, get_confidence_level, create_probability_chart


# ============================================================================
# Input Components
# ============================================================================

def resistance_input_form(antibiotic_list: List[str], 
                         use_categories: bool = True,
                         default_value: int = 0) -> Dict[str, int]:
    """
    Create input form for resistance profile.
    
    Args:
        antibiotic_list: List of antibiotics to include
        use_categories: Whether to organize by antibiotic categories
        default_value: Default resistance value (0=S, 1=I, 2=R)
        
    Returns:
        Dictionary mapping antibiotic names to resistance values
    """
    st.subheader("Enter Resistance Profile")
    st.info(config.HELP_TEXT['resistance_input'])
    
    resistance_profile = {}
    
    if use_categories:
        # Organize by categories
        for category, antibiotics in config.ANTIBIOTIC_CATEGORIES.items():
            with st.expander(f"📌 {category}", expanded=(category == 'Beta-lactams')):
                cols = st.columns(2)
                for idx, antibiotic in enumerate(antibiotics):
                    if antibiotic in antibiotic_list:
                        with cols[idx % 2]:
                            resistance_profile[antibiotic] = st.selectbox(
                                f"{antibiotic.replace('_', ' ').title()}",
                                options=[0, 1, 2],
                                format_func=lambda x: config.RESISTANCE_LABELS[x],
                                key=f"input_{antibiotic}",
                                index=default_value
                            )
    else:
        # Simple list
        cols = st.columns(2)
        for idx, antibiotic in enumerate(antibiotic_list):
            with cols[idx % 2]:
                resistance_profile[antibiotic] = st.selectbox(
                    f"{antibiotic.replace('_', ' ').title()}",
                    options=[0, 1, 2],
                    format_func=lambda x: config.RESISTANCE_LABELS[x],
                    key=f"input_{antibiotic}",
                    index=default_value
                )
    
    return resistance_profile


def file_uploader_component(help_text: str = None) -> Optional[pd.DataFrame]:
    """
    CSV upload component with validation.
    
    Args:
        help_text: Optional help text to display
        
    Returns:
        Uploaded DataFrame or None
    """
    st.subheader("Upload CSV File")
    
    if help_text:
        st.info(help_text)
    else:
        st.info(config.HELP_TEXT['batch_upload'])
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with antibiotic resistance data"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Show preview
            with st.expander("📊 Preview Data", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return None
    
    return None


def example_data_selector(example_profiles: Dict = None) -> Optional[Dict[str, int]]:
    """
    Select from example resistance profiles.
    
    Args:
        example_profiles: Dictionary of example profiles (uses config default if None)
        
    Returns:
        Selected resistance profile or None
    """
    if example_profiles is None:
        example_profiles = config.EXAMPLE_PROFILES
    
    st.subheader("Or Use Example Data")
    
    example_name = st.selectbox(
        "Select Example Profile",
        options=["None"] + list(example_profiles.keys()),
        help="Choose a pre-defined resistance profile"
    )
    
    if example_name != "None":
        example = example_profiles[example_name]
        st.info(f"📝 {example['description']}")
        return example['profile']
    
    return None


# ============================================================================
# Display Components
# ============================================================================

def display_prediction_result(prediction: int, 
                              confidence: float,
                              is_mdr: bool = True,
                              mar_index: Optional[float] = None):
    """
    Display prediction result card for MDR/MAR.
    
    Args:
        prediction: Predicted class (0 or 1)
        confidence: Confidence score (0-1)
        is_mdr: Whether this is an MDR prediction
        mar_index: Optional calculated MAR index
    """
    st.subheader("🎯 Prediction Result")
    
    # Determine prediction label
    if is_mdr:
        prediction_label = "High MAR (Multi-Drug Resistant)" if prediction == 1 else "Low MAR (Not Multi-Drug Resistant)"
        color = "red" if prediction == 1 else "green"
    else:
        prediction_label = f"Class {prediction}"
        color = "blue"
    
    # Get confidence level
    conf_level, conf_color = get_confidence_level(confidence)
    
    # Display result in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Prediction",
            value=prediction_label,
            help="Model prediction for this isolate"
        )
    
    with col2:
        st.metric(
            label="Confidence",
            value=f"{confidence:.1%}",
            delta=conf_level,
            help="Model's confidence in this prediction"
        )
    
    with col3:
        if mar_index is not None:
            st.metric(
                label="MAR Index",
                value=f"{mar_index:.3f}",
                help=config.HELP_TEXT['mar_index']
            )
    
    # Display interpretation
    st.markdown("---")
    st.subheader("📋 Interpretation")
    
    if is_mdr and prediction == 1:
        st.warning("""
        **⚠️ High MAR Detected**
        
        This isolate shows resistance to multiple antibiotics and is classified as multi-drug resistant.
        Consider alternative treatment options and infection control measures.
        """)
    elif is_mdr and prediction == 0:
        st.success("""
        **✅ Low MAR**
        
        This isolate shows limited antibiotic resistance and is not classified as multi-drug resistant.
        Standard treatment protocols may be effective.
        """)
    
    if conf_level == "Low":
        st.info(f"""
        **ℹ️ Low Confidence ({confidence:.1%})**
        
        The model has low confidence in this prediction. Consider:
        - Validating with additional testing
        - Reviewing the resistance profile for unusual patterns
        - Consulting with clinical experts
        """)


def display_species_prediction(prediction: int,
                               probabilities: np.ndarray,
                               species_names: List[str],
                               top_n: int = 3):
    """
    Display species prediction results.
    
    Args:
        prediction: Predicted species index
        probabilities: Probability array
        species_names: List of species names
        top_n: Number of top predictions to show
    """
    st.subheader("🦠 Species Prediction Result")
    
    # Get top predictions
    top_indices = np.argsort(probabilities)[::-1][:top_n]
    
    # Display primary prediction
    primary_species = species_names[prediction]
    primary_prob = probabilities[prediction]
    
    # Get confidence level
    conf_level, conf_color = get_confidence_level(primary_prob)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Predicted Species",
            value=config.SPECIES_DISPLAY_NAMES.get(primary_species, primary_species),
            help="Most likely bacterial species"
        )
    
    with col2:
        st.metric(
            label="Confidence",
            value=f"{primary_prob:.1%}",
            delta=conf_level,
            help="Model's confidence in this prediction"
        )
    
    # Display top predictions
    st.markdown("---")
    st.subheader(f"Top {top_n} Predictions")
    
    for rank, idx in enumerate(top_indices, 1):
        species = species_names[idx]
        prob = probabilities[idx]
        display_name = config.SPECIES_DISPLAY_NAMES.get(species, species)
        
        # Create progress bar
        st.write(f"**{rank}. {display_name}**")
        st.progress(float(prob))
        st.caption(f"Probability: {prob:.1%}")
    
    # Display visualization
    st.markdown("---")
    st.subheader("📊 Probability Distribution")
    fig = create_probability_chart(probabilities, 
                                   [config.SPECIES_DISPLAY_NAMES.get(s, s) for s in species_names])
    st.plotly_chart(fig, use_container_width=True)


def display_model_metrics(metrics_df: pd.DataFrame):
    """
    Display model performance metrics table.
    
    Args:
        metrics_df: DataFrame with model metrics
    """
    st.subheader("📊 Model Performance Comparison")
    
    # Style the dataframe
    styled_df = metrics_df.style.highlight_max(axis=0, props='background-color: lightgreen;')
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Display best model
    if 'accuracy' in metrics_df.columns:
        best_model = metrics_df['accuracy'].idxmax()
        best_accuracy = metrics_df.loc[best_model, 'accuracy']
        
        st.success(f"🏆 Best Model: **{best_model}** (Accuracy: {best_accuracy:.4f})")


def display_feature_importance(importances: np.ndarray, 
                               feature_names: List[str],
                               top_n: int = 20):
    """
    Display feature importance visualization.
    
    Args:
        importances: Feature importance values
        feature_names: Names of features
        top_n: Number of top features to display
    """
    st.subheader("🔍 Feature Importance")
    
    st.write("""
    This chart shows which antibiotics are most important for the model's predictions.
    Higher values indicate greater importance in determining the outcome.
    """)
    
    from utils import create_feature_importance_chart
    fig = create_feature_importance_chart(importances, feature_names, top_n)
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Layout Components
# ============================================================================

def create_sidebar_info():
    """Create standard sidebar information."""
    st.sidebar.markdown("---")
    st.sidebar.info("""
        **AMR Pattern Recognition**  
        Version 1.0.0
        
        Machine Learning-based Antimicrobial  
        Resistance Prediction System
        
        © 2024 Thesis Project
    """)


def display_page_header(title: str, description: str, icon: str = "🦠"):
    """
    Display standard page header.
    
    Args:
        title: Page title
        description: Page description
        icon: Emoji icon
    """
    st.markdown(f"# {icon} {title}")
    st.markdown(f"*{description}*")
    st.markdown("---")


def create_metric_card(label: str, value: str, delta: Optional[str] = None, 
                      help_text: Optional[str] = None):
    """
    Create a styled metric card.
    
    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta value
        help_text: Optional help text
    """
    st.metric(label=label, value=value, delta=delta, help=help_text)


# ============================================================================
# Alert Components
# ============================================================================

def show_model_not_found_alert():
    """Display alert when models are not found."""
    st.error("""
        ⚠️ **Models Not Found**
        
        The trained models could not be found in the `models/` directory.
        
        **To use this application:**
        1. Run the training notebooks (Phase 0-2) to train models
        2. Ensure models are saved to the `models/` directory
        3. Model files should be named:
           - `best_model_mar.pkl` for MDR prediction
           - `best_model_species.pkl` for species prediction
        
        **For testing:** You can still explore the interface and upload data.
    """)


def show_data_validation_error(error_message: str):
    """
    Display data validation error.
    
    Args:
        error_message: Error message to display
    """
    st.error(f"""
        ❌ **Data Validation Error**
        
        {error_message}
        
        Please check your input data and try again.
    """)


def show_success_message(message: str):
    """
    Display success message.
    
    Args:
        message: Success message
    """
    st.success(f"✅ {message}")
