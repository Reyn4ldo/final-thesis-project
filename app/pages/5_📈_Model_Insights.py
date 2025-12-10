"""
Model Insights Page

View model performance metrics, feature importances, and comparisons.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
import config
from utils import load_model, load_feature_names, create_feature_importance_chart
from components import display_page_header, create_sidebar_info, display_model_metrics

# Page configuration
st.set_page_config(
    page_title=f"{config.PAGE_TITLE} - Model Insights",
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state=config.INITIAL_SIDEBAR_STATE
)

# Page header
display_page_header(
    "Model Insights",
    "Explore model performance, feature importances, and predictions",
    "📈"
)

# Load models and feature names
mar_model = load_model(config.MAR_MODEL_PATH)
species_model = load_model(config.SPECIES_MODEL_PATH)
feature_names = load_feature_names(config.FEATURE_NAMES_PATH)

# Main content
st.markdown("""
### About Model Insights

This page provides detailed information about the machine learning models used for predictions,
including:

- **Model Performance**: Accuracy, precision, recall, F1-score
- **Feature Importance**: Which antibiotics are most predictive
- **Model Comparison**: How different algorithms perform
- **Confusion Matrices**: Where models make mistakes
""")

st.markdown("---")

# Model selection
st.subheader("🎯 Select Model")

model_choice = st.radio(
    "Choose which model to analyze:",
    ["MDR Prediction Model", "Species Prediction Model"],
    horizontal=True
)

selected_model = mar_model if model_choice == "MDR Prediction Model" else species_model
model_available = selected_model is not None

if not model_available:
    st.warning(f"⚠️ {model_choice} is not available. Please train the model first.")
    st.info("Model performance metrics will be displayed here once models are trained.")
else:
    st.success(f"✅ {model_choice} loaded successfully")

st.markdown("---")

# Feature Importance Section
st.subheader("🔍 Feature Importance Analysis")

if model_available and hasattr(selected_model, 'feature_importances_'):
    st.write("""
    Feature importance shows which antibiotics contribute most to the model's predictions.
    Higher values indicate greater importance in determining the outcome.
    """)
    
    # Get feature importances
    importances = selected_model.feature_importances_
    
    # Display top N selector
    top_n = st.slider("Number of top features to display:", min_value=5, max_value=23, value=15)
    
    # Create and display chart
    fig = create_feature_importance_chart(importances, feature_names, top_n)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show table of importances
    with st.expander("📋 View All Feature Importances", expanded=False):
        importance_df = pd.DataFrame({
            'Antibiotic': [f.replace('_encoded', '').replace('_', ' ').title() for f in feature_names],
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        st.dataframe(importance_df, hide_index=True, use_container_width=True)
    
    # Insights
    st.markdown("---")
    st.subheader("💡 Key Insights")
    
    # Get top 5 features
    top_5_idx = np.argsort(importances)[::-1][:5]
    top_5_features = [feature_names[i].replace('_encoded', '').replace('_', ' ').title() 
                     for i in top_5_idx]
    
    st.write(f"""
    **Most Important Antibiotics for {model_choice}:**
    
    The top 5 antibiotics that drive this model's predictions are:
    """)
    
    for i, feature in enumerate(top_5_features, 1):
        importance_val = importances[top_5_idx[i-1]]
        st.write(f"{i}. **{feature}** (importance: {importance_val:.4f})")
    
    st.info("""
    **Clinical Interpretation:**
    
    These antibiotics are most discriminative for predicting the target variable. This could be due to:
    - Strong association with resistance mechanisms
    - High variability in resistance patterns
    - Correlation with other resistance markers
    - Species-specific resistance profiles
    """)

elif model_available:
    st.info("Feature importance is not available for this model type.")
else:
    st.info("Load a model to view feature importance analysis.")

st.markdown("---")

# Model Performance Metrics
st.subheader("📊 Model Performance Metrics")

st.write("""
Model performance metrics provide quantitative measures of prediction accuracy and reliability.
These metrics are calculated on a held-out test set.
""")

# Mock data (would be replaced with actual metrics from saved results)
if model_available:
    st.info("""
    **Note**: To display actual performance metrics, run the model training notebooks which will
    save metrics to the `reports/results/` directory.
    
    The following are example metrics for demonstration purposes:
    """)
    
    # Example metrics table
    if model_choice == "MDR Prediction Model":
        metrics_data = {
            'Model': ['Random Forest', 'XGBoost', 'Logistic Regression', 'SVM', 'KNN', 'Naive Bayes'],
            'Accuracy': [0.8542, 0.8625, 0.8234, 0.8456, 0.8123, 0.7892],
            'Precision': [0.8421, 0.8598, 0.8156, 0.8334, 0.8045, 0.7756],
            'Recall': [0.8667, 0.8734, 0.8389, 0.8567, 0.8234, 0.7989],
            'F1-Score': [0.8542, 0.8665, 0.8271, 0.8449, 0.8138, 0.7871],
            'ROC-AUC': [0.9123, 0.9234, 0.8876, 0.9045, 0.8756, 0.8567]
        }
    else:
        metrics_data = {
            'Model': ['Random Forest', 'XGBoost', 'Logistic Regression', 'SVM', 'KNN', 'Naive Bayes'],
            'Accuracy': [0.7856, 0.8012, 0.7645, 0.7734, 0.7456, 0.7123],
            'Precision': [0.7734, 0.7923, 0.7523, 0.7612, 0.7334, 0.7001],
            'Recall': [0.7889, 0.8045, 0.7689, 0.7778, 0.7501, 0.7189],
            'F1-Score': [0.7810, 0.7983, 0.7605, 0.7694, 0.7417, 0.7094],
        }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    display_model_metrics(metrics_df)
    
    # Metric explanations
    with st.expander("ℹ️ Understanding Metrics", expanded=False):
        st.markdown("""
        **Accuracy**: Percentage of correct predictions overall
        - Good for balanced datasets
        - Range: 0-1 (higher is better)
        
        **Precision**: Of all positive predictions, how many were correct?
        - Important when false positives are costly
        - Range: 0-1 (higher is better)
        
        **Recall**: Of all actual positives, how many did we find?
        - Important when false negatives are costly
        - Range: 0-1 (higher is better)
        
        **F1-Score**: Harmonic mean of precision and recall
        - Balances precision and recall
        - Range: 0-1 (higher is better)
        
        **ROC-AUC**: Area under the ROC curve
        - Measures discrimination ability
        - Range: 0.5-1.0 (higher is better, 0.5 = random)
        """)
    
    # Visualization of metrics comparison
    st.markdown("---")
    st.subheader("📈 Performance Comparison")
    
    import plotly.graph_objects as go
    
    metrics_to_plot = st.multiselect(
        "Select metrics to compare:",
        options=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'] if 'ROC-AUC' in metrics_df.columns else ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        default=['Accuracy', 'F1-Score']
    )
    
    if metrics_to_plot:
        fig = go.Figure()
        
        for metric in metrics_to_plot:
            if metric in metrics_df.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=metrics_df['Model'],
                    y=metrics_df[metric],
                    text=metrics_df[metric].round(4),
                    textposition='auto',
                ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=500,
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Load a model to view performance metrics.")

st.markdown("---")

# Model Information
st.subheader("ℹ️ Model Information")

if model_available:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Details:**")
        st.write(f"- Model Type: {type(selected_model).__name__}")
        st.write(f"- Number of Features: {len(feature_names)}")
        
        if hasattr(selected_model, 'n_classes_'):
            st.write(f"- Number of Classes: {selected_model.n_classes_}")
        
        if hasattr(selected_model, 'n_estimators'):
            st.write(f"- Number of Estimators: {selected_model.n_estimators}")
    
    with col2:
        st.write("**Training Information:**")
        st.write(f"- Features Used: {len(feature_names)} antibiotics")
        st.write(f"- Model File: {config.MAR_MODEL_PATH.name if model_choice == 'MDR Prediction Model' else config.SPECIES_MODEL_PATH.name}")
        
    # Model parameters
    with st.expander("🔧 Model Parameters", expanded=False):
        st.write("**Model Hyperparameters:**")
        
        if hasattr(selected_model, 'get_params'):
            params = selected_model.get_params()
            params_df = pd.DataFrame([
                {'Parameter': k, 'Value': str(v)}
                for k, v in params.items()
            ])
            st.dataframe(params_df, hide_index=True, use_container_width=True)

st.markdown("---")

# Additional Resources
st.markdown("""
### 📚 Additional Resources

**Model Development:**
- Models were trained using supervised learning algorithms
- Training data split: 70% train, 20% validation, 10% test
- Cross-validation was used for hyperparameter tuning

**Performance Considerations:**
- Models are optimized for the specific dataset used
- Performance may vary on different populations
- Regular retraining recommended with new data

**Interpreting Results:**
- Higher feature importance = greater influence on predictions
- Best model is selected based on validation performance
- Multiple metrics considered to avoid overfitting
""")

# Sidebar
create_sidebar_info()
