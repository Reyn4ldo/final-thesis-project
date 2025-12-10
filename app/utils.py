"""
Utility functions for the Streamlit application

Helper functions for model loading, data processing, and visualization.
"""

import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Union
import streamlit as st

# ============================================================================
# Data Loading Functions
# ============================================================================

@st.cache_resource
def load_model(model_path: Path):
    """
    Load a trained model from file with caching.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model object or None if not found
    """
    try:
        if model_path.exists():
            return joblib.load(model_path)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


@st.cache_data
def load_feature_names(path: Path) -> List[str]:
    """
    Load feature names from JSON file.
    
    Args:
        path: Path to feature names JSON file
        
    Returns:
        List of feature names
    """
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading feature names: {str(e)}")
        return []


@st.cache_data
def load_encoding_mappings(path: Path) -> Dict:
    """
    Load encoding mappings from JSON file.
    
    Args:
        path: Path to encoding mappings JSON file
        
    Returns:
        Dictionary of encoding mappings
    """
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading encoding mappings: {str(e)}")
        return {}


# ============================================================================
# Data Processing Functions
# ============================================================================

def preprocess_input(input_data: Union[pd.DataFrame, Dict], feature_names: List[str]) -> np.ndarray:
    """
    Preprocess input data for prediction.
    
    Args:
        input_data: Input as DataFrame or dictionary
        feature_names: Expected feature names in correct order
        
    Returns:
        Preprocessed numpy array ready for prediction
    """
    if isinstance(input_data, dict):
        # Convert dictionary to DataFrame
        input_data = pd.DataFrame([input_data])
    
    # Ensure all expected features are present
    for feature in feature_names:
        if feature not in input_data.columns:
            # Extract antibiotic name without '_encoded' suffix
            antibiotic_name = feature.replace('_encoded', '')
            if antibiotic_name in input_data.columns:
                input_data[feature] = input_data[antibiotic_name]
            else:
                input_data[feature] = 0  # Default to susceptible if missing
    
    # Select and order features correctly
    X = input_data[feature_names].values
    
    return X


def parse_resistance_input(input_dict: Dict[str, int]) -> Dict[str, int]:
    """
    Convert user input to feature vector with encoded suffix.
    
    Args:
        input_dict: Dictionary mapping antibiotic names to resistance values
        
    Returns:
        Dictionary with encoded feature names
    """
    encoded_dict = {}
    for antibiotic, value in input_dict.items():
        # Add '_encoded' suffix to match feature names
        encoded_key = f"{antibiotic}_encoded"
        encoded_dict[encoded_key] = value
    
    return encoded_dict


def validate_input(input_data: pd.DataFrame, expected_features: List[str]) -> Tuple[bool, str]:
    """
    Validate input data format.
    
    Args:
        input_data: Input DataFrame to validate
        expected_features: List of expected feature names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if DataFrame is empty
    if input_data.empty:
        return False, "Input data is empty"
    
    # Check for expected columns (with or without '_encoded' suffix)
    missing_features = []
    for feature in expected_features:
        antibiotic_name = feature.replace('_encoded', '')
        if feature not in input_data.columns and antibiotic_name not in input_data.columns:
            missing_features.append(antibiotic_name)
    
    if len(missing_features) > 10:  # Allow some missing features
        return False, f"Too many missing features: {len(missing_features)}/{len(expected_features)}"
    
    return True, ""


def calculate_mar_index(resistance_profile: Union[pd.Series, np.ndarray, Dict]) -> float:
    """
    Calculate MAR index from resistance profile.
    
    Args:
        resistance_profile: Resistance values (0=S, 1=I, 2=R)
        
    Returns:
        MAR index (0.0 to 1.0)
    """
    if isinstance(resistance_profile, dict):
        values = np.array(list(resistance_profile.values()))
    elif isinstance(resistance_profile, pd.Series):
        values = resistance_profile.values
    else:
        values = resistance_profile
    
    # Count resistant (value == 2)
    num_resistant = np.sum(values == 2)
    # Count total tested (non-missing)
    num_tested = len(values[~np.isnan(values)]) if isinstance(values[0], float) else len(values)
    
    if num_tested == 0:
        return 0.0
    
    return num_resistant / num_tested


# ============================================================================
# Prediction Functions
# ============================================================================

def predict_mar(model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict MAR class with probabilities.
    
    Args:
        model: Trained classification model
        X: Feature matrix
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    predictions = model.predict(X)
    
    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
    else:
        # For models without predict_proba, use decision function or create binary
        probabilities = np.zeros((len(predictions), 2))
        probabilities[np.arange(len(predictions)), predictions] = 1.0
    
    return predictions, probabilities


def predict_species(model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict species with probabilities.
    
    Args:
        model: Trained classification model
        X: Feature matrix
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    predictions = model.predict(X)
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
    else:
        # Create one-hot encoding for models without probabilities
        num_classes = len(np.unique(predictions))
        probabilities = np.zeros((len(predictions), num_classes))
        probabilities[np.arange(len(predictions)), predictions] = 1.0
    
    return predictions, probabilities


def get_prediction_confidence(probabilities: np.ndarray) -> float:
    """
    Calculate confidence score from probabilities.
    
    Args:
        probabilities: Probability array for one sample
        
    Returns:
        Confidence score (max probability)
    """
    return np.max(probabilities)


def get_confidence_level(confidence: float) -> Tuple[str, str]:
    """
    Get confidence level and color based on confidence score.
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        Tuple of (level_name, color)
    """
    if confidence >= 0.8:
        return "High", "green"
    elif confidence >= 0.6:
        return "Medium", "orange"
    else:
        return "Low", "red"


# ============================================================================
# Visualization Functions
# ============================================================================

def create_probability_chart(probabilities: np.ndarray, class_names: List[str]) -> go.Figure:
    """
    Create Plotly bar chart of class probabilities.
    
    Args:
        probabilities: Probability array for one sample
        class_names: Names of classes
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=probabilities,
            marker_color=['#2ca02c' if p == max(probabilities) else '#1f77b4' for p in probabilities],
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Class',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False
    )
    
    return fig


def create_feature_importance_chart(importances: np.ndarray, feature_names: List[str], top_n: int = 20) -> go.Figure:
    """
    Create feature importance bar chart.
    
    Args:
        importances: Feature importance values
        feature_names: Names of features
        top_n: Number of top features to display
        
    Returns:
        Plotly figure
    """
    # Sort and select top N
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i].replace('_encoded', '') for i in indices]
    top_importances = importances[indices]
    
    fig = go.Figure(go.Bar(
        x=top_importances,
        y=top_features,
        orientation='h',
        marker=dict(
            color=top_importances,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=[f'{imp:.4f}' for imp in top_importances],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Most Important Antibiotics',
        xaxis_title='Importance Score',
        yaxis_title='Antibiotic',
        height=max(400, top_n * 25),
        yaxis=dict(autorange="reversed"),
        showlegend=False
    )
    
    return fig


def create_radar_chart(resistance_profile: Dict[str, int], antibiotic_names: List[str]) -> go.Figure:
    """
    Create radar chart of resistance profile.
    
    Args:
        resistance_profile: Dictionary mapping antibiotics to resistance values
        antibiotic_names: List of antibiotic names to include
        
    Returns:
        Plotly figure
    """
    # Get values in order
    values = []
    labels = []
    for antibiotic in antibiotic_names:
        if antibiotic in resistance_profile:
            values.append(resistance_profile[antibiotic])
            # Clean up antibiotic name for display
            labels.append(antibiotic.replace('_', ' ').title())
    
    # Close the radar chart
    values.append(values[0])
    labels.append(labels[0])
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        marker=dict(color='#1f77b4'),
        line=dict(color='#1f77b4')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 2],
                tickvals=[0, 1, 2],
                ticktext=['S', 'I', 'R']
            )
        ),
        showlegend=False,
        title='Resistance Profile',
        height=500
    )
    
    return fig


def create_umap_plot(embeddings: np.ndarray, labels: np.ndarray, 
                     label_names: Optional[List[str]] = None,
                     new_point: Optional[np.ndarray] = None,
                     title: str = 'UMAP Projection') -> go.Figure:
    """
    Create UMAP scatter plot with optional new point highlight.
    
    Args:
        embeddings: 2D UMAP embeddings
        labels: Cluster/class labels
        label_names: Optional names for labels
        new_point: Optional new point to highlight
        title: Plot title
        
    Returns:
        Plotly figure
    """
    df = pd.DataFrame({
        'UMAP1': embeddings[:, 0],
        'UMAP2': embeddings[:, 1],
        'Label': labels
    })
    
    if label_names is not None:
        df['Label'] = df['Label'].map(lambda x: label_names[x] if x < len(label_names) else str(x))
    
    fig = px.scatter(
        df,
        x='UMAP1',
        y='UMAP2',
        color='Label',
        title=title,
        opacity=0.6,
        height=600
    )
    
    # Add new point if provided
    if new_point is not None:
        fig.add_trace(go.Scatter(
            x=[new_point[0]],
            y=[new_point[1]],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='star',
                line=dict(color='darkred', width=2)
            ),
            name='New Prediction',
            showlegend=True
        ))
    
    fig.update_layout(
        xaxis_title='UMAP Dimension 1',
        yaxis_title='UMAP Dimension 2',
        legend_title='Cluster/Class'
    )
    
    return fig


def create_confusion_matrix_plot(cm: np.ndarray, labels: List[str]) -> go.Figure:
    """
    Create interactive confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix
        labels: Class labels
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=600,
        height=600
    )
    
    return fig


# ============================================================================
# Helper Functions
# ============================================================================

def format_metrics_table(metrics_dict: Dict) -> pd.DataFrame:
    """
    Format metrics dictionary as a styled DataFrame.
    
    Args:
        metrics_dict: Dictionary of model metrics
        
    Returns:
        Formatted DataFrame
    """
    df = pd.DataFrame(metrics_dict).T
    df = df.round(4)
    return df


def create_download_link(df: pd.DataFrame, filename: str, link_text: str) -> str:
    """
    Create a download link for a DataFrame.
    
    Args:
        df: DataFrame to download
        filename: Name for the downloaded file
        link_text: Text to display for the link
        
    Returns:
        HTML string for download link
    """
    csv = df.to_csv(index=False)
    return f'<a href="data:file/csv;base64,{csv}" download="{filename}">{link_text}</a>'
