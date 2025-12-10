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


def load_model(model_name: str):
    """
    Load a trained model from the models directory.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Loaded model object
        
    TODO: Implement model loading with error handling
    """
    model_path = Path(f"../models/{model_name}.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model {model_name} not found")


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess input data for prediction.
    
    Args:
        df: Raw input DataFrame
        
    Returns:
        Preprocessed DataFrame ready for prediction
        
    TODO: Apply same preprocessing steps as training data
    """
    pass


def create_prediction_plot(predictions: np.ndarray, probabilities: np.ndarray = None):
    """
    Create visualization for predictions.
    
    Args:
        predictions: Array of predicted classes
        probabilities: Optional array of prediction probabilities
        
    Returns:
        Plotly figure
        
    TODO: Implement prediction visualization
    """
    pass


def create_confusion_matrix_plot(cm: np.ndarray, labels: list):
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
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=600,
        height=600
    )
    
    return fig


def create_roc_curve_plot(fpr: np.ndarray, tpr: np.ndarray, auc: float):
    """
    Create interactive ROC curve plot.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: Area under the curve
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=700,
        height=600,
        showlegend=True
    )
    
    return fig


def create_feature_importance_plot(feature_names: list, importances: np.ndarray, top_n: int = 20):
    """
    Create feature importance bar plot.
    
    Args:
        feature_names: List of feature names
        importances: Array of feature importances
        top_n: Number of top features to display
        
    Returns:
        Plotly figure
    """
    # Sort and select top N
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    fig = go.Figure(go.Bar(
        x=top_importances,
        y=top_features,
        orientation='h',
        marker=dict(color=top_importances, colorscale='Viridis')
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Feature Importances',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=600,
        showlegend=False
    )
    
    return fig


def calculate_mar_index_from_df(df: pd.DataFrame, antibiotic_columns: list) -> pd.Series:
    """
    Calculate MAR index for uploaded data.
    
    Args:
        df: DataFrame containing antibiotic resistance data
        antibiotic_columns: List of antibiotic column names
        
    Returns:
        Series of MAR indices
        
    TODO: Implement MAR calculation
    """
    pass


def format_metrics_table(metrics_dict: dict) -> pd.DataFrame:
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
