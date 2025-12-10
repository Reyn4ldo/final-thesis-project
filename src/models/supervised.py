"""
Supervised Learning Module

This module contains implementations of supervised learning classifiers
for AMR pattern prediction.
"""

from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


def get_classifier(name: str, random_state: int = 42):
    """
    Get a classifier instance by name.
    
    Args:
        name: Name of the classifier ('random_forest', 'xgboost', 'logistic_regression',
              'svm', 'knn', 'naive_bayes')
        random_state: Random state for reproducibility
        
    Returns:
        Classifier instance
        
    Raises:
        ValueError: If classifier name is not recognized
    """
    classifiers = {
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            class_weight='balanced',
            random_state=random_state
        ),
        'xgboost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric='logloss'
        ),
        'logistic_regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state
        ),
        'svm': SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=random_state
        ),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'naive_bayes': GaussianNB()
    }
    
    if name not in classifiers:
        raise ValueError(f"Unknown classifier: {name}. Choose from: {list(classifiers.keys())}")
    
    return classifiers[name]


def train_classifier(model, X_train, y_train):
    """
    Train a classifier.
    
    Args:
        model: Scikit-learn compatible model
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    model.fit(X_train, y_train)
    return model


def get_all_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """
    Get all available classifiers as a dictionary.
    
    Args:
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary mapping classifier names to instances
    """
    classifier_names = [
        'random_forest',
        'xgboost',
        'logistic_regression',
        'svm',
        'knn',
        'naive_bayes'
    ]
    
    return {name: get_classifier(name, random_state) for name in classifier_names}


def get_param_grid(model_name: str) -> Dict[str, list]:
    """
    Get hyperparameter grid for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of hyperparameters to tune
    """
    param_grids = {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'xgboost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        },
        'logistic_regression': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'svm': {
            'C': [0.1, 1.0, 10.0],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'linear']
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'naive_bayes': {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
    }
    
    if model_name not in param_grids:
        raise ValueError(f"Unknown model: {model_name}")
    
    return param_grids[model_name]


def tune_hyperparameters(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    param_grid: Dict[str, list],
    cv: int = 3,
    scoring: str = 'f1_weighted'
) -> Tuple[Any, Dict[str, Any], Dict[str, float]]:
    """
    Tune hyperparameters using GridSearchCV with validation set.
    
    Args:
        model: Base model to tune
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        param_grid: Dictionary of parameters to search
        cv: Number of cross-validation folds
        scoring: Scoring metric
        
    Returns:
        Tuple of (best_model, best_params, validation_score)
    """
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluate on validation set
    val_score = best_model.score(X_val, y_val)
    
    validation_metrics = {
        'best_params': best_params,
        'cv_score': grid_search.best_score_,
        'val_score': val_score
    }
    
    return best_model, best_params, validation_metrics


def predict(model, X) -> np.ndarray:
    """
    Make predictions using a trained model.
    
    Args:
        model: Trained model
        X: Features to predict
        
    Returns:
        Array of predictions
    """
    return model.predict(X)


def predict_proba(model, X) -> Optional[np.ndarray]:
    """
    Get prediction probabilities from a trained model.
    
    Args:
        model: Trained model
        X: Features to predict
        
    Returns:
        Array of prediction probabilities, or None if not available
    """
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    else:
        return None
