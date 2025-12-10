"""
Unit tests for supervised learning and evaluation functions

Tests for classifier training, evaluation metrics, and visualizations.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from src.models.supervised import (
    get_classifier,
    train_classifier,
    get_all_classifiers,
    get_param_grid,
    tune_hyperparameters,
    predict,
    predict_proba
)
from src.models.evaluation import (
    evaluate_classifier,
    get_confusion_matrix,
    get_classification_report,
    calculate_roc_auc,
    get_feature_importance,
    get_permutation_importance,
    compare_models,
    get_best_model
)


class TestSupervisedLearning:
    """Test cases for supervised learning module."""
    
    @pytest.fixture
    def binary_classification_data(self):
        """Create binary classification dataset."""
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        return pd.DataFrame(X), pd.Series(y)
    
    @pytest.fixture
    def multiclass_classification_data(self):
        """Create multi-class classification dataset."""
        X, y = make_classification(
            n_samples=150,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42
        )
        return pd.DataFrame(X), pd.Series(y)
    
    def test_get_classifier(self):
        """Test getting a classifier by name."""
        # Test all classifier types
        classifier_names = [
            'random_forest', 'xgboost', 'logistic_regression',
            'svm', 'knn', 'naive_bayes'
        ]
        
        for name in classifier_names:
            clf = get_classifier(name, random_state=42)
            assert clf is not None
    
    def test_get_classifier_invalid_name(self):
        """Test getting classifier with invalid name."""
        with pytest.raises(ValueError):
            get_classifier('invalid_classifier')
    
    def test_train_classifier(self, binary_classification_data):
        """Test training a classifier."""
        X, y = binary_classification_data
        
        clf = get_classifier('random_forest', random_state=42)
        trained_clf = train_classifier(clf, X, y)
        
        assert hasattr(trained_clf, 'predict')
        # Check it's actually trained
        predictions = trained_clf.predict(X)
        assert len(predictions) == len(y)
    
    def test_get_all_classifiers(self):
        """Test getting all classifiers."""
        classifiers = get_all_classifiers(random_state=42)
        
        assert isinstance(classifiers, dict)
        assert len(classifiers) == 6
        assert 'random_forest' in classifiers
        assert 'xgboost' in classifiers
    
    def test_get_param_grid(self):
        """Test getting parameter grids."""
        classifier_names = [
            'random_forest', 'xgboost', 'logistic_regression',
            'svm', 'knn', 'naive_bayes'
        ]
        
        for name in classifier_names:
            param_grid = get_param_grid(name)
            assert isinstance(param_grid, dict)
            assert len(param_grid) > 0
    
    def test_tune_hyperparameters(self, binary_classification_data):
        """Test hyperparameter tuning."""
        X, y = binary_classification_data
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        clf = get_classifier('random_forest', random_state=42)
        
        # Use a small param grid for testing
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
        
        best_model, best_params, metrics = tune_hyperparameters(
            clf, X_train, y_train, X_val, y_val, param_grid, cv=2
        )
        
        assert best_model is not None
        assert isinstance(best_params, dict)
        assert isinstance(metrics, dict)
        assert 'val_score' in metrics
    
    def test_predict(self, binary_classification_data):
        """Test making predictions."""
        X, y = binary_classification_data
        
        clf = get_classifier('random_forest', random_state=42)
        clf = train_classifier(clf, X, y)
        
        predictions = predict(clf, X)
        
        assert len(predictions) == len(y)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_predict_proba(self, binary_classification_data):
        """Test getting prediction probabilities."""
        X, y = binary_classification_data
        
        clf = get_classifier('random_forest', random_state=42)
        clf = train_classifier(clf, X, y)
        
        probas = predict_proba(clf, X)
        
        assert probas is not None
        assert probas.shape[0] == len(y)
        assert probas.shape[1] == 2  # Binary classification
        # Check probabilities sum to 1
        assert np.allclose(probas.sum(axis=1), 1.0)


class TestEvaluation:
    """Test cases for evaluation module."""
    
    @pytest.fixture
    def binary_predictions(self):
        """Create binary classification predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 0])
        y_proba = np.array([
            [0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3],
            [0.1, 0.9], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8], [0.8, 0.2]
        ])
        return y_true, y_pred, y_proba
    
    def test_evaluate_classifier(self, binary_predictions):
        """Test classifier evaluation."""
        y_true, y_pred, y_proba = binary_predictions
        
        metrics = evaluate_classifier(y_true, y_pred, y_proba[:, 1])
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
        
        # Check metric values are in valid range
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1
    
    def test_get_confusion_matrix(self, binary_predictions):
        """Test confusion matrix generation."""
        y_true, y_pred, _ = binary_predictions
        
        cm = get_confusion_matrix(y_true, y_pred)
        
        assert isinstance(cm, np.ndarray)
        assert cm.shape == (2, 2)  # Binary classification
        assert cm.sum() == len(y_true)
    
    def test_get_classification_report(self, binary_predictions):
        """Test classification report generation."""
        y_true, y_pred, _ = binary_predictions
        
        report = get_classification_report(y_true, y_pred)
        
        assert isinstance(report, dict)
        assert '0' in report
        assert '1' in report
        assert 'accuracy' in report
        assert 'macro avg' in report
        assert 'weighted avg' in report
    
    def test_calculate_roc_auc(self, binary_predictions):
        """Test ROC-AUC calculation."""
        y_true, y_pred, y_proba = binary_predictions
        
        auc_score = calculate_roc_auc(y_true, y_proba[:, 1])
        
        assert isinstance(auc_score, float)
        assert 0 <= auc_score <= 1
    
    def test_get_feature_importance_tree_model(self):
        """Test feature importance extraction from tree model."""
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        feature_names = [f'feature_{i}' for i in range(5)]
        
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        
        importance_df = get_feature_importance(clf, feature_names)
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == 5
        # Check sorted in descending order
        assert all(importance_df['importance'].iloc[i] >= importance_df['importance'].iloc[i+1]
                  for i in range(len(importance_df)-1))
    
    def test_get_permutation_importance(self):
        """Test permutation importance calculation."""
        from sklearn.ensemble import RandomForestClassifier
        
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        feature_names = [f'feature_{i}' for i in range(5)]
        
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X, y)
        
        importance_df = get_permutation_importance(
            clf, X, y, feature_names, n_repeats=5, random_state=42
        )
        
        assert isinstance(importance_df, pd.DataFrame)
        assert 'feature' in importance_df.columns
        assert 'importance_mean' in importance_df.columns
        assert 'importance_std' in importance_df.columns
        assert len(importance_df) == 5
    
    def test_compare_models(self):
        """Test model comparison."""
        results = {
            'model_a': {'accuracy': 0.85, 'f1': 0.82, 'precision': 0.80},
            'model_b': {'accuracy': 0.90, 'f1': 0.88, 'precision': 0.85},
            'model_c': {'accuracy': 0.75, 'f1': 0.73, 'precision': 0.70}
        }
        
        comparison_df = compare_models(results)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 3
        assert 'accuracy' in comparison_df.columns
        assert 'f1' in comparison_df.columns
        # Check sorted by f1 (descending)
        assert comparison_df.index[0] == 'model_b'
    
    def test_get_best_model(self):
        """Test getting best model."""
        results = {
            'model_a': {'accuracy': 0.85, 'f1': 0.82},
            'model_b': {'accuracy': 0.90, 'f1': 0.88},
            'model_c': {'accuracy': 0.75, 'f1': 0.73}
        }
        
        best_model = get_best_model(results, metric='f1')
        assert best_model == 'model_b'
        
        best_model_acc = get_best_model(results, metric='accuracy')
        assert best_model_acc == 'model_b'


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
