import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
)
from ml.data import process_data

@pytest.fixture
def mock_data():
    X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y = np.array([1, 0, 1])
    return X, y

def test_train_model_returns_expected_algorithm(mock_data):
    """
    Test that the train_model function returns an instance of the expected algorithm.
    """
    X, y = mock_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"

def test_compute_model_metrics_values():
    """
    Test that the compute_model_metrics function returns expected metric values.
    """
    y = np.array([1, 0, 1])
    preds = np.array([1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == 1.0, "Precision calculation is incorrect"
    assert recall == 0.5, "Recall calculation is incorrect"
    assert fbeta == 0.6666666666666666, "F1-score calculation is incorrect"

def test_inference_returns_correct_shape(mock_data):
    """
    Test that the inference function returns predictions of the correct shape.
    """
    X, y = mock_data
    model = train_model(X, y)
    preds = inference(model, X)

    assert preds.shape == y.shape, "Predictions do not have the same shape as the input labels"
    assert preds.dtype == y.dtype, "Predictions do not have the same data type as the labels"