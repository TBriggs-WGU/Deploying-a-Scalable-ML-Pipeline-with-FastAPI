# TODO: add necessary import
import numpy as np
import pandas as pd

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
from sklearn.linear_model import LogisticRegression

# Categorical list used by project pipeline
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def _toy_df():
    """Tiny, deterministic dataframe so tests run instantly and predictably."""
    return pd.DataFrame(
        {
            "age": [25, 40, 50, 23],
            "workclass": ["Private", "State-gov", "Private", "Private"],
            "fnlgt": [226802, 123011, 245487, 122272],
            "education": ["Bachelors", "HS-grad", "Masters", "HS-grad"],
            "education-num": [13, 9, 14, 9],
            "marital-status": [
                "Never-married",
                "Married-civ-spouse",
                "Married-civ-spouse",
                "Never-married",
            ],
            "occupation": [
                "Prof-specialty",
                "Adm-clerical",
                "Exec-managerial",
                "Handlers-cleaners",
            ],
            "relationship": ["Not-in-family", "Husband", "Husband", "Own-child"],
            "race": ["White", "White", "White", "Black"],
            "sex": ["Male", "Male", "Female", "Male"],
            "capital-gain": [0, 0, 0, 0],
            "capital-loss": [0, 0, 0, 0],
            "hours-per-week": [40, 40, 60, 20],
            "native-country": ["United-States"] * 4,
            "salary": ["<=50K", ">50K", ">50K", "<=50K"],
        }
    )


def test_train_model_returns_expected_algorithm():
    """
    # Test 1: Verifies the training function returns the expected algorithm type
    # Ensures train_model actually builds LogisticRegression
    """
    df = _toy_df()
    X, y, enc, lb = process_data(df, CAT_FEATURES, label="salary", training=True)
    model = train_model(X, y)
    assert isinstance(model, LogisticRegression)


def test_inference_output_shape_and_binary():
    """
    # Test 2: verifies inference output shape and values
    # Ensure 'inference' returns a 1D array of binary predictions with same length as input rows
    """
    df = _toy_df()
    X, y, enc, lb = process_data(df, CAT_FEATURES, label="salary", training=True)
    model = train_model(X, y)
    X2, y2, _, _ = process_data(
        df, CAT_FEATURES, label="salary", training=False, encoder=enc, lb=lb
    )
    preds = inference(model, X2)
    assert preds.shape == (X2.shape[0],)
    assert set(np.unique(preds)).issubset({0, 1})
    assert y2.shape == preds.shape


def test_compute_model_metrics_known_values():
    """
    # Test 3: Verifies metric computation on a known tiny example
    # Confirm 'compute_model_metrics' returns exact, expected values
    """
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])
    p, r, f1 = compute_model_metrics(y, preds)
    assert abs(p - 1.0) < 1e-6
    assert abs(r - 0.5) < 1e-6
    assert abs(f1 - (2 / 3)) < 1e-6