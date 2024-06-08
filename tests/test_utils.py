import pytest
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.utils import save_object, load_object, cross_validate_model, column_division, CustomException, plot_feature_importances

# Test save_object function
def test_save_object(tmp_path):
    obj = {"key": "value"}
    file_path = tmp_path / "test.pkl"

    save_object(file_path, obj)

    # Check if file exists
    assert os.path.exists(file_path)

    # Check if the content is correct
    with open(file_path, 'rb') as file:
        loaded_obj = pickle.load(file)
        assert loaded_obj == obj

# Test load_object function
def test_load_object(tmp_path):
    obj = {"key": "value"}
    file_path = tmp_path / "test.pkl"

    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

    loaded_obj = load_object(file_path)
    assert loaded_obj == obj

# Test cross_validate_model function
def test_cross_validate_model():
    X, y = pd.DataFrame(np.random.rand(100, 10)), pd.Series(np.random.randint(0, 2, size=100))
    model_cls = RandomForestClassifier
    params = {"n_estimators": 10, "random_state": 42}

    model, score = cross_validate_model(model_cls, X, y, params=params)

    assert isinstance(model, RandomForestClassifier)
    assert isinstance(score, float)
    assert 0 <= score <= 1

# Test column_division function
def test_column_division():
    df = pd.DataFrame({
        'num_col1': [1, 2, 3, 4, 5],
        'num_col2': [10, 20, 30, 40, 50],
        'cat_col1': ['A', 'B', 'A', 'B', 'A'],
        'cat_col2': ['X', 'Y', 'X', 'Y', 'X']
    })

    cat_cols, num_cols = column_division(threshold=3, df=df)

    assert set(cat_cols) == {'cat_col1', 'cat_col2'}
    assert set(num_cols) == {'num_col1', 'num_col2'}

# Test CustomException
def test_custom_exception():
    with pytest.raises(CustomException):
        raise CustomException("This is a custom exception", sys)

# Test save_object error handling
def test_save_object_exception():
    with pytest.raises(CustomException):
        save_object("/invalid_path/test.pkl", {"key": "value"})

# Test plot_feature_importances function
def test_plot_feature_importances():
    df = pd.DataFrame(np.random.rand(100, 5), columns=[f"feature_{i}" for i in range(5)])
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(df, np.random.randint(0, 2, size=100))

    fig = plot_feature_importances(model, "RandomForest", dataframe=df)

    assert fig is not None
    assert "Importance" in fig.data[0]['x']
    assert "Feature" in fig.data[0]['y']
