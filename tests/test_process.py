import os
import pandas as pd
import pytest
from src.process import DataTransformartion, DataTransformationConfig
from src.utils import CustomException

# Create a fixture for the sample data paths
@pytest.fixture
def sample_data_paths(tmp_path):
    train_data = tmp_path / "sample_train.csv"
    test_data = tmp_path / "sample_test.csv"

    # Sample train data
    train_data.write_text(
        "Feature1,Feature2,Feature3,Target\n"
        "1,A,10,0\n"
        "2,B,20,1\n"
        "3,C,30,0\n"
    )

    # Sample test data
    test_data.write_text(
        "Feature1,Feature2,Feature3,Target\n"
        "4,A,40,1\n"
        "5,B,50,0\n"
        "6,C,60,1\n"
    )
    return train_data, test_data

# Test get_data_transformer_object method
def test_get_data_transformer_object(sample_data_paths):
    train_data, test_data = sample_data_paths
    df = pd.read_csv(train_data)
    
    data_transformation = DataTransformartion()
    preprocessor = data_transformation.get_data_transformer_object(df)
    
    assert preprocessor is not None

# Test initiate_data_transformation method
def test_initiate_data_transformation(sample_data_paths):
    train_data, test_data = sample_data_paths
    
    data_transformation = DataTransformartion()
    
    X_train, y_train, X_test, y_test = data_transformation.initiate_data_transformation(train_data, test_data)
    
    assert X_train.shape == (3, 5)  # 3 samples, 2 numeric features, and 3 one-hot encoded features (A, B, C)
    assert y_train.shape == (3,)
    assert X_test.shape == (3, 5)   # 3 samples, 2 numeric features, and 3 one-hot encoded features (A, B, C)
    assert y_test.shape == (3,)

# Test exception handling
def test_initiate_data_transformation_exception():
    with pytest.raises(CustomException):
        data_transformation = DataTransformartion()
        data_transformation.initiate_data_transformation("invalid_train_path.csv", "invalid_test_path.csv")
