import pytest
import os
import sys

import yaml
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from unittest.mock import patch, mock_open
from src.train_model import ModelTrainer, ModelTrainerConfig, CustomException

# Create fixture for the model config file content
@pytest.fixture
def model_config_content():
    return {
        'model': {
            'name': 'xgboost.XGBClassifier',
            'params': {
                'colsample_bytree': 0.911,
                'gamma': 5.714,
                'max_depth': 13,
                'min_child_weight': 3.0,
                'reg_alpha': 40.0,
                'reg_lambda': 0.44
            }
        }
    }

# Create a fixture to set up a temporary directory
@pytest.fixture
def tmp_dir(tmp_path, model_config_content):
    model_config_file_path = tmp_path / "model1.yaml"
    with open(model_config_file_path, 'w') as file:
        yaml.dump(model_config_content, file)
    return tmp_path

# Test loading the model config file
def test_load_model_config(tmp_dir):
    trainer_config = ModelTrainerConfig(
        trained_model_file_path=str(tmp_dir / 'model.pkl'),
        model_config_file_path=str(tmp_dir / 'model1.yaml')
    )
    trainer = ModelTrainer()
    trainer.model_trainer_config = trainer_config

    model_config = trainer.load_model_config()
    assert model_config['model']['name'] == 'xgboost.XGBClassifier'
    assert 'params' in model_config['model']

# Mock cross_validate_model for testing model training
@pytest.fixture
def mock_cross_validate_model():
    with patch('model_training.cross_validate_model') as mock_cv_model:
        yield mock_cv_model

# Test initiate_model_trainer
def test_initiate_model_trainer(tmp_dir, mock_cross_validate_model):
    X_train = pd.DataFrame(np.random.rand(100, 10))
    y_train = pd.Series(np.random.randint(0, 2, size=100))

    # Set the return value of the mock cross_validate_model
    mock_cross_validate_model.return_value = (XGBClassifier(), 0.85)

    trainer_config = ModelTrainerConfig(
        trained_model_file_path=str(tmp_dir / 'model.pkl'),
        model_config_file_path=str(tmp_dir / 'model1.yaml')
    )
    trainer = ModelTrainer()
    trainer.model_trainer_config = trainer_config
    trainer.model_config = trainer.load_model_config()

    result = trainer.initiate_model_trainer(X_train, y_train)

    assert "Best model accuracy score is: 0.85" in result
    assert os.path.exists(trainer.model_trainer_config.trained_model_file_path)

# Test exception handling in load_model_config
def test_load_model_config_exception():
    trainer = ModelTrainer()
    with patch('builtins.open', mock_open(read_data='invalid_yaml: [')) as mock_file:
        with pytest.raises(CustomException):
            trainer.load_model_config()

# Test exception handling in initiate_model_trainer
def test_initiate_model_trainer_exception(tmp_dir):
    X_train = pd.DataFrame(np.random.rand(100, 10))
    y_train = pd.Series(np.random.randint(0, 2, size=100))

    trainer_config = ModelTrainerConfig(
        trained_model_file_path=str(tmp_dir / 'model.pkl'),
        model_config_file_path=str(tmp_dir / 'model1.yaml')
    )
    trainer = ModelTrainer()
    trainer.model_trainer_config = trainer_config

    # Patch the load_model_config to raise an exception
    with patch.object(trainer, 'load_model_config', side_effect=CustomException("Error", sys)):
        with pytest.raises(CustomException):
            trainer.initiate_model_trainer(X_train, y_train)
