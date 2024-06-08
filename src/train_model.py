import sys
import os
import yaml

from utils import CustomException, logging, save_object, cross_validate_model
from dataclasses import dataclass

import xgboost as xgb

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('models', 'model.pkl')
    model_config_file_path = os.path.join('model1.yaml')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.model_config = self.load_model_config()

    def load_model_config(self):
        try:
            with open(self.model_trainer_config.model_config_file_path, 'r') as file:
                model_config = yaml.safe_load(file)
            return model_config
        except Exception as e:
            raise CustomException(f"Error loading model config: {str(e)}", sys)

    def initiate_model_trainer(self, X_train, y_train):
        try:
            # Get model and params from config
            model_name = self.model_config['model']['name']
            params = self.model_config['model']['params']
            
            # Dynamically import the model
            model_cls = getattr(xgb, model_name.split('.')[1])
            
            # Evaluate model
            best_model, best_model_score = cross_validate_model(model_cls, X_train, y_train, params=params)
            
            logging.info("Completed model training")
            
            # Save the trained model to the destination folder
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            return f"Best model accuracy score is: {best_model_score}"
        except Exception as e:
            raise CustomException(e, sys)