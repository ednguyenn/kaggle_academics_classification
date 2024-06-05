import sys
import os 
from utils import CustomException,logging,save_object,cross_validate_model
from dataclasses import dataclass




from sklearn.metrics import accuracy_score
import sklearn.model_selection 
from process import DataIngestionConfig, DataTransformartion
from sklearn.ensemble import GradientBoostingClassifier

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('models','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
            
    def initiate_model_trainer(self, X_train,y_train,X_test,y_test):
        try:
            #list out models and corresponding params will be used
            models= { 
                     "gdboost": GradientBoostingClassifier()
            }
            params={
                "gdboost": {
                        'learning_rate': 0.1,
                        'n_estimators': 100,
                        'max_depth': 3,
                        'min_samples_split': 2,
                        'min_samples_leaf': 1,
                        'subsample': 1,
                        'random_state': 42,
                        'ccp_alpha': 0.0001

                }
                          
            }
            
            #########################
            #use the cross_validate_model for further task
            
            #evaluate different model then save them into a dictionary
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test, models= models,params=params)
            
            #get the model with the highest score
            best_model_score = max(sorted(model_report.values()))
            
            #get the best model name
            best_model_name  = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model= models[best_model_name]
            
            #setting the baseline threshold for the best model 
            if best_model_score<0.7:
                raise CustomException("No best model is not found")
            
            logging.info("Found the best model for the dataset")
            
            #save modelTrainer into destination folder
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            return print(f"Best model auc_roc_score is : {best_model_score}")
        except Exception as e:
            raise CustomException(e,sys)