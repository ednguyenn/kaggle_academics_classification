import os 
import sys
import pandas as pd 


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import numpy as np

from utils import CustomException, logging,save_object,cross_validate_model, column_division

#a convenient way to define classes to store data
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('data/raw',"train.csv")
    test_data_path: str=os.path.join('data/raw',"test.csv")
    
    

@dataclass            
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('models',"preprocessor.pkl")
    X_train_data_path: str=os.path.join('data/processed',"X_train.csv")
    y_train_data_path: str=os.path.join('data/processed',"y_train.csv")
    X_test_data_path: str=os.path.join('data/processed',"X_test.csv")
    y_test_data_path: str=os.path.join('data/processed',"y_test.csv")
    
class DataTransformartion:
    #initialize an attribute to save the processed data paths
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
            
    def get_data_transformer_object(self,df):
        """
        This method generates data proprocessor 
        """
        try:
            #sorted numerical and categorical columnsd
            cat_cols, num_cols = column_division(threshold=8,df=df)
            #create seperate Pipeline for numerical and categorical columns
            num_pipeline = Pipeline(
                steps=[
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("oh_encoder",OneHotEncoder())
                ]
            )
            
            logging.info("Columns transformed !")
            
            #create a mutual preprocessor for all columns 
            preprocessor=ColumnTransformer(
                [("numerical_pipeline",num_pipeline,num_cols),
                 ("categorical_pipeline",categorical_pipeline,cat_cols)]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        """
        This method will user preproccesor to transform data for model training
        save X_train, y_train, X_test, y_test into data/processed folders 
        """
        try:
            
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train and test data")
                                         
            target_column_name="Target"
            
            X_train= train_df.drop(columns=[target_column_name],axis=1)
            y_train= train_df[target_column_name]
            
            X_test=test_df.drop(columns=[target_column_name],axis=1)
            y_test=test_df[target_column_name]

            logging.info("Applying preprocessing object on training set and test set")
            
            preprocessing_obj=self.get_data_transformer_object(train_df)
            
            X_train= preprocessing_obj.fit_transform(X_train)
            X_test=preprocessing_obj.transform(X_test)
            
            #save the processed datasets into data/processed folder
            X_train_df = pd.DataFrame(X_train)
            X_train_df.to_csv(self.data_transformation_config.X_train_data_path,index=False,header=True)
            y_train.to_csv(self.data_transformation_config.y_train_data_path,index=False,header=True)
            
            X_test_df = pd.DataFrame(X_test)
            X_test_df.to_csv(self.data_transformation_config.X_test_data_path,index=False,header=True)
            y_test.to_csv(self.data_transformation_config.y_test_data_path,index=False,header=True)
               
               
               
            logging.info("Saved preprocessing object")
            
            #save object into destination folder
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                X_train,
                y_train,
                X_test,
                y_test
            )
        except Exception as e:
            raise CustomException(e,sys)      
        
if __name__ == "__main__":
    obj=DataIngestionConfig()
    train_data=obj.train_data_path
    test_data=obj.test_data_path
    
    data_transformation=DataTransformartion()
    X_train,y_train,X_test,y_test=data_transformation.initiate_data_transformation(train_data,test_data)

    
    #import ModelTrainer here to avoid circular import errror between process.py and train_model.py
    from train_model import ModelTrainer
    
    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(X_train,y_train,X_test,y_test))