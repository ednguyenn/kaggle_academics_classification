import os 
import sys
import pandas as pd 


from sklearn.model_selection import train_val_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from utils import CustomException, logging,save_object, column_division

#a convenient way to define classes to store data
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('data/processed',"train.csv")
    valid_data_path: str=os.path.join('data/processed',"valid.csv")

#class with data ingestion operation methods
class DataIngestion:
    #allow DataIngestion to access the paths defined in DataIngestionConfig
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        """_summary_
        This method will retrieve data from origin source and save it to project directory
        Then perform train_val_split and save them into corresponding folder

        Returns: paths to train and val data
       
        """
        logging.info("Entered data ingestion method or component")
        try:
            #read files from original source (raw data source)
            df = pd.read_csv('data/raw/train.csv')
            logging.info("Read the raw training dataset as dataframe")
            
            #ensures that the destination directory for storing processed data exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            
            #train val split
            logging.info("Train val split initiated")
            train_set, val_set=train_val_split(df,val_size=0.2, random_state=112)
            
            #save train and val data into local destination
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            val_set.to_csv(self.ingestion_config.valid_data_path,index=False,header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.valid_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)    

@dataclass            
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('models',"preprocessor.pkl")
    X_train_data_path: str=os.path.join('data/final',"X_train.csv")
    y_train_data_path: str=os.path.join('data/final',"y_train.csv")
    X_val_data_path: str=os.path.join('data/final',"X_val.csv")
    y_val_data_path: str=os.path.join('data/final',"y_val.csv")
    
class DataTransformartion:
    #initialize an attribute to save the processed data paths
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    
    def preprocessing(self,df):
        return df 
            
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
                    ("impute",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
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
        
    def initiate_data_transformation(self,train_path,val_path):
        """
        This method will user preproccesor to transform data for model training
        save X_train, y_train, X_val, y_val into data/processed folders 
        """
        try:
            
            train_df=pd.read_csv(train_path)
            val_df=pd.read_csv(val_path)
            
            logging.info("Read train and val data")
                                         
            target_column_name="Target"
            
            # Apply preprocessing function to the entire DataFrame
            train_df = self.preprocessing(train_df)
            val_df = self.preprocessing(val_df)
            
            X_train= train_df.drop(columns=[target_column_name],axis=1)
            y_train= train_df[target_column_name]
            
            X_val=val_df.drop(columns=[target_column_name],axis=1)
            y_val=val_df[target_column_name]

            logging.info("Applying preprocessing object on training set and val set")
            
            preprocessing_obj=self.get_data_transformer_object(train_df)
            
            X_train= preprocessing_obj.fit_transform(X_train)
            X_val=preprocessing_obj.transform(X_val)
            
            #save the processed datasets into data/final folder
            X_train_df = pd.DataFrame(X_train)
            X_train_df.to_csv(self.data_transformation_config.X_train_data_path,index=False,header=True)
            y_train.to_csv(self.data_transformation_config.y_train_data_path,index=False,header=True)
            
            X_val_df = pd.DataFrame(X_val)
            X_val_df.to_csv(self.data_transformation_config.X_val_data_path,index=False,header=True)
            y_val.to_csv(self.data_transformation_config.y_val_data_path,index=False,header=True)
               
               
               
            logging.info("Saved preprocessing object")
            
            #save object into destination folder
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                X_train,
                y_train,
                X_val,
                y_val
            )
        except Exception as e:
            raise CustomException(e,sys)      
 
 
        
if __name__ == "__main__":
    obj=DataIngestion()
    train_data, val_data =obj.initiate_data_ingestion()
    
    data_transformation=DataTransformartion()
    X_train,y_train,X_val,y_val=data_transformation.initiate_data_transformation(train_data,val_data)

    
    #import ModelTrainer here to avoid circular import errror between process.py and train_model.py
    from train_model import ModelTrainer
    
    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(X_train,y_train))