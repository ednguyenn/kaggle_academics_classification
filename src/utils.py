import sys
import os

import logging
from datetime import datetime
import pickle
import dill

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV





# Create a log file name with the current date and time in the format MM_DD_YYYY_HH_MM_SS.log
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Construct the path for the logs directory and the log file
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Create the logs directory if it doesn't already exist
os.makedirs(logs_path, exist_ok=True)

# Define the full path for the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logging module
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Set the filename for the log
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Define the log message format
    level=logging.INFO,  # Set the logging level to INFO
)


def error_message_detail(error,error_detail:sys):
    """
    Constructs a detailed error message string including the file name, line number, and error message.

    Args:
        error (Exception): The exception object that was raised.
        error_detail (module): The sys module, used to extract exception details.

    Returns:
        str: A formatted error message string.
    """
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))

    return error_message

    

class CustomException(Exception):
    """
    Custom exception class that includes a detailed error message.

    Attributes:
        error_message (str): Detailed error message including file name, line number, and error message.
    """
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
    
    
    
    
def save_object(file_path, obj):
    """ save an object to a fle using pickle serialization """
    try:
        # Extract the directory path from the file path
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)





#implement cross validation using Kfold
def cross_validate_model(model, X_train, y_train, params, n_splits=5):
    """
    Performs K-Fold cross-validation for a given model, returns the last model and average validation accuracy.

    Parameters:
        model: Machine learning model class (e.g., RandomForestClassifier)
        X_train: Training feature dataset
        y_train: Training target dataset
        params: Dictionary of parameters to initialize the model
        n_splits: Number of folds for cross-validation (default: 10)

    Returns:
        last_model: The last trained model instance
        average_val_accuracy: Average validation accuracy over all folds
    """
    # Initialize variables
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    val_scores = []

    # Cross-validation loop
    for fold, (train_ind, valid_ind) in enumerate(cv.split(X_train)):
        # Data splitting
        X_fold_train = X_train.iloc[train_ind]
        y_fold_train = y_train.iloc[train_ind]
        X_val = X_train.iloc[valid_ind]
        y_val = y_train.iloc[valid_ind]
        
        # Model initialization and training
        clf = model(**params)
        clf.fit(X_fold_train, y_fold_train)
        
        # Predict and evaluate
        y_pred_trn = clf.predict(X_fold_train)
        y_pred_val = clf.predict(X_val)
        train_acc = accuracy_score(y_fold_train, y_pred_trn)
        val_acc = accuracy_score(y_val, y_pred_val)
        print(f"Fold: {fold}, Train Accuracy: {train_acc:.5f}, Val Accuracy: {val_acc:.5f}")
        print("-" * 50)
        
        # Accumulate validation scores
        val_scores.append(val_acc)

    # Calculate the average validation score
    average_val_accuracy = np.mean(val_scores)
    print("Average Validation Accuracy:", average_val_accuracy)

    return clf, average_val_accuracy


def evaluate_models(X_train, y_train, X_test, y_test,models,params):
    """
    This function takes splitted datasets after transformation and output the models' auc_roc_score
    """
    try:
        acc = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]
            
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            y_pred=model.predict(X_test)
            
            score=roc_auc_score(y_test,y_pred)
            acc[list(models.keys())[i]] = score
            return acc
    except Exception as e:
        raise CustomException(e,sys)


            
            
def load_object(file_path):
    """ load an object for a file using dill deserialization"""
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
    
def column_division(threshold,df):
    # Initialize lists to hold categorical and numerical columns
    cat_cols = []
    num_cols = []

    # Iterate over the columns and categorize based on the unique values
    for col in df.columns:
        unique_values = df[col].nunique()
        if unique_values <= threshold:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return cat_cols, num_cols