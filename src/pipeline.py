from src.utils import load_object, CustomException
from src.process import DataTransformartion

import sys 
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        """_summary_
        This method apply sequential steps to predict data and return prediction 
        """
        try:
            #assign model 
            model_path = 'models/model.pkl'
            model=load_object(file_path=model_path)
            
            #assign data transformer
            preprocessor_path='models/preprocessor.pkl'
            preprocessor=load_object(file_path=preprocessor_path)
            
            #perform data cleaning and transformation
            preprocessed_data = DataTransformartion.preprocessing(features)
            transformerd_data= preprocessor.transform(preprocessed_data)
            
            #make prediction 
            prediction= model.predict(transformerd_data)
            return prediction
        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 id: int,
                 marital_status: object,
                 application_mode: int,
                 application_order: int,
                 course: object,
                 attendance: object,
                 previous_qualification: object,
                 previous_qualification_grade: float,
                 nationality: object,
                 mother_qualification: object,
                 father_qualification: object,
                 mother_occupation: object,
                 father_occupation: object,
                 admission_grade: float,
                 displaced: object,
                 educational_special_needs: object,
                 debtor: object,
                 tuition_fees_up_to_date: object,
                 gender: object,
                 scholarship_holder: object,
                 age_at_enrollment: int,
                 international: object,
                 curricular_units_1st_sem_credited: int,
                 curricular_units_1st_sem_enrolled: int,
                 curricular_units_1st_sem_evaluations: int,
                 curricular_units_1st_sem_approved: int,
                 curricular_units_1st_sem_grade: float,
                 curricular_units_1st_sem_without_evaluations: int,
                 curricular_units_2nd_sem_credited: int,
                 curricular_units_2nd_sem_enrolled: int,
                 curricular_units_2nd_sem_evaluations: int,
                 curricular_units_2nd_sem_approved: int,
                 curricular_units_2nd_sem_grade: float,
                 curricular_units_2nd_sem_without_evaluations: int,
                 unemployment_rate: float,
                 inflation_rate: float,
                 gdp: float
                 ):
        self.id = id
        self.marital_status = marital_status
        self.application_mode = application_mode
        self.application_order = application_order
        self.course = course
        self.attendance = attendance
        self.previous_qualification = previous_qualification
        self.previous_qualification_grade = previous_qualification_grade
        self.nationality = nationality
        self.mother_qualification = mother_qualification
        self.father_qualification = father_qualification
        self.mother_occupation = mother_occupation
        self.father_occupation = father_occupation
        self.admission_grade = admission_grade
        self.displaced = displaced
        self.educational_special_needs = educational_special_needs
        self.debtor = debtor
        self.tuition_fees_up_to_date = tuition_fees_up_to_date
        self.gender = gender
        self.scholarship_holder = scholarship_holder
        self.age_at_enrollment = age_at_enrollment
        self.international = international
        self.curricular_units_1st_sem_credited = curricular_units_1st_sem_credited
        self.curricular_units_1st_sem_enrolled = curricular_units_1st_sem_enrolled
        self.curricular_units_1st_sem_evaluations = curricular_units_1st_sem_evaluations
        self.curricular_units_1st_sem_approved = curricular_units_1st_sem_approved
        self.curricular_units_1st_sem_grade = curricular_units_1st_sem_grade
        self.curricular_units_1st_sem_without_evaluations = curricular_units_1st_sem_without_evaluations
        self.curricular_units_2nd_sem_credited = curricular_units_2nd_sem_credited
        self.curricular_units_2nd_sem_enrolled = curricular_units_2nd_sem_enrolled
        self.curricular_units_2nd_sem_evaluations = curricular_units_2nd_sem_evaluations
        self.curricular_units_2nd_sem_approved = curricular_units_2nd_sem_approved
        self.curricular_units_2nd_sem_grade = curricular_units_2nd_sem_grade
        self.curricular_units_2nd_sem_without_evaluations = curricular_units_2nd_sem_without_evaluations
        self.unemployment_rate = unemployment_rate
        self.inflation_rate = inflation_rate
        self.gdp = gdp

    def get_data_as_df(self):
        """ 
        This method saves data into dictionary and transforms to DataFrame format for prediction
        """
        try:
            custom_input_dict = {
                'id': [self.id],
                'Marital status': [self.marital_status],
                'Application mode': [self.application_mode],
                'Application order': [self.application_order],
                'Course': [self.course],
                'Daytime/evening attendance': [self.attendance],
                'Previous qualification': [self.previous_qualification],
                'Previous qualification (grade)': [self.previous_qualification_grade],
                'Nationality': [self.nationality],
                "Mother's qualification": [self.mother_qualification],
                "Father's qualification": [self.father_qualification],
                "Mother's occupation": [self.mother_occupation],
                "Father's occupation": [self.father_occupation],
                'Admission grade': [self.admission_grade],
                'Displaced': [self.displaced],
                'Educational special needs': [self.educational_special_needs],
                'Debtor': [self.debtor],
                'Tuition fees up to date': [self.tuition_fees_up_to_date],
                'Gender': [self.gender],
                'Scholarship holder': [self.scholarship_holder],
                'Age at enrollment': [self.age_at_enrollment],
                'International': [self.international],
                'Curricular units 1st sem (credited)': [self.curricular_units_1st_sem_credited],
                'Curricular units 1st sem (enrolled)': [self.curricular_units_1st_sem_enrolled],
                'Curricular units 1st sem (evaluations)': [self.curricular_units_1st_sem_evaluations],
                'Curricular units 1st sem (approved)': [self.curricular_units_1st_sem_approved],
                'Curricular units 1st sem (grade)': [self.curricular_units_1st_sem_grade],
                'Curricular units 1st sem (without evaluations)': [self.curricular_units_1st_sem_without_evaluations],
                'Curricular units 2nd sem (credited)': [self.curricular_units_2nd_sem_credited],
                'Curricular units 2nd sem (enrolled)': [self.curricular_units_2nd_sem_enrolled],
                'Curricular units 2nd sem (evaluations)': [self.curricular_units_2nd_sem_evaluations],
                'Curricular units 2nd sem (approved)': [self.curricular_units_2nd_sem_approved],
                'Curricular units 2nd sem (grade)': [self.curricular_units_2nd_sem_grade],
                'Curricular units 2nd sem (without evaluations)': [self.curricular_units_2nd_sem_without_evaluations],
                'Unemployment rate': [self.unemployment_rate],
                'Inflation rate': [self.inflation_rate],
                'GDP': [self.gdp]
            }
            return pd.DataFrame(custom_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
