from flask import Flask, request, render_template
from src.pipeline import CustomData, PredictionPipeline


application=Flask(__name__)

app=application


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method=='POST':
        data = CustomData(
            id=int(request.form.get('id')),
            marital_status=request.form.get('marital_status'),
            application_mode=int(request.form.get('application_mode')),
            application_order=int(request.form.get('application_order')),
            course=request.form.get('course'),
            attendance=request.form.get('attendance'),
            previous_qualification=request.form.get('previous_qualification'),
            previous_qualification_grade=float(request.form.get('previous_qualification_grade')),
            nationality=request.form.get('nationality'),
            mother_qualification=request.form.get("mother_qualification"),
            father_qualification=request.form.get("father_qualification"),
            mother_occupation=request.form.get("mother_occupation"),
            father_occupation=request.form.get("father_occupation"),
            admission_grade=float(request.form.get('admission_grade')),
            displaced=request.form.get('displaced'),
            educational_special_needs=request.form.get('educational_special_needs'),
            debtor=request.form.get('debtor'),
            tuition_fees_up_to_date=request.form.get('tuition_fees_up_to_date'),
            gender=request.form.get('gender'),
            scholarship_holder=request.form.get('scholarship_holder'),
            age_at_enrollment=int(request.form.get('age_at_enrollment')),
            international=request.form.get('international'),
            curricular_units_1st_sem_credited=int(request.form.get('curricular_units_1st_sem_credited')),
            curricular_units_1st_sem_enrolled=int(request.form.get('curricular_units_1st_sem_enrolled')),
            curricular_units_1st_sem_evaluations=int(request.form.get('curricular_units_1st_sem_evaluations')),
            curricular_units_1st_sem_approved=int(request.form.get('curricular_units_1st_sem_approved')),
            curricular_units_1st_sem_grade=float(request.form.get('curricular_units_1st_sem_grade')),
            curricular_units_1st_sem_without_evaluations=int(request.form.get('curricular_units_1st_sem_without_evaluations')),
            curricular_units_2nd_sem_credited=int(request.form.get('curricular_units_2nd_sem_credited')),
            curricular_units_2nd_sem_enrolled=int(request.form.get('curricular_units_2nd_sem_enrolled')),
            curricular_units_2nd_sem_evaluations=int(request.form.get('curricular_units_2nd_sem_evaluations')),
            curricular_units_2nd_sem_approved=int(request.form.get('curricular_units_2nd_sem_approved')),
            curricular_units_2nd_sem_grade=float(request.form.get('curricular_units_2nd_sem_grade')),
            curricular_units_2nd_sem_without_evaluations=int(request.form.get('curricular_units_2nd_sem_without_evaluations')),
            unemployment_rate=float(request.form.get('unemployment_rate')),
            inflation_rate=float(request.form.get('inflation_rate')),
            gdp=float(request.form.get('gdp'))
        )
        #transfrom data into DataFrame get ready for prediction
        predict_df= data.get_data_as_df()
        print(predict_df)
        print("Before Prediction")
        
        prediction_pipeline= PredictionPipeline()
        print("Mid Prediction")
        
        result=prediction_pipeline.predict(predict_df)
        print("after Prediction")
        
        return render_template('home.html', result=result[0])
    else: 
        return render_template('home.html')
    
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)