
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import uvicorn
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add middleware to enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify specific origins instead of "*" if needed
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    
    age : int
    sex : int
    chest_pain_type : int
    resting_bp : int
    cholesterol : int
    fasting_blood_sugar : float
    resting_ecg : float
    max_heart_rate : int 
    exercise_angina : int   
    oldpeak : float
    ST_slope : int  
        
# loading the saved model
heartdisease_model = pickle.load(open('heart_model.sav', 'rb'))

@app.post('/heartdisease_prediction')
def heartdisease_pred(input_parameters : model_input):
    
    input_data = input_parameters.model_dump_json()
    input_dictionary = json.loads(input_data)
    age = input_dictionary['age']
    sex = input_dictionary['sex']
    cpt = input_dictionary['chest_pain_type']
    bp = input_dictionary['resting_bp']
    cho = input_dictionary['cholesterol']
    sugar = input_dictionary['fasting_blood_sugar']
    ecg = input_dictionary['resting_ecg']
    rate = input_dictionary['max_heart_rate']
    angina = input_dictionary['exercise_angina']
    peak = input_dictionary['oldpeak']
    st = input_dictionary['ST_slope']

    
    
    # input_list = [age, sex, cpt, bp, cho, sugar, ecg, rate, angina, peak, st]
    data = np.array([[age, sex, cpt, bp, cho, sugar, ecg, rate, angina, peak, st]])
    # arr = np.array([[data1, data2, data3, data4]])
    prediction = heartdisease_model.predict(data)
    # prediction = diabetes_model.predict([input_list])
    
    if (prediction[0] == 0):
        
        return 'The person does not have heart disease!'
    else:
        return 'The person has a heart disease!'
    
    
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1')


