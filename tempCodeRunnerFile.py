from flask import Flask,request,jsonify
import pickle
import numpy as np

model=pickle.load(open('model.pkl','rb'))
# read binary

app=Flask(__name__)
@app.route('/')
def home():
    return "hello world"

@app.route('/predict',methods=['POST'])
def predict():

    Pregnancies=request.form.get('Pregnancies')
    Glucose=request.form.get('Glucose')
    BloodPressure=request.form.get('BloodPressure')
    SkinThickness=request.form.get('SkinThickness')
    Insulin=request.form.get('Insulin')
    BMI=request.form.get('BMI')
    DiabetesPedigreeFunction=request.form.get('DiabetesPedigreeFunction')
    Age=request.form.get('Age')

    # result={'Pregnancies':Pregnancies,'Glucose':Glucose,'BloodPressure':BloodPressure,
    #         'SkinThickness':SkinThickness}
    
    input_query=np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,
                           BMI,DiabetesPedigreeFunction,Age]])
    
    result=model.predict(input_query)[0]
    
    return jsonify({'Diabetes':str(result)})
    




if __name__ == '__main__':
    app.run(debug=True)