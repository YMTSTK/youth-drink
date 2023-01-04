from flask import Flask,request,jsonify
import numpy as np
from sklearn import preprocessing
import pandas as pd
import pickle

model = pickle.load(open('Random_forest.sav','rb'))
app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def predict():

    a1=request.form.get('r1')
    a2=request.form.get('r2')
    a3=request.form.get('r3')
    a4=request.form.get('r4')
    a5=request.form.get('r5')
    a6=request.form.get('r6')
    a7=request.form.get('r7')
    a8=request.form.get('r8')
    a9=request.form.get('r9')

    list = [a1,a2,a3,a4,a5,a6,a7,a8,a9]

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    li = le.fit_transform(list)

    result = model.predict([li])[0]

    return jsonify({'error':'false','habit': str(result),'message':'Success'})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
