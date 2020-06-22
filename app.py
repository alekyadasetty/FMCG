# -*- coding: utf-8 -*-
""" 
Created on Wed Apr  1 14:53:49 2020

@author: surya
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app=Flask(__name__,template_folder='template',static_folder='static')


model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
   
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 4)

    return render_template('index.html', prediction_text='Achieved Sales Value is  $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
