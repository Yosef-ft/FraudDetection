from flask_app import app
from flask import request
import pandas as pd
from .serve_model import predict
from .utils import Utils

@app.post('/predict')
def prediction():
    
    data = pd.DataFrame({
        'purchase_value' : [float(request.form['purchase_value'])],
        'age' : [int(request.form['age'])],
        'purchase_time' : [request.form['purchase_time']],
        'signup_time' : [request.form['signup_time']],
        'source' : [request.form['source']],
        'browser' : [request.form['browser']],
        'sex' : [request.form['sex']],
        'trasaction_frequency' : [request.form['trasaction_frequency']],
        'country' : [request.form['country']],
    })


    prediction = predict(data)[0]
    return str(prediction)
