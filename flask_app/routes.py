from flask import Flask
from flask import request
import pandas as pd
from .serve_model import predict
import logging
from logging.handlers import RotatingFileHandler


app = Flask(__name__)

file_handler = RotatingFileHandler('app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
file_handler.setLevel(logging.DEBUG)

app.logger.addHandler(file_handler)


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

    app.logger.debug("Got the required data from user.")

    prediction = predict(data)[0]
    app.logger.info(f"The model successfully predicted with a value of {prediction}")
    if prediction == 1:
        result = 'Fraudulent'
    else:
        result = 'Non fraudulent'
    
    return result
