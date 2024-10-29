import logging
from logging.handlers import RotatingFileHandler

from flask import Flask
from flask import request, redirect
import pandas as pd
from .serve_model import predict

from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

from Dash_app.dashboard import create_dash_app, create_home_app, create_predicitons_app,create_model_app

server = Flask(__name__)
dash_app = create_dash_app(server)
home_app = create_home_app(server)
prediction_app = create_predicitons_app(server)
model_app = create_model_app(server)

file_handler = RotatingFileHandler('app.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
file_handler.setLevel(logging.DEBUG)

server.logger.addHandler(file_handler)


@server.post('/predict')
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

    server.logger.debug("Got the required data from user.")

    prediction = predict(data)[0]
    server.logger.info(f"The model successfully predicted with a value of {prediction}")
    if prediction == 1:
        result = 'Fraudulent'
    else:
        result = 'Non fraudulent'
    
    return result


@server.route('/')
def home():
    return redirect('/home')


@server.route('/dashboard/')
def render_dashboard():
    return redirect('/dashboard')


@server.route('/data')
def read_data():
    df = pd.read_csv('data/Fraud_country_Data.csv')
    return df.to_json(orient='records')


@server.route('/make-predicitons')
def make_predictions():
    return redirect('/make-predicitons')


@server.route('/model-performance')
def model_performance():
    return redirect('/model-performance')