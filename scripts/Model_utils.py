import os 
import time
import sys
import pandas as pd
import sidetable as stb
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mlflow import MlflowClient
from pprint import pprint

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import mlflow

from Logger import LOGGER
logger = LOGGER

class ModelUtils:

    def setUp_mlflow(self):
        logger.info("Setting up Mlflow")
        client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

        experiment_description = (
            "This is the Fraud forcasting project. "
            "This experiment contains the produce models for predicting frauds for ecommerce and credit card fraud data."
        )


        experiment_tags = {
            "project_name": "Ecommerce-Fraud-Data-forecasting",
            "mlflow.note.content": experiment_description,
        }

        experiment_tags2 = {
            "project_name": "creditCard-Fraud-Data-forecasting",
            "mlflow.note.content": experiment_description,
        }

        existing_experiment = {}

        all_experiments = client.search_experiments()
        for experiment in all_experiments:
            project_name = experiment.tags.get('project_name')
            if project_name == 'Ecommerce-Fraud-Data-forecasting':
                existing_experiment['Ecommerce-Fraud-Data-forecasting'] = experiment

            elif project_name == "creditCard-Fraud-Data-forecasting":
                existing_experiment['creditCard-Fraud-Data-forecasting'] = experiment

        try:
            fraud_experiment = existing_experiment['Ecommerce-Fraud-Data-forecasting']
            logger.info("Found existing experiment name: Ecommerce-Fraud-Data-forecasting")

        except:
            fraud_experiment = client.create_experiment(
                name="Fraud_Models", tags=experiment_tags
            )
            logger.info("Creating new experiment name: Ecommerce-Fraud-Data-forecasting")


        try:
            creditCard_experiment = existing_experiment['creditCard-Fraud-Data-forecasting']
            logger.info("Found existing experiment name: creditCard-Fraud-Data-forecasting")
        except:
            creditCard_experiment = client.create_experiment(
                name="creditCard_Models", tags=experiment_tags2
            )
            logger.info("Creating new experiment name: creditCard-Fraud-Data-forecasting")

        return client, fraud_experiment, creditCard_experiment
    

    def split_data(self, data):

        try:
            X_fraud = data.drop(columns=["class", "purchase_time", "signup_time", "user_id", "device_id", "ip_address"])
            y_fraud = data["class"]
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_fraud, y_fraud = smote.fit_resample(X_fraud, y_fraud)
            logger.info('Splitting Fraud data...')
            X_train, X_val, y_train, y_val = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)        
        except:
            X_creditCard = data.drop(columns=["Class"])
            y_creditCard = data["Class"]
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_creditCard, y_creditCard = smote.fit_resample(X_creditCard, y_creditCard)
            logger.info('Splitting credit card data...')
            X_train, X_val, y_train, y_val = train_test_split(X_creditCard, y_creditCard, test_size=0.2, random_state=42)


        return X_train, X_val, y_train, y_val
    
    
