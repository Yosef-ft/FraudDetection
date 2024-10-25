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
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import mlflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Conv1D, MaxPooling1D, Flatten,Reshape, SimpleRNN, RNN
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping 
from tensorflow.keras.metrics import Accuracy, Precision, F1Score, Recall
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

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
    

    def split_data(self, data:pd.DataFrame):

        try:
            X_fraud = data.drop(columns=["class", "purchase_time", "signup_time", "user_id", "device_id", "ip_address"])
            y_fraud = data["class"]

            le = LabelEncoder()
            for column in X_fraud.select_dtypes(include=['bool']).columns:
                X_fraud[column] = le.fit_transform(X_fraud[column])
            
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
    
    def param_identifier(self, model, credit = False):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")


        model_name = type(model).__name__

        if model_name == 'LogisticRegression':
            params = {
                'solver' : ['saga', 'liblinear'],
                'penalty': ['l1', 'l2'],
                'C': [0.01, 0.1, 1, 10, 100],                  
                'max_iter': [50, 100, 150],             
            }
            run_name = "Fraud_LR"
            artifact_path = "Fraud_LR_path"

        elif model_name == "DecisionTreeClassifier":
            params = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2'],
                'splitter': ['best', 'random'],
                'max_leaf_nodes': [None, 5, 10, 20]
            }
            run_name = "Fraud_Dtree"
            artifact_path = "Fraud_Dtree_path"


        elif model_name == "RandomForestClassifier":
            params ={
                "n_estimators" : [50, 100, 150],
                "criterion" : ["gini", "entropy",],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],    
                'max_features': ['auto', 'sqrt', 'log2'],    
            }
            run_name = "Fraud_RF"
            artifact_path = "Fraud_RF_path"            

        elif model_name == "GradientBoostingClassifier":
            params = {
                "loss" : ['log_loss', 'exponential'],
                "learning_rate" : [0.01, 0.5, 0.1],
                "n_estimators" : [50, 100, 150],
                "criterion" : ['friedman_mse', 'squared_error'],
                'min_samples_split': [2, 5, 10],    
                'min_samples_leaf': [1, 2, 4],  
                'max_depth': [None, 5, 10, 15],  
                'max_features': ['sqrt', 'log2'],          
            }

            run_name = "Fraud_GBC"
            artifact_path = "Fraud_GBC_path"            

        elif model_name == "MLPClassifier":
            params = {
                "activation" : ['identity', 'logistic', 'tanh', 'relu'],
                "solver" : ['lbfgs', 'sgd', 'adam'],
                "learning_rate": ['constant', 'invscaling', 'adaptive'],
                "max_iter" : [150, 200, 250],
                "early_stopping" : [True]
            }

            run_name = "Fraud_MLP"
            artifact_path = "Fraud_MLP_path"            

        return params, run_name, artifact_path

    def best_model(self, X_train, y_train, X_val,y_val,model, credit=False):

        if credit:
            fraud_experiments = mlflow.set_experiment("creditCard_Models")
        else:
            fraud_experiments = mlflow.set_experiment("Fraud_Models")  

        logger.info(f"Start searching for the best Params of a {type(model).__name__} model")
        start_time = time.time()
        
        params, run_name, artifact_path = self.param_identifier(model=model)
        grid_search = RandomizedSearchCV(estimator=model, param_distributions=params, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        auc_roc = roc_auc_score(y_val, y_pred)

        best_params = grid_search.best_params_
        
        metrics = {"Precision" : precision, "accuracy" : accuracy, "f1" : f1,
                "Recall" : recall, "AUC-ROC" : auc_roc}    

        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(best_params)

            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(sk_model=best_model, input_example=X_val, artifact_path=artifact_path)

        
        end_time = time.time()

        logger.info(f"Searching for the best params for {type(model).__name__} took {round(end_time - start_time, 2)} seconds\n\n")


     
    def train_neurals(self, model_name: str,X_train, y_train, X_val,y_val ,credit):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        times = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cp = ModelCheckpoint(f'../Models/{model_name}_model_{times}.keras', save_best_only=True)
        callback = EarlyStopping(monitor='val_accuracy', patience=3)

        input_shape = (30,) if credit else (203,1)
        if model_name == 'LSTM' and credit:
            input_shape = (30,1)
        reshape = (30,1) if credit else (203,1)
        fraud_experiments =mlflow.set_experiment("creditCard_Models") if credit else mlflow.set_experiment("Fraud_Models")
        
        if model_name == 'LSTM':
            
            model = Sequential()
            model.add(InputLayer(input_shape=input_shape))
            model.add(LSTM(64, activation='tanh', return_sequences=False))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

            model.compile(
                loss=BinaryCrossentropy(),
                optimizer=Adam(learning_rate=0.01),
                metrics=['accuracy', 'precision', 'recall', F1Score()]
            )

        elif model_name == 'CNN':
            model = Sequential()
            model.add(InputLayer(input_shape=input_shape)) # fraud 203
            model.add(Reshape(reshape))
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid')) 

            model.compile(
                loss=BinaryCrossentropy(),
                optimizer=Adam(learning_rate=0.01),
                metrics=['accuracy', 'precision', 'recall', F1Score()]
            )

        elif model_name == 'RNN':
            model = Sequential()
            model.add(InputLayer(input_shape=input_shape))
            model.add(Reshape(reshape))
            model.add(SimpleRNN(50))
            model.add(Dense(8, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

            model.compile(
                loss=BinaryCrossentropy(),
                optimizer=Adam(learning_rate=0.01),
                metrics=['accuracy', 'precision', 'recall', F1Score()]
            )        

        start_time = time.time()
        logger.info(f"Start training {model_name} model...")

        mlflow.tensorflow.autolog()
        with mlflow.start_run(run_name=f'Fraud_{model_name}'):
            history = model.fit(X_train[:100], y_train[:100], validation_data=(X_val[:100], y_val[:100]),
                            epochs=50, callbacks=[cp, callback], batch_size = 128, verbose=0)

        end_time = time.time()

        logger.info(f"Training {model_name} took {round(end_time - start_time, 2)} seconds")
      


    def train_neural_models(self, X_train, y_train, X_val, y_val, credit = False):
        models = ['LSTM', 'CNN', 'RNN']

        if credit:
            logger.info(f"Start training {len(models)} models, with credit card dataset....")
        else:
            logger.info(f"Start training {len(models)} models, with ecommerce fraud dataset....")
        start_time = time.time()

        for model in models:
            self.train_neurals(model, X_train, y_train, X_val, y_val, credit)

        end_time = time.time()
        logger.info(f"Training 3 different models took {round(end_time - start_time, 2)} seconds")      


    def plot_evaluate_neurons(self, credit: bool, metrics: str):
        if credit:
            data = pd.read_csv(f'../report/creditCard_Models/{metrics}.csv')
            val_data = pd.read_csv(f'../report/creditCard_Models/val_{metrics}.csv')
        else:
            data = pd.read_csv(f'../report/Fraud_models/{metrics}.csv')
            val_data = pd.read_csv(f'../report/Fraud_models/val_{metrics}.csv')
        
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18,8))

        sns.set_style("whitegrid")

        sns.lineplot(data=data, x='step', y='value', hue='Run', palette='pastel', linewidth=2.5, ax=axes[0])
        sns.lineplot(data=val_data, x='step', y='value', hue='Run', palette='pastel', linewidth=2.5, ax=axes[1])

        axes[0].set_xlabel('Epochs', fontsize=12)
        axes[0].set_ylabel(f'{metrics}', fontsize=12)
        axes[0].set_title(f'{metrics} vs. Epochs by model', fontsize=14)

        axes[1].set_xlabel('Epochs', fontsize=12)
        axes[1].set_ylabel(f'{metrics}', fontsize=12)
        axes[1].set_title(f'val_{metrics} vs. Epochs by model', fontsize=14)

        plt.show();    