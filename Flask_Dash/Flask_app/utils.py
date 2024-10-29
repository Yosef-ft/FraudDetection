import pandas as pd
import joblib
import plotly.express as px

import seaborn as sns
import matplotlib.pyplot as plt

class Utils:

    def time_features(self, data: pd.DataFrame, date_col: str):
        '''
        This fuction is used to extract time based features from data

        Parameter:
        ---------
            data(pd.DataFrame):
            date_col(str): column name contining date variable
        '''

        data['Hour'] = data[date_col].dt.hour
        data['quarter'] =   data[date_col].dt.quarter
        data['month'] =     data[date_col].dt.month
        data['dayofyear'] = data[date_col].dt.dayofyear
        data['DayOfWeek'] = data[date_col].dt.day_of_week
        data['weekdays'] =  data['DayOfWeek'].apply(lambda x: x < 6)
        data['weekends'] =  data['DayOfWeek'].apply(lambda x: x >= 6)   

        return data    

    def preprocess(self, data):
        
        # Getting time features
        data['purchase_time'] = pd.to_datetime(data['purchase_time'])
        data['signup_time'] = pd.to_datetime(data['signup_time'])

        data = self.time_features(data, 'purchase_time')
        data['velocity'] = (data['purchase_time'] - data['signup_time']).dt.seconds 

        data.drop(columns='signup_time', inplace=True)
        data.drop(columns='purchase_time', inplace=True)


        # Encoding country column
        with open('Encoder/Countries.txt', 'r') as file:
            for line in file:
                line =line.strip()
                col_name = f'Country_{line}'
                data[col_name] = data['country'].apply(lambda x: True if x == line else False)
        
        data.drop(columns='country', inplace=True)
        

        # Encodeing sex column
        data['sex_F'] = data['sex'].apply(lambda x: True if x == 'F' else False)
        data['sex_M'] = data['sex'].apply(lambda x: True if x == 'M' else False)
        data.drop(columns='sex', inplace=True)
        
        
        # Encoding browser column
        with open('Encoder/browser.txt', 'r') as file:
            for line in file:
                line =line.strip()
                data[f'browser_{line}'] = data['browser'].apply(lambda x: True if x == line else False)
        
        data.drop(columns='browser', inplace=True)

        # Encoding source column
        with open('Encoder/source.txt', 'r') as file:
            for line in file:
                line =line.strip()
                data[f'source_{line}'] = data['source'].apply(lambda x: True if x == line else False)
        
        data.drop(columns='source', inplace=True)        

        label_encoder = joblib.load('Encoder/label_encoder.pkl')
        for column in data.select_dtypes(include=['bool']).columns:
            data[column] = label_encoder.fit_transform(data[column])

        column_order = []
        with open ('Encoder/modelFit_order.txt', 'r') as file:
            for line in file:
                line = line.strip()
                column_order.append(line)

        data = data[column_order]
        return data
    

    def trasaction_summary(self, data: pd.DataFrame, include_other: bool):
        '''
        This funciont is used to calculate the total transactions, fraud cases, and fraud percentages

        Parameter:
        ---------
            data(pd.DataFrame)
            include_other(bool): This parameter is used to include Other countries that are not identified by ID

        Return:
        -------
            total transactions, fraud cases, and fraud percentages
        '''

        if not include_other:
            data_counties = data.loc[data['Country'] != 'Other']
        else:
            data_counties = data

        total_transaction = data_counties.shape[0]
        fraud_cases = data_counties.loc[data_counties['class'] == 1].shape[0]
        fraud_percentage = round((fraud_cases / total_transaction) * 100, 2)

        return total_transaction, fraud_cases, fraud_percentage


    def plot_evaluate_neurons(self, metrics: str):
        '''
        This function is used to evaluate trained neural network models

        Parameter:
        ----------
            metrics(str): This is used to to identify which dataset to use
        
        '''

        data = pd.read_csv(f'../report/Fraud_models/{metrics}.csv')
        val_data = pd.read_csv(f'../report/Fraud_models/{metrics}.csv')
        
        # sns.lineplot(data=val_data, x='step', y='value', hue='Run', palette='pastel', linewidth=2.5, ax=axes[1])

        if 'val' in metrics:
            fig = px.line(val_data, x='step', y='value', line_group='Run', 
                    line_dash='Run', labels={'step': 'Epochs', 'value': metrics},
                    title=f'{metrics} vs. Epochs by model')
        else:
            fig = px.line(data, x='step', y='value', line_group='Run', 
                    line_dash='Run', labels={'step': 'Epochs', 'value': metrics},
                    title=f'{metrics} vs. Epochs by model')            

        
        fig.update_xaxes(title_text='Epochs', tickfont_size=12)
        fig.update_yaxes(title_text=metrics, tickfont_size=12)

        
        fig.update_layout(title_font_size=14)


        # axes[1].set_xlabel('Epochs', fontsize=12)
        # axes[1].set_ylabel(f'{metrics}', fontsize=12)
        # axes[1].set_title(f'val_{metrics} vs. Epochs by model', fontsize=14)

        return fig    

    def plot_epochs(self, credit: bool = False):
        '''
        This function plots the stopped epochs for a given dataset:

        Parameter:
        ---------
            credit(bool): This is used to to identify which dataset to use
        '''
        if credit:
            epochs = pd.read_csv(f'../report/creditCard_Models/stopped_epoch.csv')
        else: 
            epochs = pd.read_csv(f'../report/Fraud_Models/stopped_epoch.csv')
        epochs.dropna(inplace=True)
        epochs.sort_values('stopped_epoch', inplace= True)

        fig = px.bar(epochs, x='Run', y='stopped_epoch', barmode='group', labels={'Run': 'Models', 'stopped_epoch': 'Stopped Epochs'},
                    title='Stopped epoch per model')
        
        return fig


    def plot_evaluation_model(self, metrics):
        '''
        This is used to evaluate trained scikit learn modles

        Parameter:
        ----------
            metrics(str): auc_roc, precision, recall, f1
        '''
        
        auc_roc = pd.read_csv('../report/Fraud_Models/AUC-ROC.csv')
        auc_roc.dropna(inplace=True)

        precision = pd.read_csv('../report/Fraud_Models/Precision (1).csv')
        precision.dropna(inplace=True)

        recall = pd.read_csv('../report/Fraud_Models/Recall (1).csv')
        recall.dropna(inplace=True)

        f1 = pd.read_csv('../report/Fraud_Models/f1.csv')
        f1.dropna(inplace=True)    

        if metrics == 'auc_roc':
            fig = px.bar(auc_roc, x='AUC-ROC', y='Run', barmode='group', labels={'Run': 'Models', 'AUC-ROC': 'AUC-ROC'},
                    title='AUC-ROC per model')
            
        elif metrics == 'precision':
            fig = px.bar(precision, x='Precision', y='Run', barmode='group', labels={'Run': 'Models', 'Precision': 'Precision'},
                    title='Precision per model')
            
        elif metrics == 'recall':
            fig = px.bar(recall, x='Recall', y='Run', barmode='group', labels={'Run': 'Models', 'Recall': 'Recall'},
                    title='Recall per model')  

        elif metrics == 'f1':
            fig = px.bar(f1, x='f1', y='Run', barmode='group', labels={'Run': 'Models', 'f1': 'F1 score'},
                    title='F1 score per model')                   

        
        return fig