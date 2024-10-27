import pandas as pd
import joblib

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