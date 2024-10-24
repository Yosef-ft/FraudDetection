import os 
import sys
import random
import logging
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import tensorflow as tf

from Logger import LOGGER
logger = LOGGER

ANSI_ESC = {
    "PURPLE": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
    "BOLD": "\033[1m",
    "ITALICS" :"\033[3m"
}



class DataUtils:
    # def __init__(self):
    #     self.logger = self.setup_logger()

    def load_data(self, file_name: str)->pd.DataFrame:
        '''
        Load the file name from the data directory

        Parameters:
            file_name(str): name of the file

        Returns:
            pd.DataFrame
        '''
        logger.debug("Loading data from file...")
        try:
            start_time = time.time()

            data = pd.read_csv(f"../data/{file_name}", low_memory=False)

            end_time = time.time()
            logger.info(f"Loading {file_name} took {round(end_time - start_time, 2)} seconds\n\n")
            return data

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
        
    def set_seeds(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def data_info(self, data) -> pd.DataFrame:
        '''
        Provides detailed information about the data, including:
        - Percentage of missing values per column
        - Number of missing values per column
        - Data types of the columns

        It also highlights:
        - The total number of rows and columns in the dataset
        - Columns with the most missing values
        - Columns with more than 50% missing values

        Parameter:
        ----------
            data(pd.DataFrame): The dataset 
        
        Return:
        -------
            info_df(pd.DataFrame)
        '''
        
        missing_values = data.isna().sum()
        missing_percent = round(data.isna().mean() * 100, 2)
        data_types = data.dtypes
        
        info_df = pd.DataFrame({
            "Missing Values": missing_values,
            "Missing Percentage": missing_percent,
            "Data Types": data_types
        })


        info_df = info_df[missing_percent > 0]
        info_df = info_df.sort_values(by='Missing Percentage', ascending=False)

        max_na_col = list(info_df.loc[info_df['Missing Values'] == info_df['Missing Values'].max()].index)
        more_than_half_na = list(info_df.loc[info_df['Missing Percentage'] > 50].index)
        

        print(f"\n{ANSI_ESC['BOLD']}Dataset Overview{ANSI_ESC['ENDC']}")
        print(f"---------------------")
        print(f"- {ANSI_ESC['ITALICS']}Total rows{ANSI_ESC['ENDC']}: {data.shape[0]}")
        print(f"- {ANSI_ESC['ITALICS']}Total columns{ANSI_ESC['ENDC']}: {data.shape[1]}\n")

        duplicated_rows = int(data.duplicated().sum())
        if duplicated_rows == 0:
            print(f"{ANSI_ESC['GREEN']}No Duplicated data found in the dataset.{ANSI_ESC['ENDC']}\n")
        else:
             print(f"- {ANSI_ESC['RED']}Number of duplicated rows are{ANSI_ESC['ENDC']}: {duplicated_rows}")
             print(f"- {ANSI_ESC['RED']}Percentage of duplicated rows are{ANSI_ESC['ENDC']}: {round((duplicated_rows / data.shape[0]) * 100, 2)} %\n\n")
        
        if info_df.shape[0] > 0:
            print(f"{ANSI_ESC['BOLD']}Missing Data Summary{ANSI_ESC['ENDC']}")
            print(f"------------------------")
            print(f"- {ANSI_ESC['ITALICS']}Columns with missing values{ANSI_ESC['ENDC']}: {info_df.shape[0]}\n")
            
            print(f"- {ANSI_ESC['ITALICS']}Column(s) with the most missing values{ANSI_ESC['ENDC']}: `{', '.join(max_na_col)}`")
            print(f"- {ANSI_ESC['RED']}Number of columns with more than 50% missing values{ANSI_ESC['ENDC']}: `{len(more_than_half_na)}`\n")


            if more_than_half_na:
                print(f"{ANSI_ESC['BOLD']}Columns with more than 50% missing values:{ANSI_ESC['ENDC']}")
                for column in more_than_half_na:
                    print(f"   - `{column}`")
            else:
                print(f"{ANSI_ESC['GREEN']}No columns with more than 50% missing values.{ANSI_ESC['ENDC']}")
        else:
            print(f"{ANSI_ESC['GREEN']}No missing data found in the dataset.{ANSI_ESC['ENDC']}")

        print(f"\n{ANSI_ESC['BOLD']}Detailed Missing Data Information{ANSI_ESC['ENDC']}")
        print(info_df)

        return info_df

    def get_country(self, ip, Ip_data):
        '''
        This funtion is used to extract country from IP upper and lower bounds
        '''
        country = Ip_data.loc[(Ip_data['upper_bound_ip_address'] >= ip) & (Ip_data['lower_bound_ip_address'] <= ip), 'country']
        if not country.empty:
            return country.iloc[0]  
        return None
    

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