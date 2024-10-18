import time
import logging

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

from Utils import DataUtils
from Logger import LOGGER
logger = LOGGER

class Plots:

    def __init__(self):
        pass

    def visualize_outliers(self, data: pd.DataFrame):
        '''
        This funcions helps in visualizing outliers using boxplot
        '''        
        logger.debug('Plot to visualize outliers...')
        try:
            numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
            num_cols = len(numerical_cols)
            nrows = num_cols // 5 + num_cols % 5
            
            start_time = time.time()

            fig, axes = plt.subplots(nrows=nrows, ncols=5, figsize=(12,8))
            axes = axes.flatten()

            for i, col in enumerate(numerical_cols):
                sns.boxplot(y=data[col], ax=axes[i])
                axes[i].set_title(col)

            end_time = time.time()
            logger.info(f'It took {end_time - start_time:.2f} seconds to plot outliers')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f"Error while plotting ouliers: {e}")


    def num_univariant_visualization(self, data: pd.DataFrame):
        '''
        This funcions helps in visualizing histograms for numeric columns
        '''
        numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
        num_cols = len(numerical_cols)
        nrows = num_cols // 5 + num_cols % 5
        logger.debug('Plot to visualize univariant columns...')
        try:
            start_time = time.time()

            fig, axes = plt.subplots(nrows=nrows, ncols=5, figsize=(20,12))
            axes = axes.flatten()

            for i, col in enumerate(numerical_cols):
                sns.histplot(data=data[col], ax=axes[i], kde=True)
                axes[i].set_title(col)

            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)

            end_time = time.time()
            logger.info(f'It took {end_time - start_time:.2f} seconds to plot histograms')            

            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f"Error while plotting histograms: {e}")



    def plot_ecdf (self, column, title= 'ECDF Plot', xlabel = 'Purchase vlaue', ylabel = 'Percentage'):
        '''
        Funtion to plot ecdf taking a column of data as input

        Parameters:
            column(pd.Series)
        '''
        try:

            cdf = ECDF(column)
            df = pd.DataFrame({'x': cdf.x, 
                            'y': cdf.y})
            sns.lineplot(data=df, x ='x', y='y', marker = '.',  linewidth=2)

            plt.title(title, fontweight='bold')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.margins(0.02)       

        except Exception as e:
            logger.error(f"Error while plotting ECDF: {e}")        