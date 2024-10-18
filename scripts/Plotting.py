import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from Utils import DataUtils

class Plots:

    def __init__(self):
        utils = DataUtils()
        self.logger = utils.setup_logger()


    def visualize_outliers(self, data: pd.DataFrame):
        '''
        This funcions helps in visualizing outliers using boxplot
        '''        

        numerical_cols = data.select_dtypes(include=['float64', 'int64'])
        num_cols = len(numerical_cols)
        nrows = num_cols // 5 + num_cols % 5

        fig, axes = plt.subplots(nrows=nrows, ncols=5, figsize=(12,8))
        axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            sns.boxplot(y=data[col], ax=axes[i])
            axes[i].set_title(col)

        plt.tight_layout()
        plt.show()
        

    def num_univariant_visualization(self, data: pd.DataFrame):
        '''
        This funcions helps in visualizing histograms for numeric columns
        '''

        numerical_cols = data.select_dtypes(include=['float64', 'int64'])
        num_cols = len(numerical_cols)
        nrows = num_cols // 5 + num_cols % 5

        fig, axes = plt.subplots(nrows=nrows, ncols=5, figsize=(20,12))
        axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            sns.histplot(data[col], palette='pastel', ax=axes[i], kde=True)
            axes[i].set_title(col)

        for j in range(i+1, len(num_cols)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()


