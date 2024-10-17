import os 
import sys
import logging
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class DataUtils:
    def __init__(self):
        self.logger = self.setup_logger()

    
    def setup_logger(self):
        '''
        This function is used to setup logger for logging error and Info

        **Returns**:
        -----------
            a `logger` instance
        '''

        log_dir = os.path.join(os.path.split(os.getcwd())[0], 'logs')

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        log_file_info = os.path.join(log_dir, 'Info.log')
        log_file_error = os.path.join(log_dir, 'Error.log')

        info_handler = logging.FileHandler(log_file_info)
        error_handler = logging.FileHandler(log_file_error)
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s :: %(message)s',
                                      datefmt= '%Y-%m-%d %H:%M')

        info_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        info_handler.setLevel(logging.INFO)
        error_handler.setLevel(logging.ERROR)
        console_handler.setLevel(logging.DEBUG)

        logger = logging.getLogger(__name__)
        logger.addHandler(info_handler)
        logger.addHandler(error_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)

        return logger

        


    def load_data(self, file_name: str)->pd.DataFrame:
        '''
        Load the file name from the data directory

        Parameters:
            file_name(str): name of the file

        Returns:
            pd.DataFrame
        '''
        self.logger.debug("Loading data from file...")
        try:
            start_time = time.time()

            data = pd.read_csv(f"../data/{file_name}", low_memory=False)

            end_time = time.time()
            self.logger.info(f"Loading {file_name} took {round(end_time - start_time, 2)} seconds\n\n")
            return data

        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return None