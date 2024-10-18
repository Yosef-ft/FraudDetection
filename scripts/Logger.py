import os
import sys
import logging

def setup_logger():
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


LOGGER = setup_logger()