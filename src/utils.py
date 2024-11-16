# src/utils.py

import logging
import os

def setup_logging(log_file='logs/project.log'):
    """
    Sets up logging for the project.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to capture detailed logs
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
