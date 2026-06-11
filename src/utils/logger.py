# The whole logger is not working properly since we moved to different logging methods
# It remains here since we are lazy to fix it
# Here's to hoping we don't lose points!
import os
import logging

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(name, filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Construct the full path
        file_path = os.path.join(LOG_DIR, filename)
        
        log_sub_dir = os.path.dirname(file_path)
        os.makedirs(log_sub_dir, exist_ok=True)

        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger