import logging
from datetime import datetime

def setup_logger(log_name="log") -> logging.Logger:
    """
    Sets up the logger and logs a message of when the logger was started

    Args:
        log_name (str): name of the file to store logs. Defaults to log.txt

    Returns:
        logging.Logger: the logger object
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(f"{log_name}.txt")
        formatter = logging.Formatter(
            "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        intro_msg = (
            f"--- {log_name} started on "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---"
        )
        logger.critical(intro_msg)

    return logger

logger = setup_logger()