from datetime import datetime

class Logger:
    """
    Basic Logger Class for debugging

    Attributes:
        file_name (str): full file name

    Methods:
        log(msg): appends a log to the file
    
    """
    def __init__(self, log_name="log"):
        self.file_name = log_name + ".txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.file_name, 'a') as f:
            intro_msg = f"--- {log_name} started on {timestamp} ---\n"
            f.write(intro_msg)

    def log(self, msg: str):
        """
        Given a message, adds a timestamp to it and appends it to the file

        Args:
            msg(str): log message

        Returns:
            None
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.file_name, 'a') as f:
            f.write(f"{timestamp} - {msg}\n")

logger = Logger()