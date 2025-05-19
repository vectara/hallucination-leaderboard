from datetime import datetime

class Logger:
    """
    Basic Logger Class for debugging

    Attributes:
        file_name (str): full file name

    Methods:
        append_log(log): appends a log to the file
    
    """
    def __init__(self, log_name="live_log"):
        self.file_name = log_name + ".txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.file_name, 'a') as f:
            intro_msg = f"--- {log_name} started on {timestamp} ---\n"
            f.write(intro_msg)

    def add_log(self, log: str):
        """
        Given a log, adds a timestamp to it and appends it to the file

        Args:
            log(str): log message

        Returns:
            None
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.file_name, 'a') as f:
            f.write(f"{timestamp} - {log}\n")