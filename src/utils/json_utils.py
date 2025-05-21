import json
import os
from src.logging.Logger  import logger

def load_json(json_path: str) -> list:
    json_data = None
    with open(json_path, "r") as f:
        json_data = json.load(f)
    return json_data


def save_to_json(json_path: str, summary_records: list[dict]):
    """
    Saves JSON formatted data to disk at specified path

    Args:
        json_path (str): Path to the JSON file
        summary_records (list[dict{}]): JSON formatted data

    Returns:
        None
    """
    logger.log("Saving JSON file")
    with open(json_path, "w") as f:
        json.dump(summary_records, f, indent=4)
    logger.log("JSON file saved")

def json_exists(full_path: str) -> bool:
    """
    Checks if JSON file exists, returns True if so else False

    Args:
        full_path (str): Path to JSON file

    Returns:
        (bool): State of file existing
    """
    if os.path.isfile(full_path):
        return True
    else:
        return False