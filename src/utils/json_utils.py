import json
import os
from src.logging.Logger  import logger
from pydantic import BaseModel

"""
Functions for handling JSON files

Functions:
    load_json(json_path)
    save_to_json(json_path, records)
    json_exists(json_path)
"""

def load_json(json_path: str) -> list:
    """
    Load a JSON file at the given path

    Args:
        json_path (str): Path to JSON file

    Returns:
        (list or dict): JSON formatted data
    """
    json_data = None
    with open(json_path, "r") as f:
        json_data = json.load(f)
    return json_data

def save_to_jsonl(jsonl_path: str, records: list[BaseModel]):
    # TODO: Documentation
    """
    """
    logger.log("Saving JSONL file")
    with open(jsonl_path, "w") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    logger.log("JSONL file saved")


def append_to_jsonl(json_path: str, records: list[dict]):
    # TODO: Documentation
    """
    """
    pass

def save_bm_to_json(json_path: str, record: BaseModel):
    """
    Saves BaseModel objects to JSON formatted data to disk at specified path

    Args:
        json_path (str): Path to the JSON file
        records (list[BaseModel]): JSON formatted data

    Returns:
        None
    """
    bm_dict = record.model_dump()
    with open(json_path, "w") as f:
        json.dump(bm_dict, f, indent=4)


def save_to_json(json_path: str, records: list[dict]):
    """
    Saves JSON formatted data to disk at specified path

    Args:
        json_path (str): Path to the JSON file
        records (list[dict{}]): JSON formatted data

    Returns:
        None
    """
    logger.log("Saving JSON file")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=4)
    logger.log("JSON file saved")

def json_exists(json_path: str) -> bool:
    """
    Checks if JSON file exists, returns True if so else False

    Args:
        full_path (str): Path to JSON file

    Returns:
        (bool): State of file existing
    """
    if os.path.isfile(json_path):
        return True
    else:
        return False