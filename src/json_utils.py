import json
import os
from typing import Union, List, Any

from pydantic import BaseModel

from . Logger  import logger

"""
Functions for handling JSON files

Functions:
    load_json(json_path)
    save_to_jsonl(jsonl_path, records)
    append_record_to_jsonl(jsonl_path, record)
    save_to_json(json_path, records)
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
    """
    Saves a list of BaseModel objects in JSONL format

    Args: 
        jsonl_path (str): path to file
        records (list[BaseModel]): records to be saved
    """
    logger.info("Saving JSONL file")
    with open(jsonl_path, "w") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    logger.info("JSONL file saved")

def append_record_to_jsonl(jsonl_path: str, record: BaseModel):
    """
    Appends a record to a jsonl file

    Args:
        jsonl_path (str): path to file
        record (BaseModel): record to be saved
    """
    with open(jsonl_path, "a") as f:
        f.write(record.model_dump_json() + "\n")

def save_to_json(json_path: str, data: Any):
    """
    Saves JSON formatted data to disk at specified path

    Args:
        json_path (str): Path to the JSON file
        records (list[dict{}]): JSON formatted data

    Returns:
        None
    """
    logger.info("Saving JSON file")
    json_data = None
    if isinstance(data, BaseModel):
        json_data = data.model_dump()
    elif isinstance(data, List):
        json_data = [
            item.model_dump() if isinstance(item, BaseModel) else item 
            for item in data
        ]
    else:
        raise TypeError("Data must be BaseModel or list of BaseModel or Dict")

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)
    logger.info("JSON file saved")