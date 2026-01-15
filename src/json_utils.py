"""Utility functions for JSON and JSONL file operations.

This module provides helper functions for reading and writing JSON and JSONL
(JSON Lines) files, with built-in support for Pydantic BaseModel serialization.
Used throughout the pipeline for persisting summaries, judgments, and statistics.

Functions:
    load_json: Load and parse a JSON file.
    save_to_jsonl: Write a list of records to a JSONL file.
    append_record_to_jsonl: Append a single record to a JSONL file.
    save_to_json: Write data to a JSON file with formatting.
"""

import json
import os
from typing import Union, List, Any

from pydantic import BaseModel

from . Logger import logger

def load_json(json_path: str) -> list:
    """Load and parse a JSON file from disk.

    Reads a JSON file and deserializes its contents into Python data
    structures (lists or dictionaries).

    Args:
        json_path: Absolute or relative path to the JSON file.

    Returns:
        The parsed JSON data as a list or dictionary, depending on
        the file's root structure.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    json_data = None
    with open(json_path, "r") as f:
        json_data = json.load(f)
    return json_data

def save_to_jsonl(jsonl_path: str, records: list[BaseModel]):
    """Write a list of Pydantic models to a JSONL file.

    Serializes each BaseModel object to JSON and writes one record per
    line in JSON Lines format. Overwrites the file if it already exists.

    Args:
        jsonl_path: Path where the JSONL file will be written.
        records: List of Pydantic BaseModel instances to serialize.

    Note:
        Uses Pydantic's model_dump_json() for serialization, ensuring
        proper handling of nested models and custom field serializers.
    """
    logger.info("Saving JSONL file")
    with open(jsonl_path, "w") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    logger.info("JSONL file saved")

def append_record_to_jsonl(jsonl_path: str, record: BaseModel):
    """Append a single Pydantic model record to a JSONL file.

    Serializes the BaseModel object to JSON and appends it as a new line
    to an existing JSONL file. Creates the file if it does not exist.
    Used for incremental result persistence during pipeline execution.

    Args:
        jsonl_path: Path to the JSONL file to append to.
        record: Pydantic BaseModel instance to serialize and append.

    Note:
        This function is preferred over save_to_jsonl when results are
        generated incrementally, as it avoids keeping all records in
        memory and provides durability against crashes.
    """
    with open(jsonl_path, "a") as f:
        f.write(record.model_dump_json() + "\n")

def save_to_json(json_path: str, data: Any):
    """Write data to a formatted JSON file.

    Serializes Pydantic models or lists of models/dicts to a JSON file
    with pretty-printing (4-space indentation). Handles automatic
    conversion of BaseModel instances via model_dump().

    Args:
        json_path: Path where the JSON file will be written.
        data: Data to serialize. Must be a BaseModel instance or a list
            containing BaseModel instances or dictionaries.

    Raises:
        TypeError: If data is not a BaseModel or list of BaseModel/dict.

    Note:
        Unlike JSONL functions, this writes a single JSON document
        suitable for configuration files or aggregate results.
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