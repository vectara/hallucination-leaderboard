from enum import Enum

from src.data_struct.config_model import ExecutionMode, InteractionMode
from src.constants import (
    GET_SUMM, GET_JUDGE, GET_RESULTS,
    TEST_DATA_PATH, LB_DATA_PATH
)

class SummaryError(str, Enum):
    MODEL_FAILED_TO_RETURN_OUTPUT = "MODEL FAILED TO RETURN ANY OUTPUT"
    MODEL_RETURNED_NON_STRING_TYPE_OUTPUT = (
        "DID NOT RECIEVE A STRING TYPE FROM OUTPUT"
    )
    EMPTY_SUMMARY = (
        "THIS SUMMARY IS EMPTY, THIS IS THE DEFAULT VALUE A SUMMARY "
        "VARIABLE GETS. A REAL SUMMARY WAS NOT ASSIGNED TO THIS VARIABLE."
    )
    INCOMPLETE_THINK_TAG = "FOUND <think> WITH NO CLOSING </think>"

class ModelInstantiationError(str, Enum):
    MODEL_NOT_SUPPORTED = "Model {model_name} by company {company} is not yet supported for {execution_mode} execution."
    MISSING_SETUP = "Be sure to have a `setup` and a `teardown` method in the model class {class_name}. See `__enter__` and `__exit__` methods of `AbstractLLM` for more information."

OUTPUT_DIR="output"

# Runtime config
CONFIG = {
    "pipeline": [GET_SUMM, GET_JUDGE, GET_RESULTS],
    "overwrite": True,
    "source_article_path": TEST_DATA_PATH,
    "temperature": 0.0, 
    "max_tokens": 1024,
    "simulation_count": 1, # NO EFFECT ON PROGRAM ATM
    "sample_count": 1, # NO EFFECT ON PROGRAM ATM
    "LLMs_to_eval":
    [
        # {
        #     "company": "rednote",
        #     "params": {
        #         "model_name": "rednote-hilab/dots.llm1.base",
        #         "execution_mode": ExecutionMode.LOCAL,
        #         "interaction_mode": InteractionMode.COMPLETION,
        #         "temperature": 0.001 # Doesn't accept 0.0
        #     }
        # }
        # ,
        {
            "company": "mistralai",
            "params": {
                "model_name": "mistral-small",
                "execution_mode": ExecutionMode.CLIENT,
                "interaction_mode": InteractionMode.CHAT,
                "date_code": "2506"
            }
        }
        # ,
        # {
        #     "company": "anthropic",
        #     "params": {
        #         "model_name": "claude-opus-4",
        #         "execution_mode": ExecutionMode.CLIENT,
        #         "interaction_mode": InteractionMode.CHAT,
        #         "date_code": "20250514"
        #     }
        # }
        # ,
        # {
        #     "company": "openai",
        #     "params": {
        #         "model_name": "gpt-4.1",
        #         "execution_mode": ExecutionMode.CLIENT,
        #     }
        # }
        #,
        # {
        #     "company": "anthropic",
        #     "params": {
        #         "model_name": "claude-sonnet-4",
        #         "execution_mode": ExecutionMode.CLIENT,
        #         "date_code": "20250514"
        #     }
        # }
    ]

}
