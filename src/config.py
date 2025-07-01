from src.data_struct.config_model import ExecutionMode, InteractionMode
from src.constants import (
    GET_SUMM, GET_JUDGE, GET_RESULTS,
    TEST_DATA_PATH, LB_DATA_PATH
)


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
            "company": "anthropic",
            "params": {
                "model_name": "claude-opus-4",
                "execution_mode": ExecutionMode.CLIENT,
                "interaction_mode": InteractionMode.CHAT,
                "date_code": "20250514"
            }
        }
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
