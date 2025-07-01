from src.data_struct.config_model import ExecutionMode, InteractionMode

# Files
TEST_DATA_PATH="datasets/test_articles.csv"
LB_DATA_PATH="datasets/leaderboard_dataset_revised.csv"
OUTPUT_DIR="output"
TEST_JUDGEMENTS_DATA = "datasets/test_analytics/judgements.jsonl"
TEST_RESULTS_DATA = "datasets/test_analytics/stats.jsonl"
TEST_SUMMARIES_DATA = "datasets/test_analytics/summaries.jsonl"

# Protocols
GET_SUMM = "get_summaries"
GET_JUDGE = "get_judgements"
GET_RESULTS = "get_results"

# Runtime config
CONFIG = {
    "pipeline": [GET_SUMM, GET_JUDGE, GET_RESULTS],
    "overwrite": True,
    "input_file": TEST_DATA_PATH,
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
