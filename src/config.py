
# Files
TEST_DATA_PATH="datasets/test_articles.csv"
LB_DATA_PATH="datasets/leaderboard_dataset_revised.csv"
OUTPUT_DIR="output"
TEST_JUDGEMENTS_DATA = "datasets/test_analytics/judgements.jsonl"
TEST_RESULTS_DATA = "datasets/test_analytics/stats.json"
TEST_SUMMARIES_DATA = "datasets/test_analytics/summaries.jsonl"

# Protocols
GET_SUMM = "get_summary"
GET_JUDGE = "get_judgements"
GET_RESULTS = "get_results"


# Runtime config
CONFIG = {
    "pipeline": [GET_SUMM, GET_JUDGE, GET_RESULTS],
    "input_file": TEST_DATA_PATH,
    "temperature": 0.0, 
    "max_tokens": 1024,
    "simulation_count": 5,
    "sample_count": 2,
    "LLMs_to_eval":
    [
        {
            "company": "anthropic",
            "enabled": False,
            "params": {
                "model_name": "claude-opus-4",
                "date_code": "20250514"
            }
        },
        {
            "company": "anthropic",
            "enabled": False,
            "params": {
                "model_name": "claude-sonnet-4",
                "date_code": "20250514"
            }
        }
    ]

}
