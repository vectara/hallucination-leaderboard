from datetime import datetime

# Evaluation configs 
# This is a series of evaluation configurations. 
# Please only append so we can always know how previous evaluations were done.
# To select between configs, use the --config_key flag in `main.py`
eval_configs = { 
    # TODO: add HHEM version to config
    "test": {
        "eval_date": datetime.now().strftime('%Y%m%d'), # Today's date
        "hhem_version": "2.3",
        "pipeline": ["summarize", "judge", "reduce"],
        "overwrite_summaries": True,
        "source_article_path": "datasets/test_data.csv",
        "temperature": 0.0, 
        "max_tokens": 1024,
        "simulation_count": 1, # NO EFFECT ON PROGRAM ATM
        "sample_count": 1, # NO EFFECT ON PROGRAM ATM
    },
    "REAL_RUN_1": {
        "eval_date": "20250703",
        "hhem_version": "2.3",
        "pipeline": ["summarize", "judge", "reduce"],
        "overwrite_summaries": True,
        "source_article_path": "datasets/leaderboard_data.csv",
        "temperature": 0.0, 
        "max_tokens": 1024,
        "simulation_count": 1, # NO EFFECT ON PROGRAM ATM
        "sample_count": 1, # NO EFFECT ON PROGRAM ATM
        "LLM_Configs": [
            # {
            #     "company": "rednote",
        #         "model_name": "rednote-hilab/dots.llm1.base",
        #         "execution_mode": "local",
        #         "interaction_mode": "completion",
        #         "temperature": 0.001 # Doesn't accept 0.0
        #     }
            # ,
            {
                "company": "mistralai",
                "model_name": "mistral-small",
                "execution_mode": "client",
                "interaction_mode": "chat",
                "model_date_code": "2506"
            }
            # ,
            # {
            #     "company": "anthropic",
            #     "model_name": "claude-opus-4",
            #     "execution_mode": "client",
            #     "interaction_mode": "chat",
            #     "model_date_code": "20250514"
            # }
            # ,
            # {
            #     "company": "openai",
            #     "model_name": "gpt-4.1",
            #     "execution_mode": "client",
            # }
            #,
            # {
            #     "company": "anthropic",
            #     "model_name": "claude-sonnet-4",
            #     "execution_mode": "client",
            # }
        ]
    }
}

today = datetime.now().strftime('%Y%m%d')
