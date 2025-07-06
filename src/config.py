from datetime import datetime
from typing import List, Dict

from . data_model import EvalConfig

# Please only append so we can always know how previous evaluations were done.
# To select between configs, use the --eval_name flag in `main.py`

from . LLMs import AnthropicConfig, OpenAIConfig

eval_configs: List[EvalConfig] = [
    EvalConfig(**
        {
            "eval_name": "test",
            "eval_date":datetime.now().strftime('%Y-%m-%d'), #today
            "hhem_version": "2.3",
            # "pipeline": ["summarize", "judge", "aggregate"],
            "pipeline": ["summarize", "judge", "aggregate"],
            "overwrite_summaries": True,
            "source_article_path": "datasets/test_articles.csv",
            "temperature": 1.0, 
            "max_tokens": 1024,
            "simulation_count": 1, # NO EFFECT ON PROGRAM ATM
            "sample_count": 1, # NO EFFECT ON PROGRAM ATM
            "output_dir": "output",
            "LLM_Configs": [
                AnthropicConfig(**
                {
                    "company": "anthropic",
                    "model_name": "claude-sonnet-4", 
                    "max_tokens": 2345,
                    "date_code": "20250514",
                }), 
                OpenAIConfig(**
                {
                    "company": "openai",
                    "model_name": "gpt-4.1",
                    "max_tokens": 2345,
                }),
            ]
        }
    ),
    EvalConfig(**
        {
            "eval_name": "eval_for_new_models",
            "eval_date": "2025-07-03",
            "hhem_version": "2.3",
            "pipeline": ["summarize", "judge", "aggregate"],
            "overwrite_summaries": True,
            "source_article_path": "datasets/leaderboard_data.csv",
            "temperature": 0.0, 
            "max_tokens": 1024,
            "simulation_count": 1, # NO EFFECT ON PROGRAM ATM
            "sample_count": 1, # NO EFFECT ON PROGRAM ATM
            "output_dir": "output",
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
                    "execution_mode": "api",
                    "interaction_mode": "chat",
                    "model_date_code": "2506"
                }
                # ,
                # {
                #     "company": "anthropic",
                #     "model_name": "claude-opus-4",
                # }
                # ,
                # {
                #     "company": "openai",
                #     "model_name": "gpt-4.1"
                # }
                #,
                # {
                #     "company": "anthropic",
                #     "model_name": "claude-sonnet-4"
                # }
            ]
        }
    ),
]

# today = datetime.now().strftime('%Y%m%d')
