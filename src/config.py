from datetime import datetime
from typing import List, Dict

from . data_model import EvalConfig, BasicLLMConfig
from . LLMs import AnthropicConfig, OpenAIConfig

# Please only append so we can always know how previous evaluations were done.
# To select between configs, use the --eval_name flag in `main.py`

# `eval_configs` is a list of dictionaries. Each dictionary is a config for an evaluation that can be instantiated as an EvalConfig object.
eval_configs = [
  EvalConfig(**
    {
      "eval_name": "test",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["summarize", "judge", "aggregate"],
      "output_dir": "output",
      "overwrite_summaries": True,
      "source_article_path": "datasets/test_articles.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 1.0, 
            "max_tokens": 1024
          }
        ),
      "per_LLM_configs": [
        AnthropicConfig(**
          {
            "company": "anthropic",
            "model_name": "claude-3-5-haiku",
            "max_tokens": 2345,
            "date_code": "20241022",
          }
        ),
        OpenAIConfig(**
          {
            "company": "openai",
            "model_name": "gpt-4.1-nano",
            # "max_tokens": 2345,
          }
        ),
      ]
    }
  )
]

# today = datetime.now().strftime('%Y%m%d') 