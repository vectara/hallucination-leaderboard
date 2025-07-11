from datetime import datetime
from typing import List, Dict

from . data_model import EvalConfig, BasicLLMConfig
from . LLMs import AnthropicConfig, OpenAIConfig, AlibabaConfig, XAIConfig

# Please only append so we can always know how previous evaluations were done.
# To select between configs, use the --eval_name flag in `main.py`

# `eval_configs` is a list of dictionaries. Each dictionary is a config for an evaluation that can be instantiated as an EvalConfig object.
eval_configs = [
  EvalConfig(**
    {
      "eval_name": "test",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      # "pipeline": ["summarize", "judge", "aggregate"],
      "pipeline": ["summarize"],
      "output_dir": "output",
      "overwrite_summaries": True,
      "source_article_path": "datasets/test_articles.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 1.0, 
            "max_tokens": 1024, 
            "prompt": """
You are a chat bot answering questions using data.
You must stick to the answers provided solely by the text in the 
passage provided. You are asked the question 'Provide a concise 
summary of the following passage, covering the core pieces of 
information described.'

Just provide your answer in a single paragraph, without any prompt like "Here is the summary:" or any endings like "I hope I have answered your question."

If you cannot answer the question, for reasons like insufficient information in the passage, 
just say 'I cannot do it. 2389fdsi2389432ksad' and do not say anything else. 
  
Here is the passage:
{article}
""",
          }
        ),
      "per_LLM_configs": [
        XAIConfig(**
          {
            "company": "xai",
            "model_name": "grok-3",
            "temperature": 0.0,
          }
        ),
        XAIConfig(**
          {
            "company": "xai",
            "model_name": "grok-3-mini",
            "temperature": 0.0,
          }
        ),
        XAIConfig(**
          {
            "company": "xai",
            "model_name": "grok-3-fast",
            "temperature": 0.0,
          }
        ),
        XAIConfig(**
          {
            "company": "xai",
            "model_name": "grok-3-mini-fast",
            "temperature": 0.0,
          }
        ),
        XAIConfig(**
          {
            "company": "xai",
            "model_name": "grok-2-vision",
            "temperature": 0.0,
            "date_code": "1212",
          }
        ),
        # XAIConfig(**
        #   {
        #     "company": "xai",
        #     "model_name": "grok-4",
        #     "temperature": 0.0,
        #     "date_code": "0709",
        #     "min_throttle_time": 4.0
        #   }
        # ),
        # AlibabaConfig(**
        #   {
        #     "company": "alibaba",
        #     "model_name": "qwen2.5-72b-instruct",
        #   }
        # ),
        # AlibabaConfig(**
        #   {
        #     "company": "alibaba",
        #     "model_name": "qwen2.5-32b-instruct",
        #   }
        # ),
        # AlibabaConfig(**
        #   {
        #     "company": "alibaba",
        #     "model_name": "qwen2.5-14b-instruct",
        #   }
        # ),
        # AlibabaConfig(**
        #   {
        #     "company": "alibaba",
        #     "model_name": "qwen2.5-7b-instruct",
        #   }
        # ),
        # AlibabaConfig(**
        #   {
        #     "company": "alibaba",
        #     "model_name": "qwen-max",
        #     "date_code": "2025-01-25", # AKA Qwen2.5-Max
        #   }
        # ),
        # AlibabaConfig(**
        #   {
        #     "company": "alibaba",
        #     "model_name": "qwen3-32b",
        #     "thinking_tokens": 0
        #   }
        # ),
        # AlibabaConfig(**
        #   {
        #     "company": "alibaba",
        #     "model_name": "qwen3-14b",
        #     "thinking_tokens": 0
        #   }
        # ),
        # AlibabaConfig(**
        #   {
        #     "company": "alibaba",
        #     "model_name": "qwen3-8b",
        #     "thinking_tokens": 0
        #   }
        # ),
        # AlibabaConfig(**
        #   {
        #     "company": "alibaba",
        #     "model_name": "qwen3-4b",
        #     "thinking_tokens": 0
        #   }
        # ),
        # AlibabaConfig(**
        #   {
        #     "company": "alibaba",
        #     "model_name": "qwen3-1.7b",
        #     "thinking_tokens": 0
        #   }
        # ),
        # AlibabaConfig(**
        #   {
        #     "company": "alibaba",
        #     "model_name": "qwen3-0.6b",
        #     "thinking_tokens": 0
        #   }
        # ),
        # AnthropicConfig(**
        #   {
        #     "company": "anthropic",
        #     "model_name": "claude-3-5-haiku",
        #     "max_tokens": 2345,
        #     "date_code": "20241022",
        #   }
        # ),
        # OpenAIConfig(**
        #   {
        #     "company": "openai",
        #     "model_name": "gpt-4.1-nano",
        #   }
        # ),
      ]
    }
  ),
  EvalConfig(**
    {
      "eval_name": "live",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["summarize", "judge", "aggregate"],
      "output_dir": "output",
      "overwrite_summaries": True,
      "source_article_path": "datasets/leaderboard_dataset_revised.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 1.0, 
            "max_tokens": 1024, 
            "prompt": """
You are a chat bot answering questions using data.
You must stick to the answers provided solely by the text in the 
passage provided. You are asked the question 'Provide a concise 
summary of the following passage, covering the core pieces of 
information described.'

Just provide your answer in a single paragraph, without any prompt like "Here is the summary:" or any endings like "I hope I have answered your question."

If you cannot answer the question, for reasons like insufficient information in the passage, 
just say 'I cannot do it. 2389fdsi2389432ksad' and do not say anything else. 
  
Here is the passage:
{article}
""",
          }
        ),
      "per_LLM_configs": [
        XAIConfig(**
          {
            "company": "xai",
            "model_name": "grok-4",
            "temperature": 0.0,
            "date_code": "0709",
            "min_throttle_time": 4.0
          }
        ),
      ]
    }
  )
]