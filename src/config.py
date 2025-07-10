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