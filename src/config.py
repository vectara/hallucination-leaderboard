from datetime import datetime
from typing import List, Dict
import os

from . data_model import EvalConfig, BasicLLMConfig
from . LLMs import (
  AnthropicConfig,
  AppleConfig,
  OpenAIConfig,
  QwenConfig,
  XAIConfig,
  CohereConfig,
  GoogleConfig,
  MoonshotAIConfig,
  DeepSeekAIConfig,
  MistralAIConfig,
  MetaLlamaConfig,
  MicrosoftConfig,
  _01AIConfig,
  AI21LabsConfig,
  AllenAIConfig,
  IBMGraniteConfig,
  TngTechConfig,
  AntGroupMIConfig,
  ZhipuAIConfig,
  VectaraConfig,
  AmazonConfig,
  DatabricksConfig,
  IntelConfig,
  InternLmConfig,
  TiiuaeConfig,
  SnowflakeConfig
)

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
      # "pipeline": ["aggregate"],
      "pipeline": ["summarize"],
      "output_dir": "output_test",
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

Just provide your answer without any prompt like "Here is the summary:" or any endings like "I hope I have answered your question."

If you are unable to summarize the text due to missing, unreadable, irrelevant or insufficient content, respond only with:

"I am unable to summarize this text."
  
Here is the passage:
{article}
""",
          }
        ),
      "per_LLM_configs": [
        # _01AIConfig(**
        #   {
        #     "company": "01-ai",
        #     "model_name": "Yi-1.5-34B-Chat",
        #     "temperature": 0.01, # Cant be 0.0 has to be positive
        #   }
        # ),
        # AI21LabsConfig(**
        #   {
        #     "company": "ai21labs",
        #     "model_name": "jamba-large-1.7",
        #     "date_code": "2025-07",
        #     "temperature": 0.0
        #   }
        # ),
        # AI21LabsConfig(**
        #   {
        #     "company": "ai21labs",
        #     "model_name": "jamba-mini-1.7",
        #     "date_code": "2025-07",
        #     "temperature": 0.0
        #   }
        # ),
        # AI21LabsConfig(**
        #   {
        #     "company": "ai21labs",
        #     "model_name": "jamba-large-1.6",
        #     "date_code": "2025-03",
        #     "temperature": 0.0
        #   }
        # ),
        # AI21LabsConfig(**
        #   {
        #     "company": "ai21labs",
        #     "model_name": "jamba-mini-1.6",
        #     "date_code": "2025-03",
        #     "temperature": 0.0
        #   }
        # ),
        # AllenAIConfig(**
        #   {
        #     "company": "allenai",
        #     "model_name": "OLMo-2-7B-Instruct",
        #     "date_code": 1124,
        #     "temperature": 0.01, # Cant be 0.0 has to be positive
        #   }
        # ),
        # AntGroupMIConfig(**
        #   {
        #     "company": "antgroup",
        #     "model_name": "antfinix-ir1",
        #     "date_code": "",
        #     "temperature": 0.0,
        #   }
        # ),
        # AntGroupMIConfig(**
        #   {
        #     "company": "antgroup",
        #     "model_name": "antfinix-a1",
        #     "date_code": "",
        #     "temperature": 0.0,
        #   }
        # ),
        # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-sonnet-4", "date_code": "20250514", "temperature": 0.0}),
        # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-opus-4", "date_code": "20250514", "temperature": 0.0}),
        # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-opus-4-1", "date_code": "20250805", "temperature": 0.0}),
        # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-2.0", "temperature": 0.0}),
        # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-3-5-haiku", "max_tokens": 2345, "date_code": "20241022"}),
        # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-3-5-sonnet", "date_code": "20241022", "temperature": 0.0}),
        # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-3-7-sonnet", "date_code": "20250219", "temperature": 0.0}),
        # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-3-opus", "date_code": "20240229", "temperature": 0.0}),
        # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-3-sonnet", "date_code": "20240229", "temperature": 0.0}),
        # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-sonnet-4", "date_code": "20250514", "temperature": 0.0}),
        # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-opus-4", "date_code": "20250514", "temperature": 0.0}),
        # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-sonnet-4-5", "date_code": "20250929", "temperature": 0.0}),
        # CohereConfig(**{"company": "cohere", "model_name": "c4ai-aya-expanse-32b", "temperature": 0.0}),
        # CohereConfig(**{"company": "cohere", "model_name": "c4ai-aya-expanse-8b", "temperature": 0.0}),
        # CohereConfig(**{"company": "cohere", "model_name": "command-a", "date_code": "03-2025", "temperature": 0.0}),
        # CohereConfig(**{"company": "cohere", "model_name": "command-r", "date_code": "08-2024", "temperature": 0.0}),
        # CohereConfig(**{"company": "cohere", "model_name": "command-r-plus", "date_code": "04-2024", "temperature": 0.0}),
        # CohereConfig(**{"company": "cohere", "model_name": "command-r7b", "date_code": "12-2024", "temperature": 0.0}),
        # CohereConfig(**{"company": "cohere", "model_name": "command-a-reasoning", "date_code": "08-2025", "temperature": 0.0, "max_tokens": 4096, "min_throttle_time": 5.0}),
        # DeepSeekAIConfig(**{"company": "deepseek-ai", "model_name": "DeepSeek-R1", "temperature": 0.0, "min_throttle_time": 4.0}),
        # DeepSeekAIConfig(**{"company": "deepseek-ai", "model_name": "DeepSeek-V3", "date_code": "0324", "temperature": 0.0, "min_throttle_time": 4.0}),
        # DeepSeekAIConfig(**{"company": "deepseek-ai", "model_name": "DeepSeek-V3", "temperature": 0.0, "min_throttle_time": 4.0}),
        # DeepSeekAIConfig(**{"company": "deepseek-ai", "model_name": "DeepSeek-V3.1", "temperature": 0.0, "min_throttle_time": 4.0}),
        # DeepSeekAIConfig(**{"company": "deepseek-ai", "model_name": "DeepSeek-V3.2-Exp", "temperature": 0.0, "min_throttle_time": 4.0}),
        # DeepSeekAIConfig(**{"company": "deepseek-ai", "model_name": "DeepSeek-V3.1-Terminus", "temperature": 0.0, "min_throttle_time": 4.0}),
        # GoogleConfig(**{"company": "google", "model_name": "gemini-2.5-flash", "date_code": "", "temperature": 0.0, "thinking_budget": -1}),
        # GoogleConfig(**{"company": "google", "model_name": "gemini-2.5-pro", "date_code": "", "temperature": 0.0, "thinking_budget": -1}),
        # GoogleConfig(**{"company": "google", "model_name": "gemini-1.5-flash", "temperature": 0.0}),
        # GoogleConfig(**{"company": "google", "model_name": "gemini-1.5-flash-002", "temperature": 0.0}),
        # GoogleConfig(**{"company": "google", "model_name": "gemini-1.5-pro", "temperature": 0.0}),
        # GoogleConfig(**{"company": "google", "model_name": "gemini-1.5-pro-002", "temperature": 0.0}),
        # GoogleConfig(**{"company": "google", "model_name": "gemini-2.0-flash-001", "temperature": 0.0, "thinking_budget": -1}),
        # GoogleConfig(**{"company": "google", "model_name": "gemini-2.0-flash", "date_code":"", "temperature": 0.0, "thinking_budget": -1}), #Odd bug with date code if its not set here?
        # GoogleConfig(**{"company": "google", "model_name": "gemini-2.0-flash-exp", "temperature": 0.0}),
        # GoogleConfig(**{"company": "google", "model_name": "gemini-2.0-flash-lite", "temperature": 0.0}),
        # GoogleConfig(**{"company": "google", "model_name": "gemini-2.5-flash-preview", "date_code": "05-20", "temperature": 0.0}),
        # GoogleConfig(**{"company": "google", "model_name": "gemini-2.5-flash-lite-instinct", "date_code": "", "temperature": 0.0, "thinking_budget": 0}),
        # GoogleConfig(**{"company": "google", "model_name": "gemini-2.5-flash-lite-think", "date_code": "", "temperature": 0.0, "thinking_budget": -1}),
        # GoogleConfig(**{"company": "google", "model_name": "gemma-3-12b-it", "temperature": 0.0}),
        # GoogleConfig(**{"company": "google", "model_name": "gemma-3-1b-it", "temperature": 0.0}),
        # GoogleConfig(**{"company": "google", "model_name": "gemma-3-27b-it", "temperature": 0.0}),
        # GoogleConfig(**{"company": "google", "model_name": "gemma-3-4b-it", "temperature": 0.0}),
        # IBMGraniteConfig(**
        #   {
        #     "model_name": "granite-4.0-h-small", # Has an odd error but seems to work? The attention mask is not set and cannot be inferred from input because pad token is same as eos token as a consequence you may observe unexpected behavior please pass your inputs attention_mask to obtain reliable results
        #     "temperature": 0.01, # Cant be 0.0 has to be positive
        #   }
        # ),
        # IBMGraniteConfig(**
        #   {
        #     "model_name": "granite-4.0-h-tiny", # Has an odd error but seems to work? The attention mask is not set and cannot be inferred from input because pad token is same as eos token as a consequence you may observe unexpected behavior please pass your inputs attention_mask to obtain reliable results
        #     "temperature": 0.01, # Cant be 0.0 has to be positive
        #   }
        # ),
        # IBMGraniteConfig(**
        #   {
        #     "model_name": "granite-4.0-h-micro", # Has an odd error but seems to work? The attention mask is not set and cannot be inferred from input because pad token is same as eos token as a consequence you may observe unexpected behavior please pass your inputs attention_mask to obtain reliable results
        #     "temperature": -1, # Cant be 0.0 has to be positive
        #   }
        # ),
        # IBMGraniteConfig(**
        #   {
        #     "model_name": "granite-4.0-micro", # Has an odd error but seems to work? The attention mask is not set and cannot be inferred from input because pad token is same as eos token as a consequence you may observe unexpected behavior please pass your inputs attention_mask to obtain reliable results
        #     "temperature": -1, # Cant be 0.0 has to be positive
        #   }
        # ),
        # IBMGraniteConfig(**
        #   {
        #     "company": "ibm-granite",
        #     "model_name": "granite-3.2-2b-instruct", # Has an odd error but seems to work? The attention mask is not set and cannot be inferred from input because pad token is same as eos token as a consequence you may observe unexpected behavior please pass your inputs attention_mask to obtain reliable results
        #     "temperature": 0.01, # Cant be 0.0 has to be positive
        #   }
        # ),
        # IBMGraniteConfig(**
        #   {
        #     "company": "ibm-granite",
        #     "model_name": "granite-3.2-8b-instruct", # Has an odd error but seems to work? The attention mask is not set and cannot be inferred from input because pad token is same as eos token as a consequence you may observe unexpected behavior please pass your inputs attention_mask to obtain reliable results
        #     "temperature": 0.01, # Cant be 0.0 has to be positive
        #   }
        # ),
        # IBMGraniteConfig(**
        #   {
        #     "company": "ibm-granite",
        #     "model_name": "granite-3.1-2b-instruct",
        #     "temperature": 0.01, # Cant be 0.0 has to be positive
        #   }
        # ),
        # IBMGraniteConfig(**
        #   {
        #     "company": "ibm-granite",
        #     "model_name": "granite-3.1-8b-instruct",
        #     "temperature": 0.01, # Cant be 0.0 has to be positive
        #   }
        # ),
        # IBMGraniteConfig(**
        #   {
        #     "company": "ibm-granite",
        #     "model_name": "granite-3.0-2b-instruct",
        #     "temperature": 0.01, # Cant be 0.0 has to be positive
        #   }
        # ),
        # IBMGraniteConfig(**
        #   {
        #     "company": "ibm-granite",
        #     "model_name": "granite-3.0-8b-instruct",
        #     "temperature": 0.01, # Cant be 0.0 has to be positive
        #   }
        # ),
        # MetaLlamaConfig(**
        #   {
        #     "company": "meta-llama",
        #     "model_name": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        #     "temperature": 0.0
        #   }
        # ),
        # MetaLlamaConfig(**
        #   {
        #     "company": "meta-llama",
        #     "model_name": "Llama-4-Scout-17B-16E-Instruct",
        #     "temperature": 0.0
        #   }
        # ),
        # MetaLlamaConfig(**
        #   {
        #     "company": "meta-llama",
        #     "model_name": "Meta-Llama-3.1-8B-Instruct-Turbo",
        #     "temperature": 0.0
        #   }
        # ),
        # MetaLlamaConfig(**
        #   {
        #     "company": "meta-llama",
        #     "model_name": "Llama-3.3-70B-Instruct-Turbo",
        #     "temperature": 0.0
        #   }
        # ),
        # MetaLlamaConfig(**
        #   {
        #     "company": "meta-llama",
        #     "model_name": "Llama-3.3-70B-Instruct-Turbo-Free",
        #     "temperature": 0.0
        #   }
        # ),
        # MetaLlamaConfig(**
        #   {
        #     "company": "meta-llama",
        #     "model_name": "Meta-Llama-3.1-405B-Instruct-Turbo",
        #     "temperature": 0.0
        #   }
        # ),
        # MetaLlamaConfig(**
        #   {
        #     "company": "meta-llama",
        #     "model_name": "Llama-3.2-3B-Instruct-Turbo",
        #     "temperature": 0.0
        #   }
        # ),
        # MetaLlamaConfig(** # Unable to Access Model
        #   {
        #     "company": "meta-llama",
        #     "model_name": "Llama-3.2-11B-Vision-Instruct-Turbo*",
        #     "temperature": 0.0
        #   }
        # ),
        # MetaLlamaConfig(** # Unable to Access Model
        #   {
        #     "company": "meta-llama",
        #     "model_name": "Llama-3.2-90B-Vision-Instruct-Turbo*",
        #     "temperature": 0.0
        #   }
        # ),
        # MetaLlamaConfig(**
        #   {
        #     "company": "meta-llama",
        #     "model_name": "Meta-Llama-3.1-405B-Instruct-Turbo",
        #     "temperature": 0.0
        #   }
        # ),
        # MetaLlamaConfig(**
        #   {
        #     "company": "meta-llama",
        #     "model_name": "Meta-Llama-3.1-8B-Instruct-Turbo",
        #     "temperature": 0.0
        #   }
        # ),
        # MetaLlamaConfig(**
        #   {
        #     "company": "meta-llama",
        #     "model_name": "Meta-Llama-3-8B-Instruct-Lite",
        #     "temperature": 0.0
        #   }
        # ),
        # MetaLlamaConfig(**
        #   {
        #     "company": "meta-llama",
        #     "model_name": "Llama-3-8b-chat-hf*",
        #     "temperature": 0.0
        #   }
        # ),
        # MetaLlamaConfig(**
        #   {
        #     "company": "meta-llama",
        #     "model_name": "Llama-3-70b-chat-hf",
        #     "temperature": 0.0
        #   }
        # ),
        # MetaLlamaConfig(**
        #   {
        #     "company": "meta-llama",
        #     "model_name": "Llama-2-70b-hf",
        #     "temperature": 0.0,
        #     "endpoint": "response"
        #   }
        # ),
        # MicrosoftConfig(**
        #   {
        #     "company": "microsoft",
        #     "model_name": "Phi-4-mini-instruct",
        #     "model_key": os.getenv("PHI_4_MINI_INSTRUCT_API_KEY"),
        #     "azure_endpoint": "https://hhem-lb-phi-4-mini-inst-resource.services.ai.azure.com/models",
        #     "temperature": 0.0,
        #   }
        # ),
        # MicrosoftConfig(**
        #   {
        #     "company": "microsoft",
        #     "model_name": "Phi-4",
        #     "model_key": os.getenv("PHI_4_API_KEY"),
        #     "azure_endpoint": "https://hhem-lb-phi-4-resource.services.ai.azure.com/models",
        #     "temperature": 0.0,
        #   }
        # ),
        # MistralAIConfig(**{"company": "mistralai", "model_name": "ministral-3b", "date_code": "2410", "temperature": 0.0}),
        # MistralAIConfig(**{"company": "mistralai", "model_name": "ministral-8b", "date_code": "2410", "temperature": 0.0}),
        # MistralAIConfig(**{"company": "mistralai", "model_name": "mistral-large", "date_code": "2411", "temperature": 0.0}),
        # MistralAIConfig(**{"company": "mistralai", "model_name": "mistral-small", "date_code": "2407", "temperature": 0.0}), # Invalid model? code 1500
        # MistralAIConfig(**{"company": "mistralai", "model_name": "mistral-small", "date_code": "2407", "temperature": 0.0}), # Invalidmodel? code 1500
        # MistralAIConfig(**{"company": "mistralai", "model_name": "mistral-small", "date_code": "2501", "temperature": 0.0}),
        # MistralAIConfig(**{"company": "mistralai", "model_name": "open-mistral-nemo", "temperature": 0.0}),
        # MistralAIConfig(**{"company": "mistralai", "model_name": "pixtral-12b", "date_code": "2409", "temperature": 0.0}),
        # MistralAIConfig(**{"company": "mistralai", "model_name": "pixtral-large", "date_code": "2411", "temperature": 0.0}),
        # MistralAIConfig(**{"company": "mistralai", "model_name": "mistral-medium", "date_code": "2508", "temperature": 0.0}), #mistral medium 3.1 date code
        # MoonshotAIConfig(**{"company": "moonshotai", "model_name": "moonshotai/Kimi-K2-Instruct", "temperature": 0.0, "min_throttle_time": 4.0}),
        # MoonshotAIConfig(**{"company": "moonshotai", "model_name": "Kimi-K2-Instruct", "temperature": 0.0, "min_throttle_time": 4.0}),
        # MoonshotAIConfig(**{"company": "moonshotai", "model_name": "Kimi-K2-Instruct", "date_code": "0905","temperature": 0.0, "min_throttle_time": 4.0}),
        # OpenAIConfig(**{"company": "openai", "model_name": "gpt-5-high", "date_code": "2025-08-07", "reasoning_effort": "high", "temperature": -1.0, "max_tokens": 4096}),
        # OpenAIConfig(**{"company": "openai", "model_name": "gpt-4.1", "date_code": "2025-04-14", "temperature": 0.0}),
        # OpenAIConfig(**{"company": "openai", "model_name": "gpt-5-minimal", "date_code": "2025-08-07", "reasoning_effort": "minimal", "temperature": -1.0}),
        # OpenAIConfig(**{"company": "openai", "model_name": "gpt-5-mini", "date_code": "2025-08-07", "reasoning_effort": "minimal", "temperature": -1.0}),
        # OpenAIConfig(**{"company": "openai", "model_name": "gpt-5-nano", "date_code": "2025-08-07", "reasoning_effort":"minimal", "temperature": -1.0}),
        # OpenAIConfig(**{"company": "openai", "execution_mode": "gpu", "model_name": "gpt-oss-20b", "date_code": "", "temperature": 0.01}),
        # OpenAIConfig(**{"company": "openai", "model_name": "gpt-oss-120b", "date_code": "", "temperature": 0.0}),
        # OpenAIConfig(**{"company": "openai", "model_name": "gpt-3.5-turbo", "date_code": "0125", "temperature": 0.0}),
        # OpenAIConfig(**{"company": "openai", "model_name": "gpt-4", "date_code": "0613", "temperature": 0.0}),
        # OpenAIConfig(**{"company": "openai", "model_name": "gpt-4-turbo", "date_code": "2024-04-09", "temperature": 0.0}),
        # OpenAIConfig(**{"company": "openai", "model_name": "gpt-4.1-mini", "date_code": "2025-04-14", "temperature": 0.0}),
        # OpenAIConfig(**{"company": "openai", "model_name": "gpt-4.1-nano"}),
        # OpenAIConfig(**{"company": "openai", "model_name": "gpt-4o", "date_code": "2024-08-06", "temperature": 0.0}),
        # OpenAIConfig(**{"company": "openai", "model_name": "gpt-4o-mini", "date_code": "2024-07-18", "temperature": 0.0, "reasoning_effort": "low"}),
        # OpenAIConfig(**{"company": "openai", "model_name": "o1", "date_code": "2024-12-17", "temperature": -1, "reasoning_effort": "low"}),
        # OpenAIConfig(**{"company": "openai", "model_name": "o1-mini", "date_code": "2024-09-12", "temperature": 1.0, "reasoning_effort": "low"}),
        # OpenAIConfig(**{"company": "openai", "model_name": "o1-pro", "date_code": "2025-03-19", "temperature": -1, "endpoint": "response", "reasoning_effort": "low"}),
        # OpenAIConfig(**{"company": "openai", "model_name": "o3-pro", "temperature": 0.0, "endpoint": "response", "reasoning_effort": "low"}),
        # OpenAIConfig(**{"company": "openai", "model_name": "o4-mini-low", "date_code": "2025-04-16", "temperature": 1.0, "reasoning_effort": "low"}),
        # OpenAIConfig(**{"company": "openai", "model_name": "o4-mini-high", "date_code": "2025-04-16", "temperature": 1.0, "reasoning_effort": "high"}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-30b-a3b-thinking", "date_code": "2507", "temperature": 0.0, "enable_thinking": True}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-next-80b-a3b-thinking", "date_code": "", "temperature": 0.0, "enable_thinking": True}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-omni-30b-a3b-thinking", "date_code": "", "temperature": 0.0, "enable_thinking": True}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen-max", "date_code": "2025-01-25", "temperature": 0.0}), # AKA Qwen2.5-Max
        # QwenConfig(**{"company": "qwen", "model_name": "qwen-plus", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        # QwenConfig(**{"company": "qwen", "model_name": "qwen-turbo", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-14b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-32b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-7b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-72b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-0.6b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-1.7b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-14b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-32b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-4b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-8b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen-max", "date_code": "2025-01-25", "temperature": 0.0}), # AKA Qwen2.5-Max
        # QwenConfig(**{"company": "qwen", "model_name": "qwen-plus", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        # QwenConfig(**{"company": "qwen", "model_name": "qwen-turbo", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-14b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-32b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-7b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-72b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-0.6b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-1.7b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-14b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-32b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-4b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-8b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-max-preview", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # TngTechConfig(**{"company": "tngtech", "model_name": "DeepSeek-TNG-R1T2-Chimera", "temperature": 0.0}),
        # XAIConfig(**{"company": "xai", "model_name": "grok-2-vision", "temperature": 0.0, "date_code": "1212"}),
        # XAIConfig(**{"company": "xai", "model_name": "grok-3", "temperature": 0.0}),
        # XAIConfig(**{"company": "xai", "model_name": "grok-3-fast", "temperature": 0.0}),
        # XAIConfig(**{"company": "xai", "model_name": "grok-3-mini", "temperature": 0.0}),
        # XAIConfig(**{"company": "xai", "model_name": "grok-3-mini-fast", "temperature": 0.0}),
        # XAIConfig(**{"company": "xai", "model_name": "grok-4", "temperature": 0.0, "date_code": "0709", "min_throttle_time": 4.0}),
        # _01AIConfig(**
        #   {
        #     "company": "01-ai",
        #     "model_name": "Yi-1.5-9B-Chat",
        #     "temperature": 0.01, # Cant be 0.0 has to be positive
        #   }
        # ),
        # ZhipuAIConfig(**{"company": "zai-org", "model_name": "GLM-4.5-AIR-FP8", "temperature": 0.0}),
        # ZhipuAIConfig(**{"company": "zai-org", "model_name": "glm-4p5", "temperature": 0.0}),
        ZhipuAIConfig(**{"model_name": "GLM-4.6", "temperature": 0.0}),
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

Just provide your answer without any prompt like "Here is the summary:" or any endings like "I hope I have answered your question."

If you are unable to summarize the text due to missing, unreadable, irrelevant or insufficient content, respond only with:

"I am unable to summarize this text."
  
Here is the passage:
{article}
""",
          }
        ),
      "per_LLM_configs": [
        # IBMGraniteConfig(**
        #   {
        #     "model_name": "granite-4.0-h-small", # Has an odd error but seems to work? The attention mask is not set and cannot be inferred from input because pad token is same as eos token as a consequence you may observe unexpected behavior please pass your inputs attention_mask to obtain reliable results
        #     "temperature": 0.01, # Cant be 0.0 has to be positive
        #   }
        # ),
        # IBMGraniteConfig(**
        #   {
        #     "model_name": "granite-4.0-h-tiny", # Has an odd error but seems to work? The attention mask is not set and cannot be inferred from input because pad token is same as eos token as a consequence you may observe unexpected behavior please pass your inputs attention_mask to obtain reliable results
        #     "temperature": 0.01, # Cant be 0.0 has to be positive
        #   }
        # ),
        # IBMGraniteConfig(**
        #   {
        #     "model_name": "granite-4.0-h-micro", # Has an odd error but seems to work? The attention mask is not set and cannot be inferred from input because pad token is same as eos token as a consequence you may observe unexpected behavior please pass your inputs attention_mask to obtain reliable results
        #     "temperature": -1,
        #   }
        # ),
        # IBMGraniteConfig(**
        #   {
        #     "model_name": "granite-4.0-micro", # Has an odd error but seems to work? The attention mask is not set and cannot be inferred from input because pad token is same as eos token as a consequence you may observe unexpected behavior please pass your inputs attention_mask to obtain reliable results
        #     "temperature": -1,
        #   }
        # ),
        ZhipuAIConfig(**{"model_name": "GLM-4.6", "temperature": 0.0, "min_throttle_time": 4.0}),
      ]
    }
  ),
  EvalConfig(**
    {
      "eval_name": "all_stats",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["aggregate"],
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

Just provide your answer without any prompt like "Here is the summary:" or any endings like "I hope I have answered your question."

If you are unable to summarize the text due to missing, unreadable, irrelevant or insufficient content, respond only with:

"I am unable to summarize this text."
  
Here is the passage:
{article}
""",
          }
        ),
      "per_LLM_configs": [
      ]
    }
  ),
  EvalConfig(**
    {
      "eval_name": "beta_set",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["summarize", "judge", "aggregate"],
      "output_dir": "output_beta",
      "overwrite_summaries": True,
      "source_article_path": "datasets/beta_set.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 1.0, 
            "max_tokens": 2048, 
            "prompt": """
You are a chat bot answering questions using data.
You must stick to the answers provided solely by the text in the 
passage provided. You are asked the question 'Provide a concise 
summary of the following passage, covering the core pieces of 
information described.'

Just provide your answer without any prompt like "Here is the summary:" or any endings like "I hope I have answered your question."

If you are unable to summarize the text due to missing, unreadable, irrelevant or insufficient content, respond only with:

"I am unable to summarize this text."
  
Here is the passage:
{article}
""",
          }
        ),
      "per_LLM_configs": [
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-sonnet-4-5", "date_code": "20250929", "temperature": 0.0}),
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-opus-4-1", "date_code": "20250805", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-next-80b-a3b-thinking", "date_code": "", "temperature": 0.0, "enable_thinking": True}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-32b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "gpt-5-mini", "date_code": "2025-08-07", "reasoning_effort": "high", "temperature": -1.0, "max_tokens": 4096}),
        OpenAIConfig(**{"company": "openai", "model_name": "gpt-4.1", "date_code": "2025-04-14", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "model_name": "gemini-2.5-flash", "date_code": "", "temperature": 0.0, "thinking_budget": -1}),
        GoogleConfig(**{"company": "google", "model_name": "gemini-2.5-pro", "date_code": "", "temperature": 0.0, "thinking_budget": -1}),
      ]
    }
  ),
  EvalConfig(**
    {
      "eval_name": "alpha_set",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["summarize", "judge", "aggregate"],
      "output_dir": "output_alpha",
      "overwrite_summaries": True,
      "source_article_path": "datasets/alpha_set.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 1.0, 
            "max_tokens": 2048, 
            "prompt": """
You are a chat bot answering questions using data.
You must stick to the answers provided solely by the text in the 
passage provided. You are asked the question 'Provide a concise 
summary of the following passage, covering the core pieces of 
information described.'

Just provide your answer without any prompt like "Here is the summary:" or any endings like "I hope I have answered your question."

If you are unable to summarize the text due to missing, unreadable, irrelevant or insufficient content, respond only with:

"I am unable to summarize this text."
  
Here is the passage:
{article}
""",
          }
        ),
      "per_LLM_configs": [
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-sonnet-4-5", "date_code": "20250929", "temperature": 0.0}),
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-opus-4-1", "date_code": "20250805", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-next-80b-a3b-thinking", "date_code": "", "temperature": 0.0, "enable_thinking": True}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-32b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "gpt-5-high", "date_code": "2025-08-07", "reasoning_effort": "high", "temperature": -1.0, "max_tokens": 4096}),
        OpenAIConfig(**{"company": "openai", "model_name": "gpt-4.1", "date_code": "2025-04-14", "temperature": 0.0}),
        DeepSeekAIConfig(**{"company": "deepseek-ai", "model_name": "DeepSeek-V3.1-Terminus", "temperature": 0.0, "min_throttle_time": 4.0}),
        GoogleConfig(**{"company": "google", "model_name": "gemini-2.5-flash", "date_code": "", "temperature": 0.0, "thinking_budget": -1}),
        GoogleConfig(**{"company": "google", "model_name": "gemini-2.5-pro", "date_code": "", "temperature": 0.0, "thinking_budget": -1}),
      ]
    }
  ),
  EvalConfig(**
    {
      "eval_name": "pre_2025-07",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["aggregate"],
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

Just provide your answer without any prompt like "Here is the summary:" or any endings like "I hope I have answered your question."

If you are unable to summarize the text due to missing, unreadable, irrelevant or insufficient content, respond only with:

"I am unable to summarize this text."
  
Here is the passage:
{article}
""",
          }
        ),
      "per_LLM_configs": [
        _01AIConfig(**
          {
            "company": "01-ai",
            "model_name": "Yi-1.5-6B-Chat",
            "date_code": "",
            "temperature": 0.01, # Cant be 0.0 has to be positive
          }
        ),
        _01AIConfig(**
          {
            "company": "01-ai",
            "model_name": "Yi-1.5-9B-Chat",
            "date_code": "",
            "temperature": 0.01, # Cant be 0.0 has to be positive
          }
        ),
        _01AIConfig(**
          {
            "company": "01-ai",
            "model_name": "Yi-1.5-34B-Chat",
            "date_code": "",
            "temperature": 0.01, # Cant be 0.0 has to be positive
          }
        ),

        AI21LabsConfig(**
          {
            "company": "ai21labs",
            "model_name": "AI21-Jamba-Mini-1.5",
            "date_code": "",
            "temperature": 0.0
          }
        ),

        AllenAIConfig(**
          {
            "company": "allenai",
            "model_name": "OLMo-2-0325-32B-Instruct",
            "date_code": "0325",
            "temperature": 0.0
          }
        ),
        AllenAIConfig(**
          {
            "company": "allenai",
            "model_name": "OLMo-2-1124-7b-instruct",
            "date_code": "1124",
            "temperature": 0.0
          }
        ),
        AllenAIConfig(**
          {
            "company": "allenai",
            "model_name": "OLMo-2-1124-13b-instruct",
            "date_code": "1124",
            "temperature": 0.0
          }
        ),

        AmazonConfig(**
          {
            "company": "amazon",
            "model_name": "nova-lite-v1:0",
            "date_code": "",
            "temperature": 0.0
          }
        ),
        AmazonConfig(**
          {
            "company": "amazon",
            "model_name": "nova-micro-v1:0",
            "date_code": "",
            "temperature": 0.0
          }
        ),
        AmazonConfig(**
          {
            "company": "amazon",
            "model_name": "nova-pro-v1:0",
            "date_code": "",
            "temperature": 0.0
          }
        ),

        AntGroupMIConfig(**
          {
            "company": "antgroup",
            "model_name": "finix_s1_32b",
            "date_code": "",
          }
        ),

        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-2.0", "date_code": "", "temperature": 0.0}),
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-3-5-haiku", "date_code": "20241022", "temperature": 0.0}),
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-3-5-sonnet", "date_code": "20241022", "temperature": 0.0}),
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-3-opus", "date_code": "20240229", "temperature": 0.0}),
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-3-sonnet", "date_code": "20240229", "temperature": 0.0}),
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-opus-4-1", "date_code": "20250805", "temperature": 0.0}),

        AppleConfig(**{"company": "apple", "model_name": "OpenELM-3B-Instruct", "date_code": "", "temperature": 0.0}),

        CohereConfig(**{"company": "CohereLabs", "model_name": "aya-expanse-32b", "date_code": "", "temperature": 0.0}),
        CohereConfig(**{"company": "CohereLabs", "model_name": "aya-expanse-8b", "date_code": "", "temperature": 0.0}),
        CohereConfig(**{"company": "CohereLabs", "model_name": "c4ai-command-r-plus", "date_code": "", "temperature": 0.0}),
        CohereConfig(**{"company": "CohereLabs", "model_name": "command", "date_code": "", "temperature": 0.0}),
        CohereConfig(**{"company": "CohereLabs", "model_name": "command-a", "date_code": "03-2025", "temperature": 0.0}),
        CohereConfig(**{"company": "CohereLabs", "model_name": "command-a-reasoning", "date_code": "08-2025", "temperature": 0.0}),
        CohereConfig(**{"company": "CohereLabs", "model_name": "command-chat", "date_code": "", "temperature": 0.0}),
        CohereConfig(**{"company": "CohereLabs", "model_name": "command-r", "date_code": "08-2024", "temperature": 0.0}),

        DatabricksConfig(**{"company": "databricks", "model_name": "dbrx-instruct", "date_code": "", "temperature": 0.0}),

        DeepSeekAIConfig(**{"company": "deepseek-ai", "model_name": "DeepSeek-R1", "date_code": "", "temperature": 0.0}),
        DeepSeekAIConfig(**{"company": "deepseek-ai", "model_name": "DeepSeek-R1-0528", "date_code": "", "temperature": 0.0}),
        DeepSeekAIConfig(**{"company": "deepseek-ai", "model_name": "DeepSeek-V3", "date_code": "", "temperature": 0.0}),
        DeepSeekAIConfig(**{"company": "deepseek-ai", "model_name": "DeepSeek-V3.1", "date_code": "", "temperature": 0.0}),

        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "chat-bison-001", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "flan-t5-large", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemini-1.5-flash", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemini-1.5-flash-001", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemini-1.5-flash-002", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemini-1.5-pro", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemini-1.5-pro-001", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemini-1.5-pro-002", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemini-2.0-flash-001", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemini-2.0-flash-exp", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "02-05", "model_name": "gemini-2.0-flash-lite-preview", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "02-05", "model_name": "gemini-2.0-pro-exp", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemini-2.5-flash", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemini-2.5-flash-lite", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "04-17", "model_name": "gemini-2.5-flash-preview", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "03-25", "model_name": "gemini-2.5-pro-exp", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemini-2.5-pro-preview", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemma-1.1-2b-it", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemma-1.1-7b-it", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemma-2-2b-it", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemma-2-9b-it", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemma-3-1b-it", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemma-3-4b-it", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemma-3-27b-it", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "gemma-7b-it", "temperature": 0.0}),
        GoogleConfig(**{"company": "google", "date_code": "", "model_name": "text-bison-001", "temperature": 0.0}),

        IBMGraniteConfig(**
          {
            "company": "ibm-granite",
            "model_name": "granite-3.2-2b-instruct", # Has an odd error but seems to work? The attention mask is not set and cannot be inferred from input because pad token is same as eos token as a consequence you may observe unexpected behavior please pass your inputs attention_mask to obtain reliable results
            "temperature": 0.01, # Cant be 0.0 has to be positive
            "date_code": ""
          }
        ),
        IBMGraniteConfig(**
          {
            "company": "ibm-granite",
            "model_name": "granite-3.2-8b-instruct", # Has an odd error but seems to work? The attention mask is not set and cannot be inferred from input because pad token is same as eos token as a consequence you may observe unexpected behavior please pass your inputs attention_mask to obtain reliable results
            "temperature": 0.01, # Cant be 0.0 has to be positive
            "date_code": ""
          }
        ),
        IBMGraniteConfig(**
          {
            "company": "ibm-granite",
            "model_name": "granite-3.1-2b-instruct",
            "temperature": 0.01, # Cant be 0.0 has to be positive
            "date_code": ""
          }
        ),
        IBMGraniteConfig(**
          {
            "company": "ibm-granite",
            "model_name": "granite-3.1-8b-instruct",
            "temperature": 0.01, # Cant be 0.0 has to be positive
            "date_code": ""
          }
        ),
        IBMGraniteConfig(**
          {
            "company": "ibm-granite",
            "model_name": "granite-3.0-2b-instruct",
            "temperature": 0.01, # Cant be 0.0 has to be positive
            "date_code": ""
          }
        ),
        IBMGraniteConfig(**
          {
            "company": "ibm-granite",
            "model_name": "granite-3.0-8b-instruct",
            "temperature": 0.01, # Cant be 0.0 has to be positive
            "date_code": ""
          }
        ),

        IntelConfig(**
          {
            "company": "Intel",
            "model_name": "neural-chat-7b-v3-3",
            "temperature": 0.0,
            "date_code": ""
          }
        ),

        InternLmConfig(**
          {
            "company": "internlm",
            "model_name": "internlm3-8b-instruct",
            "temperature": 0.0,
            "date_code": ""
          }
        ),

        MetaLlamaConfig(**
          {
            "company": "meta-llama",
            "model_name": "Llama-2-7b-chat-hf",
            "date_code": "",
            "temperature": 0.0
          }
        ),
        MetaLlamaConfig(**
          {
            "company": "meta-llama",
            "model_name": "Llama-2-13b-chat-hf",
            "date_code": "",
            "temperature": 0.0
          }
        ),
        MetaLlamaConfig(**
          {
            "company": "meta-llama",
            "model_name": "Llama-2-70b-chat-hf",
            "date_code": "",
            "temperature": 0.0
          }
        ),
        MetaLlamaConfig(**
          {
            "company": "meta-llama",
            "model_name": "Llama-3-8B-chat-hf",
            "date_code": "",
            "temperature": 0.0
          }
        ),
        MetaLlamaConfig(**
          {
            "company": "meta-llama",
            "model_name": "Llama-3-70B-chat-hf",
            "date_code": "",
            "temperature": 0.0
          }
        ),
        MetaLlamaConfig(**
          {
            "company": "meta-llama",
            "model_name": "Llama-3.2-1B-Instruct",
            "date_code": "",
            "temperature": 0.0
          }
        ),
        MetaLlamaConfig(**
          {
            "company": "meta-llama",
            "model_name": "Llama-3.2-3B-Instruct-Turbo",
            "date_code": "",
            "temperature": 0.0
          }
        ),
        MetaLlamaConfig(**
          {
            "company": "meta-llama",
            "model_name": "Llama-3.2-11B-Vision-Instruct-Turbo",
            "date_code": "",
            "temperature": 0.0
          }
        ),
        MetaLlamaConfig(**
          {
            "company": "meta-llama",
            "model_name": "Llama-3.2-90B-Vision-Instruct-Turbo",
            "date_code": "",
            "temperature": 0.0
          }
        ),
        MetaLlamaConfig(**
          {
            "company": "meta-llama",
            "model_name": "Llama-3.3-70B-Instruct",
            "date_code": "",
            "temperature": 0.0
          }
        ),
        MetaLlamaConfig(**
          {
            "company": "meta-llama",
            "model_name": "Meta-Llama-3.1-70B-Instruct",
            "date_code": "",
            "temperature": 0.0
          }
        ),
        MetaLlamaConfig(**
          {
            "company": "meta-llama",
            "model_name": "Meta-Llama-3.1-405B-Instruct",
            "date_code": "",
            "temperature": 0.0
          }
        ),

        MicrosoftConfig(**
          {
            "company": "microsoft",
            "model_name": "Orca-2-13b",
            "date_code": "",
            "temperature": 0.0,
          }
        ),
        MicrosoftConfig(**
          {
            "company": "microsoft",
            "model_name": "phi-2",
            "date_code": "",
            "temperature": 0.0,
          }
        ),
        MicrosoftConfig(**
          {
            "company": "microsoft",
            "model_name": "Phi-3-mini-4k-instruct",
            "date_code": "",
            "temperature": 0.0,
          }
        ),
        MicrosoftConfig(**
          {
            "company": "microsoft",
            "model_name": "Phi-3-mini-128k-instruct",
            "date_code": "",
            "temperature": 0.0,
          }
        ),
        MicrosoftConfig(**
          {
            "company": "microsoft",
            "model_name": "Phi-3.5-mini-instruct",
            "date_code": "",
            "temperature": 0.0,
          }
        ),
        MicrosoftConfig(**
          {
            "company": "microsoft",
            "model_name": "Phi-3.5-MoE-instruct",
            "date_code": "",
            "temperature": 0.0,
          }
        ),
        MicrosoftConfig(**
          {
            "company": "microsoft",
            "model_name": "phi-4",
            "date_code": "",
            "temperature": 0.0,
          }
        ),
        MicrosoftConfig(**
          {
            "company": "microsoft",
            "model_name": "Phi-4-mini-instruct",
            "date_code": "",
            "temperature": 0.0,
          }
        ),
        MicrosoftConfig(**
          {
            "company": "microsoft",
            "model_name": "WizardLM-2-8x22B",
            "date_code": "",
            "temperature": 0.0,
          }
        ),

        MistralAIConfig(**{"company": "mistralai", "model_name": "Ministral-8B-Instruct", "date_code": "2410", "temperature": 0.0}),
        MistralAIConfig(**{"company": "mistralai", "model_name": "Mistral-7B-Instruct-v0.3", "date_code": "", "temperature": 0.0}),
        MistralAIConfig(**{"company": "mistralai", "model_name": "mistral-medium", "date_code": "2508", "temperature": 0.0}),
        MistralAIConfig(**{"company": "mistralai", "model_name": "Mistral-Nemo-Instruct", "date_code": "2407", "temperature": 0.0}),
        MistralAIConfig(**{"company": "mistralai", "model_name": "mistral-small", "date_code": "2506", "temperature": 0.0}),
        MistralAIConfig(**{"company": "mistralai", "model_name": "Mistral-Small-3.1-24b-instruct", "date_code": "2503", "temperature": 0.0}),
        MistralAIConfig(**{"company": "mistralai", "model_name": "Mistral-Small-24B-Instruct", "date_code": "2501", "temperature": 0.0}),
        MistralAIConfig(**{"company": "mistralai", "model_name": "Mixtral-8x7B-Instruct-v0.1", "date_code": "", "temperature": 0.0}),
        MistralAIConfig(**{"company": "mistralai", "model_name": "Mixtral-8x22B-Instruct-v0.1", "date_code": "", "temperature": 0.0}),
        MistralAIConfig(**{"company": "mistralai", "model_name": "Pixtral-Large-Instruct", "date_code": "2411", "temperature": 0.0}),

        OpenAIConfig(**{"company": "openai", "model_name": "chatgpt-4o", "date_code": "", "temperature": -1.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "gpt-4-turbo", "date_code": "2024-04-09", "temperature": -1.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "gpt-4.1", "date_code": "2025-04-14", "temperature": -1.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "gpt-4.1-mini", "date_code": "2025-04-14", "temperature": -1.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "gpt-4.1-nano", "date_code": "2025-04-14", "temperature": -1.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "gpt-4.5-preview", "date_code": "2025-02-27", "temperature": -1.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "gpt-4o-mini", "date_code": "2024-07-18", "temperature": -1.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "gpt-5-high", "date_code": "2025-08-07", "temperature": -1.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "gpt-5-mini", "date_code": "2025-08-07", "temperature": -1.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "gpt-5-minimal", "date_code": "2025-08-07", "temperature": -1.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "gpt-5-nano", "date_code": "2025-08-07", "temperature": -1.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "gpt-oss-20b", "date_code": "", "temperature": -1.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "gpt-oss-120b", "date_code": "", "temperature": -1.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "o1", "date_code": "2024-12-17", "temperature": -1.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "o1-mini", "date_code": "2024-09-12", "temperature": -1.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "o1-preview", "date_code": "2024-09-12", "temperature": -1.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "o1-pro", "date_code": "2025-03-19", "temperature": -1.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "o3", "date_code": "2025-04-16", "temperature": -1.0}),
        OpenAIConfig(**{"company": "openai", "model_name": "o4-mini", "date_code": "2025-04-16", "temperature": -1.0}),

        QwenConfig(**{"company": "qwen", "model_name": "qwen-max", "date_code": "2025-01-25", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "Qwen2-72B-Instruct", "date_code": "", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "Qwen2-VL-2B-Instruct", "date_code": "", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "Qwen2-VL-7B-Instruct", "date_code": "", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "Qwen2.5-0.5B-Instruct", "date_code": "", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "Qwen2.5-1.5B-Instruct", "date_code": "", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "Qwen2.5-3B-Instruct", "date_code": "", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "Qwen2.5-7B-Instruct", "date_code": "", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "Qwen2.5-14B-Instruct", "date_code": "", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "Qwen2.5-32B-Instruct", "date_code": "", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "Qwen2.5-72B-Instruct", "date_code": "", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "Qwen3-0.6B", "date_code": "", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "Qwen3-1.7B", "date_code": "", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "Qwen3-4B", "date_code": "", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "Qwen3-8B", "date_code": "", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "Qwen3-14B", "date_code": "", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "Qwen3-30B-A3B", "date_code": "", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "Qwen3-32B", "date_code": "", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "Qwen3-235B-A22B", "date_code": "", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "QwQ-32B-Preview", "date_code": "", "temperature": 0.0}),

        SnowflakeConfig(**{"company": "snowflake", "model_name": "snowflake-arctic-instruct", "date_code": "", "temperature": 0.0}),

        TiiuaeConfig(**{"company": "tiiuae", "model_name": "falcon-7b-instruct", "date_code": "", "temperature": 0.0}),

        TngTechConfig(**{"company": "tngtech", "model_name": "DeepSeek-TNG-R1T2-Chimera", "date_code": "", "temperature": 0.0}),

        XAIConfig(**{"company": "xai-org", "model_name": "grok-3", "temperature": 0.0, "date_code": ""}),
        XAIConfig(**{"company": "xai-org", "model_name": "grok-3-mini", "temperature": 0.0, "date_code": ""}),
        XAIConfig(**{"company": "xai-org", "model_name": "grok-4", "temperature": 0.0, "date_code": "0709"}),

        ZhipuAIConfig(**{"company": "zai-org", "model_name": "glm-4-9b-chat", "date_code": "", "temperature": 0.0}),
        ZhipuAIConfig(**{"company": "zai-org", "model_name": "GLM-4.5-AIR-FP8", "date_code": "", "temperature": 0.0}),
        ZhipuAIConfig(**{"company": "zai-org", "model_name": "glm-4p5", "date_code": "", "temperature": 0.0}),
      ]
    }
  ),


  EvalConfig(**
    {
      "eval_name": "short_summary_2.3",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["summarize", "judge", "aggregate"],
      "output_dir": "output_short_summary_2.3",
      "overwrite_summaries": True,
      "source_article_path": "datasets/leaderboard_dataset_revised_and_bbc_dataset.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 1.0, 
            "max_tokens": 4096, 
            "prompt": """
You are a chat bot answering questions using data.
You must stick to the answers provided solely by the text in the 
passage provided. You are asked the question 'Provide a concise 
summary of the following passage, covering the core pieces of 
information described.' It is crucial to make the summary as short as possible.

If you cannot answer the question, for reasons like insufficient information in the passage, 
just say 'I cannot do it. 2389fdsi2389432ksad' and do not say anything else. 
  
Here is the passage:
{article}
""",
          }
        ),
      "per_LLM_configs": [
        GoogleConfig(**{"company": "google", "model_name": "gemini-2.5-flash", "date_code":"", "temperature": 0.0, "thinking_budget": -1}), #Odd bug with date code if its not set here?
      ]
    }
  ),
  EvalConfig(**
    {
      "eval_name": "long_summary_2.3",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["summarize", "judge", "aggregate"],
      "output_dir": "output_long_summary_2.3",
      "overwrite_summaries": True,
      "source_article_path": "datasets/leaderboard_dataset_revised_and_bbc_dataset.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 1.0, 
            "max_tokens": 4096, 
            "prompt": """
You are a chat bot answering questions using data.
You must stick to the answers provided solely by the text in the 
passage provided. You are asked the question 'Provide a  
summary of the following passage, covering the core pieces of 
information described.' It is crucial the summary covers ALL information
in the passage. This summary should be long but do not make it longer
than the original passage.

If you cannot answer the question, for reasons like insufficient information in the passage, 
just say 'I cannot do it. 2389fdsi2389432ksad' and do not say anything else. 

Here is the passage:
{article}
""",
          }
        ),
      "per_LLM_configs": [
        GoogleConfig(**{"company": "google", "model_name": "gemini-2.5-flash", "date_code":"", "temperature": 0.0, "thinking_budget": -1}), #Odd bug with date code if its not set here?
      ]
    }
  ),
  EvalConfig(**
    {
      "eval_name": "short_summary_2.1-open",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.1-open",
      "pipeline": ["judge", "aggregate"],
      "output_dir": "output_short_summary_2.1-open",
      "overwrite_summaries": True,
      "source_article_path": "datasets/leaderboard_dataset_revised_and_bbc_dataset.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 1.0, 
            "max_tokens": 4096, 
            "prompt": """
You are a chat bot answering questions using data.
You must stick to the answers provided solely by the text in the 
passage provided. You are asked the question 'Provide a concise 
summary of the following passage, covering the core pieces of 
information described.' It is crucial to make the summary as short as possible.

If you cannot answer the question, for reasons like insufficient information in the passage, 
just say 'I cannot do it. 2389fdsi2389432ksad' and do not say anything else. 
  
Here is the passage:
{article}
""",
          }
        ),
      "per_LLM_configs": [
        GoogleConfig(**{"company": "google", "model_name": "gemini-2.5-flash", "date_code":"", "temperature": 0.0, "thinking_budget": -1}), #Odd bug with date code if its not set here?
      ]
    }
  ),
  EvalConfig(**
    {
      "eval_name": "long_summary_2.1-open",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.1-open",
      "pipeline": ["judge", "aggregate"],
      "output_dir": "output_long_summary_2.1-open",
      "overwrite_summaries": True,
      "source_article_path": "datasets/leaderboard_dataset_revised_and_bbc_dataset.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 1.0, 
            "max_tokens": 4096, 
            "prompt": """
You are a chat bot answering questions using data.
You must stick to the answers provided solely by the text in the 
passage provided. You are asked the question 'Provide a  
summary of the following passage, covering the core pieces of 
information described.' It is crucial the summary covers ALL information
in the passage. This summary should be long but do not make it longer
than the original passage.

If you cannot answer the question, for reasons like insufficient information in the passage, 
just say 'I cannot do it. 2389fdsi2389432ksad' and do not say anything else. 

Here is the passage:
{article}
""",
          }
        ),
      "per_LLM_configs": [
        GoogleConfig(**{"company": "google", "model_name": "gemini-2.5-flash", "date_code":"", "temperature": 0.0, "thinking_budget": -1}), #Odd bug with date code if its not set here?
      ]
    }
  ),
  EvalConfig(**
    {
      "eval_name": "bbc",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["summarize", "judge", "aggregate"],
      "output_dir": "output_bbc",
      "overwrite_summaries": True,
      "source_article_path": "datasets/bbc_dataset.csv",
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
        AntGroupMIConfig(**
          {
            "company": "antgroup",
            "model_name": "antfinix-a1",
            "date_code": "",
            "temperature": 0.0,
          }
        ),
      ]
    }
  ),
  EvalConfig(**
    {
      "eval_name": "shuffle",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["summarize", "judge", "aggregate"],
      "output_dir": "shuffle_output",
      "overwrite_summaries": True,
      "source_article_path": "datasets/leaderboard_dataset_revised_shuffle.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 0.0, 
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
        GoogleConfig(**{"company": "google", "model_name": "gemini-2.0-flash", "date_code":"", "temperature": 0.0, "thinking_budget": -1}), #Odd bug with date code if its not set here?
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-sonnet-4", "date_code": "20250514", "temperature": 0.0}),
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-opus-4", "date_code": "20250514", "temperature": 0.0}),
        MoonshotAIConfig(**{"company": "moonshotai", "model_name": "Kimi-K2-Instruct", "temperature": 0.0, "min_throttle_time": 4.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen-max", "date_code": "2025-01-25", "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen-plus", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen-turbo", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-14b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-32b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-7b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-72b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-0.6b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-1.7b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-14b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-32b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-4b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-8b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
      ]
    }
  ),
  EvalConfig(**
    {
      "eval_name": "delete_first_half",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["summarize", "judge", "aggregate"],
      "output_dir": "delete_first_half_output",
      "overwrite_summaries": True,
      "source_article_path": "datasets/leaderboard_dataset_revised_delete_first_half.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 0.0, 
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
        GoogleConfig(**{"company": "google", "model_name": "gemini-2.0-flash", "date_code":"", "temperature": 0.0, "thinking_budget": -1}), #Odd bug with date code if its not set here?
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-sonnet-4", "date_code": "20250514", "temperature": 0.0}),
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-opus-4", "date_code": "20250514", "temperature": 0.0}),
        MoonshotAIConfig(**{"company": "moonshotai", "model_name": "Kimi-K2-Instruct", "temperature": 0.0, "min_throttle_time": 4.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen-max", "date_code": "2025-01-25", "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen-plus", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen-turbo", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-14b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-32b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-7b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-72b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-0.6b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-1.7b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-14b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-32b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-4b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-8b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
      ]
    }
  ),
  EvalConfig(**
    {
      "eval_name": "delete_second_half",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["summarize", "judge", "aggregate"],
      "output_dir": "delete_second_half_output",
      "overwrite_summaries": True,
      "source_article_path": "datasets/leaderboard_dataset_revised_delete_second_half.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 0.0, 
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
        GoogleConfig(**{"company": "google", "model_name": "gemini-2.0-flash", "date_code":"", "temperature": 0.0, "thinking_budget": -1}), #Odd bug with date code if its not set here?
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-sonnet-4", "date_code": "20250514", "temperature": 0.0}),
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-opus-4", "date_code": "20250514", "temperature": 0.0}),
        MoonshotAIConfig(**{"company": "moonshotai", "model_name": "Kimi-K2-Instruct", "temperature": 0.0, "min_throttle_time": 4.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen-max", "date_code": "2025-01-25", "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen-plus", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen-turbo", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-14b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-32b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-7b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-72b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-0.6b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-1.7b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-14b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-32b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-4b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-8b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
      ]
    }
  ),
  EvalConfig(**
    {
      "eval_name": "delete_half_random",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["summarize", "judge", "aggregate"],
      "output_dir": "delete_half_random_output",
      "overwrite_summaries": True,
      "source_article_path": "datasets/leaderboard_dataset_revised_delete_half_random.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 0.0, 
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
        GoogleConfig(**{"company": "google", "model_name": "gemini-2.0-flash", "date_code":"", "temperature": 0.0, "thinking_budget": -1}), #Odd bug with date code if its not set here?
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-sonnet-4", "date_code": "20250514", "temperature": 0.0}),
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-opus-4", "date_code": "20250514", "temperature": 0.0}),
        MoonshotAIConfig(**{"company": "moonshotai", "model_name": "Kimi-K2-Instruct", "temperature": 0.0, "min_throttle_time": 4.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen-max", "date_code": "2025-01-25", "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen-plus", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen-turbo", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-14b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-32b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-7b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-72b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-0.6b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-1.7b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-14b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-32b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-4b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-8b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
      ]
    }
  ),
  EvalConfig(**
    {
      "eval_name": "delete_fifth_random",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["summarize", "judge", "aggregate"],
      "output_dir": "delete_fifth_random_output",
      "overwrite_summaries": True,
      "source_article_path": "datasets/leaderboard_dataset_revised_delete_fifth_random.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 0.0, 
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
        GoogleConfig(**{"company": "google", "model_name": "gemini-2.0-flash", "date_code":"", "temperature": 0.0, "thinking_budget": -1}), #Odd bug with date code if its not set here?
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-sonnet-4", "date_code": "20250514", "temperature": 0.0}),
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-opus-4", "date_code": "20250514", "temperature": 0.0}),
        MoonshotAIConfig(**{"company": "moonshotai", "model_name": "Kimi-K2-Instruct", "temperature": 0.0, "min_throttle_time": 4.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen-max", "date_code": "2025-01-25", "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen-plus", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen-turbo", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-14b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-32b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-7b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-72b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-0.6b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-1.7b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-14b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-32b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-4b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-8b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
      ]
    }
  ),
  # Special Experiment: Predict End
  EvalConfig(**
    {
      "eval_name": "last_20_percent_summs",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["summarize"],
      "output_dir": "output_last_20_percent_predict",
      "overwrite_summaries": True,
      "source_article_path": "datasets/leaderboard_dataset_revised_delete_last_20_percent.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 1.0, 
            "max_tokens": 8192, 
            "prompt": """
Continue the passage starting at the very next character after the given incomplete passage below. Write the next part as it would appear in the same document; it may span a couple words, multiple sentences, or paragraphs. Stop when it feels natural to conclude the passage.

Just provide your answer without any prompt like "Here is the answer:" or any endings like "I hope I have answered your question." Do not repeat the provided passage and do not add commentary, headings, quotes, or anything else.

If you cannot finish the passage, just say 'I cannot do it' and do not say anything else. 

Here is the passage:
{article}
""",
          }
        ),
      "per_LLM_configs": [
        GoogleConfig(**{"company": "google", "model_name": "gemini-2.0-flash", "date_code":"", "temperature": 0.0, "thinking_budget": -1}), #Odd bug with date code if its not set here?
        # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-sonnet-4", "date_code": "20250514", "temperature": 0.0}),
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-opus-4", "date_code": "20250514", "temperature": 0.0}),
        # MoonshotAIConfig(**{"company": "moonshotai", "model_name": "Kimi-K2-Instruct", "temperature": 0.0, "min_throttle_time": 4.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen-max", "date_code": "2025-01-25", "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen-plus", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        # QwenConfig(**{"company": "qwen", "model_name": "qwen-turbo", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-14b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-32b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-7b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-72b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-0.6b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-1.7b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-14b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-32b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-4b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-8b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
      ]
    }
  ),
  EvalConfig(**
    {
      "eval_name": "last_20_percent_summs_eval",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["judge", "aggregate"],
      "output_dir": "output_last_20_percent_predict",
      "overwrite_summaries": True,
      "source_article_path": "datasets/leaderboard_dataset_revised_delete_first_80_percent.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 1.0, 
            "max_tokens": 8192, 
            "prompt": """
Continue the passage starting at the very next character after the given incomplete passage below. Write the next part as it would appear in the same document; it may span a couple words, multiple sentences, or paragraphs. Stop when it feels natural to conclude the passage.

Just provide your answer without any prompt like "Here is the answer:" or any endings like "I hope I have answered your question." Do not repeat the provided passage and do not add commentary, headings, quotes, or anything else.

If you cannot finish the passage, just say 'I cannot do it' and do not say anything else. 

Here is the passage:
{article}
""",
          }
        ),
      "per_LLM_configs": [
        GoogleConfig(**{"company": "google", "model_name": "gemini-2.0-flash", "date_code":"", "temperature": 0.0, "thinking_budget": -1}), #Odd bug with date code if its not set here?
        # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-sonnet-4", "date_code": "20250514", "temperature": 0.0}),
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-opus-4", "date_code": "20250514", "temperature": 0.0}),
        # MoonshotAIConfig(**{"company": "moonshotai", "model_name": "Kimi-K2-Instruct", "temperature": 0.0, "min_throttle_time": 4.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen-max", "date_code": "2025-01-25", "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen-plus", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        # QwenConfig(**{"company": "qwen", "model_name": "qwen-turbo", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-14b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-32b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-7b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-72b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-0.6b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-1.7b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-14b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-32b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-4b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-8b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
      ]
    }
  ),
  # End of Special Experiment
  # Special Experiment: Predict Beginning
  EvalConfig(**
    {
      "eval_name": "first_20_percent_summs",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["summarize"],
      "output_dir": "output_first_20_percent_predict",
      "overwrite_summaries": True,
      "source_article_path": "datasets/leaderboard_dataset_revised_delete_first_20_percent.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 1.0, 
            "max_tokens": 8192, 
            "prompt": """
The text at the beginning of the passage given below has been deleted. Given the rest of the passage output text such that it would appear at the beginning of the passage until it would seamlessly combine with the given incomplete passage. Your output with the given passage appended to it should create a complete passage.

Just provide your answer without any prompt like "Here is the answer:" or any endings like "I hope I have answered your question." Do not repeat the provided passage and do not add commentary, headings, quotes, or anything else. Do not add the newline character between your output and the passage unless it makes sense to add this character.

If you cannot finish the passage, just say 'I cannot do it' and do not say anything else. 

Here is the passage:
{article}
""",
          }
        ),
      "per_LLM_configs": [
        GoogleConfig(**{"company": "google", "model_name": "gemini-2.0-flash", "date_code":"", "temperature": 0.0, "thinking_budget": -1}), #Odd bug with date code if its not set here?
        # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-sonnet-4", "date_code": "20250514", "temperature": 0.0}),
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-opus-4", "date_code": "20250514", "temperature": 0.0}),
        # MoonshotAIConfig(**{"company": "moonshotai", "model_name": "Kimi-K2-Instruct", "temperature": 0.0, "min_throttle_time": 4.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen-max", "date_code": "2025-01-25", "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen-plus", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        # QwenConfig(**{"company": "qwen", "model_name": "qwen-turbo", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-14b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-32b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-7b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-72b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-0.6b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-1.7b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-14b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-32b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-4b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-8b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
      ]
    }
  ),
  EvalConfig(**
    {
      "eval_name": "first_20_percent_summs_eval",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["judge", "aggregate"],
      "output_dir": "output_first_20_percent_predict",
      "overwrite_summaries": True,
      "source_article_path": "datasets/leaderboard_dataset_revised_delete_last_80_percent.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 1.0, 
            "max_tokens": 8192, 
            "prompt": """
The text at the beginning of the passage given below has been deleted. Given the rest of the passage output text such that it would appear at the beginning of the passage until it would seamlessly combine with the given incomplete passage. Your output with the given passage appended to it should create a complete passage.

Just provide your answer without any prompt like "Here is the answer:" or any endings like "I hope I have answered your question." Do not repeat the provided passage and do not add commentary, headings, quotes, or anything else. Do not add the newline character between your output and the passage unless it makes sense to add this character.

If you cannot finish the passage, just say 'I cannot do it' and do not say anything else. 

Here is the passage:
{article}
""",
          }
        ),
      "per_LLM_configs": [
        GoogleConfig(**{"company": "google", "model_name": "gemini-2.0-flash", "date_code":"", "temperature": 0.0, "thinking_budget": -1}), #Odd bug with date code if its not set here?
        # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-sonnet-4", "date_code": "20250514", "temperature": 0.0}),
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-opus-4", "date_code": "20250514", "temperature": 0.0}),
        # MoonshotAIConfig(**{"company": "moonshotai", "model_name": "Kimi-K2-Instruct", "temperature": 0.0, "min_throttle_time": 4.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen-max", "date_code": "2025-01-25", "temperature": 0.0}), # AKA Qwen2.5-Max
        QwenConfig(**{"company": "qwen", "model_name": "qwen-plus", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        # QwenConfig(**{"company": "qwen", "model_name": "qwen-turbo", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-14b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-32b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-7b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-72b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-0.6b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-1.7b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-14b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen3-32b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-4b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-8b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
      ]
    }
  ),
  # End of Special Experiment
  # Special Experiment: Mask Predict
  EvalConfig(**
    {
      "eval_name": "mask_predict_summs",
      "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
      "hhem_version": "2.3",
      "pipeline": ["summarize"],
      "output_dir": "output_mask_predict",
      "overwrite_summaries": True,
      "source_article_path": "datasets/leaderboard_dataset_revised_masked.csv",
      "common_LLM_config": 
        BasicLLMConfig(**
          {
            "temperature": 1.0, 
            "max_tokens": 8192, 
            "prompt": """
You will be given a passage with a singular tag of the form <mask_id=n, words=m>. Your job is given the passage predict the m words.

Output the results in the following way

mask_id_NUMBER = m word string

Provide exactly m words, no more no less

The predicted words should fit in with the passage seamlessly

A word is defined as a continous string of non space characters made of letters/digits/apostrophe/hyphen with no spaces. (Don't | state-of-the-art | 1999)

Isolated punctuation symbols also count as words for example "( Hello World )" is 4 words

Other examples of single words include: 3-3 | "Hello, | $200million

Do not rewrite the other parts of the passage

Just provide your answer without any prompt like "Here is the answer:" or any endings like "I hope I have answered your question." Do not repeat the provided passage and do not add commentary, headings, quotes, or anything else.

If you cannot predict the mask just say "" and nothing else. 

Here is the passage:

{article}
""",
          }
        ),
      "per_LLM_configs": [
        GoogleConfig(**{"company": "google", "model_name": "gemini-2.0-flash", "date_code":"", "temperature": 0.0, "thinking_budget": -1}), #Odd bug with date code if its not set here?
        # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-sonnet-4", "date_code": "20250514", "temperature": 0.0}),
        AnthropicConfig(**{"company": "anthropic", "model_name": "claude-opus-4", "date_code": "20250514", "temperature": 0.0}),
        # MoonshotAIConfig(**{"company": "moonshotai", "model_name": "Kimi-K2-Instruct", "temperature": 0.0, "min_throttle_time": 4.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen-max", "date_code": "2025-01-25", "temperature": 0.0}), # AKA Qwen2.5-Max
        # QwenConfig(**{"company": "qwen", "model_name": "qwen-plus", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        # QwenConfig(**{"company": "qwen", "model_name": "qwen-turbo", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-14b-instruct", "temperature": 0.0}),
        QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-32b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-7b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-72b-instruct", "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-0.6b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-1.7b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-14b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-32b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-4b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
        # QwenConfig(**{"company": "qwen", "model_name": "qwen3-8b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
      ]
    }
  ),
#   EvalConfig(** # Not completed yet
#     {
#       "eval_name": "mask_predict_summs_eval",
#       "eval_date": datetime.now().strftime('%Y-%m-%d'), #today
#       "hhem_version": "2.3",
#       "pipeline": ["judge", "aggregate"],
#       "output_dir": "output_mask_predict",
#       "overwrite_summaries": True,
#       "source_article_path": "datasets/leaderboard_dataset_revised_masked.csv",
#       "common_LLM_config": 
#         BasicLLMConfig(**
#           {
#             "temperature": 1.0, 
#             "max_tokens": 8192, 
#             "prompt": """
# Continue the passage starting at the very next character after the given incomplete passage below. Write the next part as it would appear in the same document; it may span a couple words, multiple sentences, or paragraphs. Stop when it feels natural to conclude the passage.
# 
# Just provide your answer without any prompt like "Here is the answer:" or any endings like "I hope I have answered your question." Do not repeat the provided passage and do not add commentary, headings, quotes, or anything else.
# 
# If you cannot finish the passage, just say 'I cannot do it' and do not say anything else. 
# 
# Here is the passage:
# {article}
# """,
#           }
#         ),
#       "per_LLM_configs": [
#         GoogleConfig(**{"company": "google", "model_name": "gemini-2.0-flash", "date_code":"", "temperature": 0.0, "thinking_budget": -1}), #Odd bug with date code if its not set here?
#         # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-sonnet-4", "date_code": "20250514", "temperature": 0.0}),
#         # AnthropicConfig(**{"company": "anthropic", "model_name": "claude-opus-4", "date_code": "20250514", "temperature": 0.0}),
#         # MoonshotAIConfig(**{"company": "moonshotai", "model_name": "Kimi-K2-Instruct", "temperature": 0.0, "min_throttle_time": 4.0}),
#         # QwenConfig(**{"company": "qwen", "model_name": "qwen-max", "date_code": "2025-01-25", "temperature": 0.0}), # AKA Qwen2.5-Max
#         # QwenConfig(**{"company": "qwen", "model_name": "qwen-plus", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
#         # QwenConfig(**{"company": "qwen", "model_name": "qwen-turbo", "date_code": "2025-04-28", "enable_thinking": False, "temperature": 0.0}), # AKA Qwen2.5-Max
#         # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-14b-instruct", "temperature": 0.0}),
#         # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-32b-instruct", "temperature": 0.0}),
#         # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-7b-instruct", "temperature": 0.0}),
#         # QwenConfig(**{"company": "qwen", "model_name": "qwen2.5-72b-instruct", "temperature": 0.0}),
#         # QwenConfig(**{"company": "qwen", "model_name": "qwen3-0.6b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
#         # QwenConfig(**{"company": "qwen", "model_name": "qwen3-1.7b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
#         # QwenConfig(**{"company": "qwen", "model_name": "qwen3-14b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
#         # QwenConfig(**{"company": "qwen", "model_name": "qwen3-32b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
#         # QwenConfig(**{"company": "qwen", "model_name": "qwen3-4b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
#         # QwenConfig(**{"company": "qwen", "model_name": "qwen3-8b", "thinking_tokens": 0, "enable_thinking": False, "temperature": 0.0}),
#       ]
#     }
#   ),
]