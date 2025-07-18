# Code for migrating between different version of the LB

## 2025-07-17:  From HF summaries repo `leaderboard_results` to July 2025 format. 

The working directory for this migration is `old_summarie_to_2025_July_format/`

Vectara has two dataset repos on HF: 
1. `vectarra/leaderboard_results`: The summaries produced by LLMs.
2. `vectarra/results`: The hallucination rates, answer rates, avg. word count, etc. of each LLM.

Convert summaries previously generated from HuggingFace Dataset repo https://huggingface.co/datasets/vectara/leaderboard_results to new output format of the LB.

Migration notes: 
1. The Google Drive snapshot of summaries is no longer needed. It was last updated in July 2024. It's superset is the HF dataset repo `vectarra/leaderboard_results`.
2. `gpt-4-turbo` in HF repo `vectara/results` should have been `openai/gpt-4-turbo-2024-04-09`. But no need to change as we will discard `vectara/results`. 
3. The summaries for `gemma-3-12b-it` and `google/gemini-2.5-pro-preview-06-05` cannot be found in `vectara/leaderboard_results` although they are on our LB. We may or may not need to reproduce it. 
4. [TODO] Unifying company/model names (Please expand the mapping in `company_and_model_name_mapping.py` and use the mapping to unify company/model names in `convert_HF_summaries_to_2025_July_format_summaries_jsonl.ipynb`): 
  - Merge company names: 
    - `x-ai` to `xai`. 
    - `gemini-2.0-flash-exp` to `google/gemini-2.0-flash-exp`
    - `anthropic` to `Anthropic`
  - For non-open-source models, use their names in their official API
    - Anthropic's official model names (see `anthropic_list_models.py`)
    - OpenAI's official model names (see `openai_list_models.py`)
    - TODO: add more such for Google, xai, etc. If they don't have offical API to list models, check their docs. 
  - For open source models, use their company name and model name as in HF. Keep the case consistent. 
   

Summaries of several models, such as Mockingbird, are missing from the `leaderboard_results` dataset. We only have the following model's summaries in the `leaderboard_results` dataset: 

```
'01-ai/Yi-1.5-34B-Chat',
 '01-ai/Yi-1.5-6B-Chat',
 '01-ai/Yi-1.5-9B-Chat',
 'Anthropic/claude-3-5-sonnet-20240620',
 'CohereForAI/c4ai-command-r-plus',
 'Intel/neural-chat-7b-v3-3',
 'Qwen/QwQ-32B-Preview',
 'Qwen/Qwen2-72B-Instruct',
 'Qwen/Qwen2-VL-2B-Instruct',
 'Qwen/Qwen2-VL-7B-Instruct',
 'Qwen/Qwen2.5-0.5B-Instruct',
 'Qwen/Qwen2.5-1.5B-Instruct',
 'Qwen/Qwen2.5-14B-Instruct',
 'Qwen/Qwen2.5-32B-Instruct',
 'Qwen/Qwen2.5-3B-Instruct',
 'Qwen/Qwen2.5-72B-Instruct',
 'Qwen/Qwen2.5-7B-Instruct',
 'Qwen/Qwen3-0.6B',
 'Qwen/Qwen3-1.7B',
 'Qwen/Qwen3-14B',
 'Qwen/Qwen3-32B',
 'Qwen/Qwen3-4B',
 'Qwen/Qwen3-8B',
 'THUDM/glm-4-9b-chat',
 'ai21/jamba-1.6-large',
 'ai21/jamba-1.6-mini',
 'ai21labs/AI21-Jamba-1.5-Mini',
 'allenai/OLMo-2-1124-13B-Instruct',
 'allenai/OLMo-2-1124-7B-Instruct',
 'allenai/olmo-2-0325-32b-instruct',
 'amazon/Titan-Express',
 'amazon/nova-lite-v1',
 'amazon/nova-micro-v1',
 'amazon/nova-pro-v1',
 'anthropic/Claude-2',
 'anthropic/Claude-3-opus',
 'anthropic/Claude-3-sonnet',
 'anthropic/claude-3-5-haiku-20241022',
 'anthropic/claude-3-5-sonnet-20241022',
 'anthropic/claude-3-7-sonnet-latest',
 'anthropic/claude-3-7-sonnet-latest-think',
 'apple/OpenELM-3B-Instruct',
 'cohere/Cohere',
 'cohere/Cohere-Chat',
 'cohere/c4ai-aya-expanse-32b',
 'cohere/c4ai-aya-expanse-8b',
 'cohere/command-a-03-2025',
 'cohere/command-r-08-2024',
 'cohere/command-r-plus-08-2024',
 'databricks/dbrx-instruct',
 'deepseek/deepseek-chat',
 'deepseek/deepseek-r1',
 'deepseek/deepseek-v3',
 'deepseek/deepseek-v3-0324',
 'gemini-2.0-flash-exp',
 'google/Gemini-1.5-Pro',
 'google/Gemini-1.5-flash',
 'google/Gemini-Pro',
 'google/PaLM-2',
 'google/PaLM-2-Chat',
 'google/flan-t5-large',
 'google/gemini-1.5-flash-001',
 'google/gemini-1.5-flash-002',
 'google/gemini-1.5-pro-001',
 'google/gemini-1.5-pro-002',
 'google/gemini-2.0-flash-001',
 'google/gemini-2.0-flash-lite-preview-02-05',
 'google/gemini-2.0-flash-thinking-exp',
 'google/gemini-2.0-pro-exp-02-05',
 'google/gemini-2.5-flash-preview-04-17',
 'google/gemini-2.5-pro-exp-03-25',
 'google/gemini-flash-experimental',
 'google/gemini-pro-experimental',
 'google/gemma-1.1-2b-it',
 'google/gemma-1.1-7b-it',
 'google/gemma-2-2b-it',
 'google/gemma-2-9b-it',
 'google/gemma-3-1b-it',
 'google/gemma-3-27b-it',
 'google/gemma-3-4b-it',
 'google/gemma-7b-it',
 'ibm-granite/granite-3.0-2b-instruct',
 'ibm-granite/granite-3.0-8b-instruct',
 'ibm-granite/granite-3.1-2b-instruct',
 'ibm-granite/granite-3.1-8b-instruct',
 'ibm-granite/granite-3.2-2b-instruct',
 'ibm-granite/granite-3.2-8b-instruct',
 'internlm/internlm3-8b-instruct',
 'meta-llama/Llama-2-13b-chat-hf',
 'meta-llama/Llama-2-70b-chat-hf',
 'meta-llama/Llama-2-7b-chat-hf',
 'meta-llama/Llama-3-70B-chat-hf',
 'meta-llama/Llama-3-8B-chat-hf',
 'meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo',
 'meta-llama/Llama-3.2-1B-Instruct',
 'meta-llama/Llama-3.2-3B-Instruct-Turbo',
 'meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo',
 'meta-llama/Llama-3.3-70B-Instruct',
 'meta-llama/Meta-Llama-3.1-405B-Instruct',
 'meta-llama/Meta-Llama-3.1-70B-Instruct',
 'meta-llama/Meta-Llama-3.1-8B-Instruct',
 'meta-llama/llama-4-maverick',
 'meta-llama/llama-4-scout',
 'microsoft/Orca-2-13b',
 'microsoft/Phi-2',
 'microsoft/Phi-3-mini-128k-instruct',
 'microsoft/Phi-3-mini-4k-instruct',
 'microsoft/Phi-3.5-MoE-instruct',
 'microsoft/Phi-3.5-mini-instruct',
 'microsoft/Phi-4-mini-instruct',
 'microsoft/WizardLM-2-8x22B',
 'microsoft/phi-4',
 'mistralai/Mistral-7B-Instruct-v0.3',
 'mistralai/Mistral-Large2',
 'mistralai/Mistral-Nemo-Instruct-2407',
 'mistralai/Mistral-Small-24B-Instruct-2501',
 'mistralai/Mixtral-8x22B-Instruct-v0.1',
 'mistralai/Mixtral-8x7B-Instruct-v0.1',
 'mistralai/ministral-3b-latest',
 'mistralai/ministral-8b-latest',
 'mistralai/mistral-large-latest',
 'mistralai/mistral-small-3.1-24b-instruct',
 'mistralai/mistral-small-latest',
 'mistralai/pixtral-large-latest',
 'openai/GPT-3.5-Turbo',
 'openai/GPT-4',
 'openai/GPT-4-Turbo-2024-04-09',
 'openai/GPT-4o-mini',
 'openai/chatgpt-4o-latest',
 'openai/gpt-4.1',
 'openai/gpt-4.1-mini',
 'openai/gpt-4.1-nano',
 'openai/gpt-4.5-preview',
 'openai/gpt-4o',
 'openai/o1',
 'openai/o1-mini',
 'openai/o1-preview',
 'openai/o1-pro',
 'openai/o3',
 'openai/o4-mini',
 'qwen/qwen-max',
 'qwen/qwen3-235b-a22b',
 'qwen/qwen3-30b-a3b',
 'snowflake/snowflake-arctic-instruct',
 'tiiuae/falcon-7b-instruct',
 'x-ai/grok-2-1212',
 'x-ai/grok-2-vision-1212',
 'xai/grok-3-latest',
 'xai/grok-3-mini-latest',
 'xai/grok-beta'
 ```





