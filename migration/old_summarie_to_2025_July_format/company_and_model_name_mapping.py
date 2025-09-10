# This file defines how to substitute company/model names in our old HF summary dataset with official names on HuggingFace and or their own API, in whole or partially

# Notes

# 1. For cohere/Cohere and cohere/Cohere-Chat, see old README https://github.com/vectara/hallucination-leaderboard/tree/hhem-1.0-final?tab=readme-ov-file#cohere-models
# 2. For mistral models, this offical model listing has 

# Only models that are mentioned below will be migrated. Others will not, mostly because of missing date code. They will be re-evaluated later.

# Full List
    # 01-ai - Correct
        # Models Correct
    # ai21labs - Correct
        # Errors (Done)
    # allenai - Correct
        # Errors (DONE)
    # amazon - Correct
        # Errors (Done)
    # anthropic - Correct
        # Errors (Done)
    # apple - Correct
        # Models Correct
    # !!! cohere -> CohereLabs
        # Errors (Done)
    # databricks - Correct
        # Models Correct
    # !!! deepseek -> deepseek-ai
        # Errors (Done)
    # google - Correct
        # Errors (Done)
    # ibm-granite - Correct
        # Models Correct
    # intel - Correct
        # Models correct
    # internlm - Correct
        # Models Correct
    # meta-llama - Correct
        # Errors (Done)
    # microsoft - Correct
        # Errors
    # mistralai - Correct
    # openai - Correct
    # qwen - Correct
    # snowflake - Correct
    # thudm - Correct
    # tiiuae - Correct
    # !!! xai -> xai-org

name_mapping = {
    'xai' : 'xai-org',
    'cohere': 'CohereLabs',
    'deepseek': 'deepseek-ai',
    'THUDM': 'zai-org',

    '01-ai/Yi-1.5-6B-Chat': '01-ai/Yi-1.5-6B-Chat', # Same
    '01-ai/Yi-1.5-9B-Chat': '01-ai/Yi-1.5-9B-Chat', # Same
    '01-ai/Yi-1.5-34B-Chat': '01-ai/Yi-1.5-34B-Chat', # Same

    'ai21labs/AI21-Jamba-1.5-Mini': 'ai21labs/AI21-Jamba-Mini-1.5',
    'ai21labs/jamba-1.6-large': 'ai21labs/AI21-Jamba-Large-1.6',
    'ai21labs/jamba-1.6-mini': 'ai21labs/AI21-Jamba-Mini-1.6',

    'allenai/olmo-2-0325-32b-instruct': 'allenai/OLMo-2-0325-32B-Instruct',
    'allenai/OLMo-2-1124-7b-instruct': 'allenai/OLMo-2-1124-7b-instruct', # Same
    'allenai/OLMo-2-1124-13b-instruct': 'allenai/OLMo-2-1124-13b-instruct', # Same

    'amazon/nova-lite-v1': 'amazon/nova-lite-v1:0',
    'amazon/nova-micro-v1': 'amazon/nova-micro-v1:0',
    'amazon/nova-pro-v1': 'amazon/nova-pro-v1:0',
    # TODO 'amazon/Titan-Express': 'amazon/titan-text-express-v1', # MAYBE: Ambiguous can skip

    'anthropic/Claude-2': 'anthropic/claude-2.0',
    'anthropic/claude-3-5-haiku': 'anthropic/claude-3-5-haiku', # Same
    'anthropic/claude-3-5-sonnet': 'anthropic/claude-3-5-sonnet-20241022',
    'anthropic/claude-3-5-sonnet-20240620': 'anthropic/claude-3-5-sonnet-20240620', # Same
    # TODO 'anthropic/claude-3-7-sonnet-latest': 'ambiguous',
    # TODO 'anthropic/claude-3-7-sonnet-latest-think': 'ambiguous',
    'anthropic/Claude-3-opus': 'anthropic/claude-3-opus-20240229',
    'anthropic/Claude-3-sonnet': 'anthropic/claude-3-sonnet-20240229',

    'apple/OpenELM-3B-Instruct': 'apple/OpenELM-3B-Instruct', # Same

    'cohere/c4ai-aya-expanse-8b': 'CohereLabs/aya-expanse-8b', # Open source on HF
    'cohere/c4ai-aya-expanse-32b': 'CohereLabs/aya-expanse-32b', # Open source on HF
    'CohereForAI/c4ai-command-r-plus' : 'CohereLabs/c4ai-command-r-plus', # Open source on HF
    'cohere/Cohere': 'CohereLabs/command', # commerical
    'cohere/Cohere-Chat': 'CohereLabs/command-chat', # the same Cohere Command model but using the /chat endpoint
    'cohere/command-a': 'CohereLabs/command-a-03-2025', # Open source on HF
    'cohere/command-r': 'CohereLabs/command-r-08-2024', # Open source on HF
    'cohere/command-r-plus': 'CohereLabs/command-r-plus-08-2024',

    'databricks/dbrx-instruct': 'databricks/dbrx-instruct', # Same

    # TODO 'deepseek/deepseek-chat': 'ambiguous',
    'deepseek/deepseek-r1': 'deepseek-ai/DeepSeek-R1',
    'deepseek/deepseek-v3': 'deepseek-ai/DeepSeek-V3',
    'deepseek/deepseek-v3-0324': 'deepseek-ai/DeepSeek-V3-0324',

    'google/flan-t5-large': 'google/flan-t5-large', 
    'google/Gemini-1.5-flash': 'google/gemini-1.5-flash',
    # TODO 'google/gemini-1.5-flash-001': 'google/gemini-1.5-flash-001', # Model not listed online
    # TODO 'google/gemini-1.5-flash-002': 'google/gemini-1.5-flash-002', # Model not listed online
    'google/Gemini-1.5-Pro': 'google/gemini-1.5-pro',
    # TODO 'google/gemini-1.5-pro-001': 'google/gemini-1.5-pro-001', # Model not listed online
    # TODO 'google/gemini-1.5-pro-002': 'google/gemini-1.5-pro-002', # Model not listed online
    # TODO 'google/gemini-2.0-flash-001': 'google/gemini-2.0-flash-001' # Model not listed but it is mentioned?
    'google/gemini-2.0-flash-exp': 'google/gemini-2.0-flash-exp',
    # TODO 'google/gemini-2.0-flash-lite-preview': 'google/gemini-2.0-flash-lite-preview' # Model not listed
    # TODO 'google/gemini-2.0-flash-thinking-exp': 'ambiguous' # Date code could be 01-21 or 1219
    'google/gemini-2.0-pro-exp': 'google/gemini-2.0-pro-exp-02-05',
    'google/gemini-2.5-flash-preview': 'google/gemini-2.5-flash-preview-04-17',
    # TODO 'google/gemini-2.5-pro-exp': 'ambiguous', #Date Code could be 01-21 or 1219
    # TODO 'google/gemini-flash-experimental': 'google/gemini-flash-experimental', #Model not listed anywhere but naming pattern is correct
    # TODO 'google/Gemini-Pro': 'google/gemini-pro', # Model not listed anywhere
    # TODO 'google/gemini-pro-experimental': 'gemini-pro-experimental', #Model not listed anywhere
    'google/gemma-1.1-2b-it': 'google/gemma-1.1-2b-it',
    'google/gemma-1.1-7b-it': 'google/gemma-1.1-7b-it',
    'google/gemma-2-2b-it': 'gooogle/gemma-2-2b-it',
    'google/gemma-2-9b-it': 'gooogle/gemma-2-9b-it',
    'google/gemma-3-1b-it': 'gooogle/gemma-3-1b-it',
    'google/gemma-3-4b-it': 'gooogle/gemma-3-4b-it',
    'google/gemma-3-27b-it': 'gooogle/gemma-3-27b-it',
    'google/gemma-7b-it': 'gooogle/gemma-7b-it',
    # TODO 'google/PaLM-2': 'google/test-bison-001', # Commenting out for now but according to docs this is the reference internally(https://ai.google.dev/palm_docs/palm)
    # TODO 'google/PaLM-2-Chat': 'google/chat-bison-001', # Commenting out for now but according to docs this is the reference internally(https://ai.google.dev/palm_docs/palm)

    'ibm-granite/granite-3.0-2b-instruct': 'ibm-granite/granite-3.0-2b-instruct',
    'ibm-granite/granite-3.0-8b-instruct': 'ibm-granite/granite-3.0-8b-instruct',
    'ibm-granite/granite-3.1-2b-instruct': 'ibm-granite/granite-3.1-2b-instruct',
    'ibm-granite/granite-3.1-8b-instruct': 'ibm-granite/granite-3.1-8b-instruct',
    'ibm-granite/granite-3.2-2b-instruct': 'ibm-granite/granite-3.2-2b-instruct',
    'ibm-granite/granite-3.2-8b-instruct': 'ibm-granite/granite-3.2-8b-instruct',

    'Intel/neural-chat-7b-v3-3': 'Intel/neural-chat-7b-v3-3',

    'internlm/internlm3-8b-instruct': 'internlm/internlm3-8b-instruct',

    'meta-llama/Llama-2-7b-chat-hf': 'meta-llama/Llama-2-7b-chat-hf',
    'meta-llama/Llama-2-13b-chat-hf': 'meta-llama/Llama-2-13b-chat-hf',
    'meta-llama/Llama-2-70b-chat-hf': 'meta-llama/Llama-2-70b-chat-hf',
    # TODO 'meta-llama/Llama-3-8B-chat-hf': 'Not On Huggingface',
    # TODO 'meta-llama/Llama-3-70B-chat-hf': 'Not On Huggingface',
    'meta-llama/': 'meta-llama/',
    # TODO 'meta-llama/Llama-3.2-3B-Instruct-Turbo': 'Not On Huggingface',
    # TODO 'meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo': 'Not On Huggingface',
    # TODO 'meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo': 'Not On Huggingface',
    'meta-llama/Llama-3.3-70B-Instruct': 'meta-llama/Llama-3.3-70B-Instruct',
    # TODO 'meta-llama/llama-4-maverick': 'ambiguous',
    # TODO 'meta-llama/llama-4-scout': 'ambiguous',
    'meta-llama/Meta-Llama-3.1-8b-Instruct': 'meta-llama/Meta-Llama-3.1-8b-Instruct',
    'meta-llama/Meta-Llama-3.1-70B-Instruct': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'meta-llama/Meta-Llama-3.1-405B-Instruct': 'meta-llama/Meta-Llama-3.1-405B-Instruct',

    'microsoft/Orca-2-13b': 'microsoft/Orca-2-13b',
    'microsoft/Phi-2': 'microsoft/phi-2',
    'microsoft/Phi-3-mini-4k-instruct': 'microsoft/Phi-3-mini-4k-instruct',
    'microsoft/Phi-3-mini-128k-instruct': 'microsoft/Phi-3-mini-128k-instruct',
    'microsoft/Phi-3.5-mini-instruct': 'microsoft/Phi-3.5-mini-instruct',
    'microsoft/Phi-3.5-MoE-instruct': 'microsoft/Phi-3.5-MoE-instruct',
    'microsoft/phi-4': 'microsoft/phi-4',
    'microsoft/Phi-4-mini-instruct': 'microsoft/Phi-4-mini-instruct',
    'microsoft/WizardLM-2-8x22B': 'microsoft/WizardLM-2-8x22B',

    # TODO 'mistralai/ministral-3b-latest': 'ambiguous', # I cannot find this model. There is only ministral-3b-instruct. 
    'mistralai/ministral-8b-latest': 'mistralai/Ministral-8B-Instruct-2410',  # there seems to be only one ministral-8b. So i assume it is this one
    'mistralai/Mistral-7B-Instruct-v0.3': 'mistralai/Mistral-7B-Instruct-v0.3',
    # TODO 'mistralai/mistral-large-latest': 'ambiguous', # I cannot determine the version or date code. But 
    # TODO 'mistralai/Mistral-Large2': 'ambiguous', # I cannot determine the version or date code.
    'mistralai/Mistral-Nemo-Instruct-2407': 'mistralai/Mistral-Nemo-Instruct-2407',
    'mistralai/mistral-small-3.1-24b-instruct': 'mistralai/Mistral-Small-3.1-24b-instruct-2503',  # I cannot find the model mistral-small-3.1-24b-instruct in google. the closest I can find is mistral-small-3.1-24b-instruct-2503. 
    'mistralai/Mistral-Small-24B-Instruct-2501': 'mistralai/Mistral-Small-24B-Instruct-2501',
    # TODO 'mistralai/mistral-small-latest': 'ambiguous, # I cannot determine the version or date code. Mistral-small has v.2, v.3.x.
    'mistralai/Mixtral-8x7B-Instruct-v0.1': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'mistralai/Mixtral-8x22B-Instruct-v0.1': 'mistralai/Mixtral-8x22B-Instruct-v0.1',
    'mistralai/pixtral-large-latest': 'mistralai/Pixtral-Large-Instruct-2411', # there is only one version pixal-large so far, which is 2411

    'openai/chatgpt-4o': 'openai/chatgpt-4o-latest',
    # TODO 'openai/GPT-3.5-Turbo': 'ambiguous', two date code versions
    # TODO 'openai/GPT-4': 'ambiguous', two date code versions
    'openai/GPT-4-Turbo': 'openai/gpt-4-turbo-2024-04-09',
    'openai/gpt-4.1': 'openai/gpt-4.1-2025-04-14',
    'openai/gpt-4.1-mini': 'openai/gpt-4.1-mini-2025-04-14',
    'openai/gpt-4.1-nano': 'openai/gpt-4.1-nano-2025-04-14',
    'openai/gpt-4.5-preview': 'openai/gpt-4.5-preview-2025-02-27',
    # 'openai/gpt-4o': 'ambiguous', 3 date codes
    'openai/GPT-4o-mini': 'openai/gpt-4o-mini-2024-07-18',
    'openai/o1': 'openai/o1-2024-12-17',
    'openai/o1-mini': 'openai/o1-mini-2024-09-12',
    'openai/o1-preview': 'openai/o1-preview-2024-09-12',
    'openai/o1-pro': 'openai/o1-pro-2025-03-19',
    'openai/o3': 'openai/o3-2025-04-16',
    'openai/o4-mini': 'openai/o4-mini-2025-04-16',

    # qwen
    'qwen/qwen-max': 'qwen/qwen-max-2025-01-25',
    'qwen/Qwen2-72B-Instruct': 'qwen/Qwen2-72B-Instruct',
    'qwen/Qwen2-VL-2B-Instruct': 'qwen/Qwen2-VL-2B-Instruct',
    'qwen/Qwen2-VL-7B-Instruct': 'qwen/Qwen2-VL-7B-Instruct',
    'qwen/Qwen2.5-0.5B-Instruct': 'qwen/Qwen2.5-0.5B-Instruct',
    'qwen/Qwen2.5-1.5B-Instruct': 'qwen/Qwen2.5-1.5B-Instruct',
    'qwen/Qwen2.5-3B-Instruct': 'qwen/Qwen2.5-3B-Instruct',
    'qwen/Qwen2.5-7B-Instruct': 'qwen/Qwen2.5-7B-Instruct',
    'qwen/Qwen2.5-14B-Instruct': 'qwen/Qwen2.5-14B-Instruct',
    'qwen/Qwen2.5-32B-Instruct': 'qwen/Qwen2.5-32B-Instruct',
    'qwen/Qwen2.5-72B-Instruct': 'qwen/Qwen2.5-72B-Instruct',
    'qwen/Qwen3-0.6B': 'qwen/Qwen3-0.6B',
    'qwen/Qwen3-1.7B': 'qwen/Qwen3-1.7B',
    'qwen/Qwen3-4B': 'qwen/Qwen3-4B',
    'qwen/Qwen3-8B': 'qwen/Qwen3-8B',
    'qwen/Qwen3-14B': 'qwen/Qwen3-14B',
    'qwen/qwen3-30b-a3b': 'qwen/Qwen3-30B-A3B',
    'qwen/Qwen3-32B': 'qwen/Qwen3-32B',
    'qwen/qwen3-235b-a22b': 'qwen/Qwen3-235B-A22B',
    'qwen/QwQ-32B-Preview': 'qwen/QwQ-32B-Preview',

    'snowflake/snowflake-arctic-instruct': 'snowflake/snowflake-arctic-instruct',

    'THUDM/glm-4-9b-chat': 'zai-org/glm-4-9b-chat',

    'tiiuae/falcon-7b-instruct': 'tiiuae/falcon-7b-instruct',

    'xai/grok-2-1212': 'xai-org/grok-2-1212',
    'xai/grok-2-vision-1212': 'xai-org/grok-2-vision-1212',
    'xai/grok-3-latest': 'xai-org/grok-3',
    'xai/grok-3-mini-latest': 'xai-org/grok-3-mini',
    # TODO 'xai/grok-beta': 'ambiguous', Grok-1-beta?

    # 'gemini-2.0-flash-exp' : 'google/gemini-2.0-flash-exp', shouldn't need this one
}