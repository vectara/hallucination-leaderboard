# This file defines how to substitute company/model names in our old HF summary dataset with official names on HuggingFace and or their own API, in whole or partially

# Notes

# 1. For cohere/Cohere and cohere/Cohere-Chat, see old README https://github.com/vectara/hallucination-leaderboard/tree/hhem-1.0-final?tab=readme-ov-file#cohere-models
# 2. For mistral models, this offical model listing has 

# Only models that are mentioned below will be migrated. Others will not, mostly because of missing date code. They will be re-evaluated later.

# Full List
    # 01-ai - Correct
        # Models Correct
    # ai21labs - Correct
        # Model names incorrect (Done)
    # allenai - Correct
        # Model names incorrect (DONE)
    # amazon - Correct
        # Model Names Incorrect (Done)
    # anthropic - Correct
        # Model Names incorrect (Done)
    # apple - Correct
        # Models Correct
    # !!! cohere -> CohereLabs
        # Model names incorrect (Done)
    # databricks - Correct
        # Models Correct
    # !!! deepseek -> deepseek-ai
        # Models Incorrect (Done)
    # google - Correct
        # Models Incorrect (Done but needs review)
    # ibm-granite - Correct
        # Models Correct
    # intel - Correct
        # Models correct
    # internlm - Correct
        # Models Correct
    # meta-llama - Correct
    # microsoft - Correct
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

    'ai21labs/AI21-Jamba-1.5-Mini': 'ai21labs/AI21-Jamba-Mini-1.5',
    'ai21labs/jamba-1.6-large': 'ai21labs/AI21-Jamba-Large-1.6',
    'ai21labs/jamba-1.6-mini': 'ai21labs/AI21-Jamba-Mini-1.6',

    'allenai/olmo-2-0325-32b-instruct': 'allenai/OLMo-2-0325-32B-Instruct',

    'amazon/nova-lite-v1': 'amazon/nova-lite-v1:0',
    'amazon/nova-micro-v1': 'amazon/nova-micro-v1:0',
    'amazon/nova-pro-v1': 'amazon/nova-pro-v1:0',
    'amazon/Titan-Express': 'amazon/titan-text-express-v1', # MAYBE: Ambiguous can skip

    'anthropic/Claude-2': 'anthropic/claude-2.0',
    'anthropic/claude-3-5-sonnet': 'anthropic/claude-3-5-sonnet-20241022',
    'anthropic/claude-3-7-sonnet-latest': 'ambiguous',
    'anthropic/claude-3-7-sonnet-latest-think': 'ambiguous',
    'anthropic/Claude-3-opus': 'anthropic/claude-3-opus-20240229',
    'anthropic/Claude-3-sonnet': 'anthropic/claude-3-sonnet-20240229',

    'cohere/c4ai-aya-expanse-8b': 'CohereLabs/aya-expanse-8b', # Open source on HF
    'cohere/c4ai-aya-expanse-32b': 'CohereLabs/aya-expanse-32b', # Open source on HF
    'CohereForAI/c4ai-command-r-plus' : 'CohereLabs/c4ai-command-r-plus', # Open source on HF
    'cohere/Cohere': 'CohereLabs/command', # commerical
    'cohere/Cohere-Chat': 'CohereLabs/command-chat', # the same Cohere Command model but using the /chat endpoint
    'cohere/command-a': 'CohereLabs/command-a-03-2025', # Open source on HF
    'cohere/command-r': 'CohereLabs/command-r-08-2024', # Open source on HF
    'cohere/command-r-plus': 'CohereLabs/command-r-plus-08-2024',

    'deepseek/deepseek-chat': 'ambiguous',
    'deepseek/deepseek-r1': 'deepseek-ai/DeepSeek-R1',
    'deepseek/deepseek-v3': 'deepseek-ai/DeepSeek-V3',
    'deepseek/deepseek-v3-0324': 'deepseek-ai/DeepSeek-V3-0324',

    'google/Gemini-1.5-flash': 'google/gemini-1.5-flash',
    'google/Gemini-1.5-Pro': 'google/gemini-1.5-pro',
    'google/gemini-2.5-pro-exp': 'ambiguous', #Date Code could be 01-21 or 1219
    # 'google/gemini-flash-experimental': 'ambiguous', #Model not listed anywhere but naming pattern is correct
    'google/Gemini-Pro': 'google/gemini-pro', # or 'ambiguous', #Model not listed anywhere and naming patter is incorrect
    # 'google/gemini-pro-experimental': 'ambiguous', #Model not listed anywhere but naming pattern is correct
    # 'google/PaLM-2': 'google/test-bison-001', # Commenting out for now but according to docs this is the reference internally(https://ai.google.dev/palm_docs/palm)
    # 'google/PaLM-2-Chat': 'google/chat-bison-001', # Commenting out for now but according to docs this is the reference internally(https://ai.google.dev/palm_docs/palm)

    '': '',
    


    'mistralai/mistral-small-3.1-24b-instruct': 'mistralai/Mistral-Small-3.1-24b-instruct-2503',  # I cannot find the model mistral-small-3.1-24b-instruct in google. the closest I can find is mistral-small-3.1-24b-instruct-2503. 
    'mistralai/Mixtral-8x22B-Instruct-v0.1': 'mistralai/Mixtral-8x22B-Instruct-v0.1',
    # 'mistralai/mistral-small-latest', # I cannot determine the version or date code. Mistral-small has v.2, v.3.x.
    'mistralai/Mistral-Nemo-Instruct-2407': 'mistralai/Mistral-Nemo-Instruct-2407',
    'mistralai/ministral-8b-latest': 'mistralai/Ministral-8B-Instruct-2410',  # there seems to be only one ministral-8b. So i assume it is this one
    'mistralai/Mixtral-8x7B-Instruct-v0.1': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    # 'mistralai/mistral-large-latest', # I cannot determine the version or date code. But 
    # 'mistralai/Mistral-Large2', # I cannot determine the version or date code.
    'mistralai/Mistral-Small-24B-Instruct-2501': 'mistralai/Mistral-Small-24B-Instruct-2501',
    # 'mistralai/ministral-3b-latest', # I cannot find this model. There is only ministral-3b-instruct. 
    'mistralai/Mistral-7B-Instruct-v0.3': 'mistralai/Mistral-7B-Instruct-v0.3',
    'mistralai/pixtral-large-latest': 'mistralai/Pixtral-Large-Instruct-2411', # there is only one version pixal-large so far, which is 2411


    # 'gemini-2.0-flash-exp' : 'google/gemini-2.0-flash-exp',
}