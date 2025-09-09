# This file defines how to substitute company/model names in our old HF summary dataset with official names on HuggingFace and or their own API, in whole or partially

# Notes

# 1. For cohere/Cohere and cohere/Cohere-Chat, see old README https://github.com/vectara/hallucination-leaderboard/tree/hhem-1.0-final?tab=readme-ov-file#cohere-models
# 2. For mistral models, this offical model listing has 

# Only models that are mentioned below will be migrated. Others will not, mostly because of missing date code. They will be re-evaluated later.
name_mapping = {
    'xai' : 'xai-org',

    'CohereForAI/c4ai-command-r-plus' : 'CohereLabs/c4ai-command-r-plus', # Open source on HF
    'cohere/Cohere': 'CohereLabs/command', # commerical 
    'cohere/command-r-plus-08-2024': 'CohereLabs/command-r-plus-08-2024',
    'cohere/command-a-03-2025': 'CohereLabs/command-a-03-2025', # Open source on HF
    'cohere/c4ai-aya-expanse-8b': 'CohereLabs/aya-expanse-8b', # Open source on HF
    'cohere/c4ai-aya-expanse-32b': 'CohereLabs/aya-expanse-32b', # Open source on HF
    'cohere/Cohere-Chat': 'CohereLabs/command-chat', # the same Cohere Command model but using the /chat endpoint
    'cohere/command-r-08-2024': 'CohereLabs/command-r-08-2024', # Open source on HF

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


    'gemini-2.0-flash-exp' : 'google/gemini-2.0-flash-exp',
}