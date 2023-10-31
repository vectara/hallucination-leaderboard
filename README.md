# Hallucination Leaderboard

Public LLM leaderboard computed using our Hallucination Evaluation Model. We plan to update this monthly, or sooner when appropriate.

Last updated on November 1st, 2023

|Model|Accuracy|Hallucination Rate|Average Summary Length (Words)|Answer Rate|
|----|----:|----:|----:|----:|
|GPT 4|97.0 %|3.0 %|81.1|100.0 %|
|GPT 3.5|96.5 %|3.5 %|84.1|99.6 %|
|Llama 2 70B|94.9 %|5.1 %|84.9|99.9 %|
|Llama 2 7B|94.4 %|5.6 %|119.9|99.6 %|
|Llama 2 13B|94.1 %|5.9 %|82.1|99.8 %|
|Cohere-Chat|92.5 %|7.5 %|74.4|98.0 %|
|Cohere|91.5 %|8.5 %|59.8|99.8 %|
|Anthropic Claude 2|91.5 %|8.5 %|87.5|99.3 %|
|Mistral 7B|90.6 %|9.4 %|96.1|98.7 %|
|Google Palm (*text-bison--001*)|87.9 %|12.1 %|36.2|92.4 %|
|Google Palm-Chat (*chat-bison-001*)|72.8 %|27.2 %|221.1|88.8 %|

## API Details
For GPT 3.5 we used the model name ```gpt-3.5-turbo``` in their API, and ```gpt-4``` for GPT4. For the 3 Llama models, we used the Anyscale hosted endpoints for each model. For the Cohere models, we used the ```/generate``` endpoint for *Cohere*, and ```/chat``` for *Cohere-Chat*. For Anthropic, we used the largest ```claude 2``` model they offer through their API. For the Miustral 7B model, we used the  [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) model, hosted via Hugging Face's API. For Google Palm we used the ```text-bison-001``` model, and for Google Palm Chat we used ```chat-bison-001```.

**TODO**
* Link to Github model. Replicate my HF instructions on how to use model
* Links to the 2 blog posts.

