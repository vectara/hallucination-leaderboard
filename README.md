# Hallucination Leaderboard

Public LLM leaderboard computed using Vectara's [Hughes Hallucination Evaluation Model](https://huggingface.co/vectara/hallucination_evaluation_model). This evaluates how often an LLM introduces hallucinations when summarizing a document. We plan to update this regularly as our model and the LLMs get updated over time.

Also, feel free to check out our [hallucination leaderboard](https://huggingface.co/spaces/vectara/leaderboard) on Hugging Face. 

The rankings in this leaderboard are computed using the HHEM-2.1 hallucination evaluation model. 
If you are interested in the previous leaderboard, which was based on HHEM-1.0, it is available [here](https://github.com/vectara/hallucination-leaderboard/tree/hhem-1.0-final)

<table style="border-collapse: collapse;">
  <tr>
    <td style="text-align: center; vertical-align: middle; border: none;">
      <img src="img/candle.png" width="50" height="50">
    </td>
    <td style="text-align: left; vertical-align: middle; border: none;">
      In loving memory of <a href="https://www.ivinsfuneralhome.com/obituaries/Simon-Mark-Hughes?obId=30000023">Simon Mark Hughes</a>...
    </td>
  </tr>
</table>

<!-- LEADERBOARD_START -->
Last updated on September 22, 2025

![Plot: hallucination rates of various LLMs](./img/top25_hallucination_rates_2025-09-22.png)

|Model|Hallucination Rate|Factual Consistency Rate|Answer Rate|Average Summary Length (Words)|
|----|----:|----:|----:|----:|
|antgroup/finix_s1_32b|0.6 %|99.4 %|99.8 %|86.9|
|google/gemini-2.0-flash-lite-preview|1.2 %|98.8 %|97.4 %|62.6|
|google/gemini-2.0-flash-001|1.2 %|98.8 %|100.0 %|65.8|
|openai/gpt-5-high|1.4 %|98.6 %|99.3 %|96.4|
|google/gemini-2.0-flash-exp|1.6 %|98.4 %|99.9 %|61.1|
|google/gemini-2.0-pro-exp|1.7 %|98.3 %|99.1 %|62.3|
|zai-org/GLM-4.5-AIR-FP8|1.9 %|98.1 %|99.1 %|74.4|
|google/gemini-2.5-flash|2.1 %|97.9 %|98.8 %|83.9|
|openai/gpt-oss-120b|2.4 %|97.6 %|100.0 %|82.3|
|openai/gpt-4-turbo|2.7 %|97.3 %|100.0 %|85.6|
|google/gemini-2.5-pro-preview|2.8 %|97.2 %|99.9 %|72.9|
|meta-llama/Meta-Llama-3.1-405B-Instruct|2.9 %|97.1 %|99.1 %|86.2|
|google/gemini-2.5-flash-lite|2.9 %|97.1 %|99.7 %|78.4|
|openai/gpt-5-mini|3.2 %|96.8 %|99.6 %|87.2|
|xai-org/grok-3|3.2 %|96.8 %|100.0 %|96.9|
|openai/gpt-4o-mini|3.3 %|96.7 %|100.0 %|76.4|
|amazon/nova-pro-v1:0|3.4 %|96.6 %|100.0 %|85.8|
|qwen/Qwen3-14B|3.6 %|96.4 %|100.0 %|81.6|
|openai/o1-preview|3.6 %|96.4 %|100.0 %|117.8|
|anthropic/claude-3-5-haiku|3.6 %|96.4 %|100.0 %|92.7|
|qwen/Qwen3-4B|3.7 %|96.3 %|100.0 %|86.9|
|zai-org/glm-4p5|3.7 %|96.3 %|99.1 %|81.1|
|amazon/nova-micro-v1:0|3.7 %|96.3 %|100.0 %|89.5|
|openai/gpt-oss-20b|3.7 %|96.3 %|99.4 %|90.0|
|qwen/qwen-max|3.8 %|96.2 %|8.5 %|90.0|
|amazon/nova-lite-v1:0|3.9 %|96.1 %|99.9 %|80.7|
|xai-org/grok-3-mini|4.1 %|95.9 %|100.0 %|89.9|
|openai/gpt-4.1|4.1 %|95.9 %|100.0 %|71.5|
|Intel/neural-chat-7b-v3-3|4.2 %|95.8 %|100.0 %|60.8|
|anthropic/claude-opus-4-1|4.2 %|95.8 %|98.2 %|107.3|
|google/gemini-2.5-pro-exp|4.2 %|95.8 %|32.8 %|73.8|
|openai/o1-mini|4.2 %|95.8 %|100.0 %|78.2|
|openai/o1-pro|4.4 %|95.6 %|100.0 %|79.9|
|openai/gpt-4.5-preview|4.4 %|95.6 %|100.0 %|76.5|
|google/gemma-1.1-2b-it|4.5 %|95.5 %|100.0 %|69.0|
|xai-org/grok-4|4.5 %|95.5 %|99.5 %|100.9|
|google/gemma-1.1-7b-it|4.6 %|95.4 %|100.0 %|67.4|
|google/gemini-2.5-flash-preview|4.6 %|95.4 %|13.4 %|71.5|
|openai/o1|4.7 %|95.3 %|99.9 %|72.5|
|openai/gpt-5-nano|4.7 %|95.3 %|99.9 %|72.7|
|qwen/Qwen2-72B-Instruct|4.8 %|95.2 %|100.0 %|100.3|
|openai/gpt-5-minimal|4.9 %|95.1 %|99.7 %|83.6|
|google/gemini-1.5-flash-002|4.9 %|95.1 %|99.9 %|59.5|
|anthropic/claude-3-5-sonnet|4.9 %|95.1 %|100.0 %|96.7|
|snowflake/snowflake-arctic-instruct|5.0 %|95.0 %|100.0 %|68.7|
|qwen/Qwen3-32B|5.0 %|95.0 %|100.0 %|81.8|
|openai/gpt-4.1-mini|5.0 %|95.0 %|100.0 %|79.2|
|qwen/Qwen3-8B|5.0 %|95.0 %|100.0 %|78.1|
|openai/gpt-4.1-nano|5.1 %|94.9 %|100.0 %|69.6|
|meta-llama/Llama-3.3-70B-Instruct|5.1 %|94.9 %|100.0 %|85.4|
|deepseek-ai/DeepSeek-V3|5.1 %|94.9 %|100.0 %|87.6|
|meta-llama/Meta-Llama-3.1-70B-Instruct|5.2 %|94.8 %|100.0 %|79.7|
|CohereLabs/aya-expanse-8b|5.2 %|94.8 %|99.9 %|85.5|
|microsoft/phi-4|5.3 %|94.7 %|100.0 %|100.2|
|microsoft/Phi-3.5-MoE-instruct|5.4 %|94.6 %|40.0 %|69.7|
|tngtech/DeepSeek-TNG-R1T2-Chimera|5.5 %|94.5 %|99.6 %|78.6|
|deepseek-ai/DeepSeek-V3.1|5.5 %|94.5 %|98.7 %|78.0|
|mistralai/Mixtral-8x22B-Instruct-v0.1|5.6 %|94.4 %|99.9 %|91.9|
|ai21labs/AI21-Jamba-Mini-1.5|5.6 %|94.4 %|43.0 %|74.4|
|01-ai/Yi-1.5-34B-Chat|5.6 %|94.4 %|100.0 %|83.6|
|mistralai/Pixtral-Large-Instruct|5.6 %|94.4 %|100.0 %|79.5|
|mistralai/Mistral-Small-24B-Instruct|5.6 %|94.4 %|100.0 %|75.9|
|meta-llama/Llama-3-70B-chat-hf|5.7 %|94.3 %|93.7 %|69.1|
|internlm/internlm3-8b-instruct|5.8 %|94.2 %|100.0 %|96.8|
|zai-org/glm-4-9b-chat|5.9 %|94.1 %|100.0 %|58.0|
|mistralai/mistral-small|6.0 %|94.0 %|99.9 %|99.1|
|openai/o4-mini|6.1 %|93.9 %|100.0 %|83.8|
|google/gemini-1.5-pro-002|6.2 %|93.8 %|99.9 %|61.9|
|mistralai/Mistral-Small-3.1-24b-instruct|6.5 %|93.5 %|100.0 %|74.2|
|qwen/Qwen2.5-7B-Instruct|6.5 %|93.5 %|100.0 %|71.2|
|meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo|6.6 %|93.4 %|100.0 %|76.7|
|anthropic/claude-3-opus|6.7 %|93.3 %|42.7 %|91.2|
|microsoft/WizardLM-2-8x22B|6.7 %|93.3 %|99.9 %|141.1|
|qwen/Qwen2-VL-7B-Instruct|6.7 %|93.3 %|100.0 %|73.8|
|meta-llama/Llama-3-8B-chat-hf|6.7 %|93.3 %|99.6 %|79.5|
|google/gemma-3-27b-it|6.7 %|93.3 %|80.6 %|64.9|
|qwen/Qwen2.5-32B-Instruct|6.7 %|93.3 %|100.0 %|67.9|
|microsoft/Phi-3.5-mini-instruct|6.8 %|93.2 %|100.0 %|74.7|
|google/gemma-2-2b-it|6.9 %|93.1 %|100.0 %|61.8|
|google/gemma-3-4b-it|6.9 %|93.1 %|100.0 %|63.4|
|CohereLabs/command-a|6.9 %|93.1 %|100.0 %|76.8|
|qwen/Qwen2.5-72B-Instruct|7.0 %|93.0 %|100.0 %|80.6|
|meta-llama/Llama-2-70b-chat-hf|7.1 %|92.9 %|100.0 %|83.7|
|qwen/Qwen2.5-14B-Instruct|7.1 %|92.9 %|100.0 %|75.0|
|mistralai/Ministral-8B-Instruct|7.2 %|92.8 %|100.0 %|64.3|
|microsoft/Orca-2-13b|7.2 %|92.8 %|100.0 %|65.0|
|microsoft/Phi-4-mini-instruct|7.4 %|92.6 %|100.0 %|69.3|
|01-ai/Yi-1.5-9B-Chat|7.5 %|92.5 %|100.0 %|86.0|
|CohereLabs/command-r|7.7 %|92.3 %|100.0 %|64.6|
|deepseek-ai/DeepSeek-R1-0528|7.7 %|92.3 %|100.0 %|138.9|
|ibm-granite/granite-3.0-8b-instruct|7.8 %|92.2 %|100.0 %|74.2|
|anthropic/claude-3-sonnet|8.4 %|91.6 %|100.0 %|106.7|
|microsoft/Phi-3-mini-4k-instruct|8.6 %|91.4 %|100.0 %|86.7|
|google/gemma-7b-it|8.6 %|91.4 %|100.0 %|109.7|
|meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo|8.7 %|91.3 %|99.2 %|68.8|
|ibm-granite/granite-3.2-8b-instruct|8.7 %|91.3 %|100.0 %|118.5|
|google/gemini-1.5-flash-001|8.8 %|91.2 %|99.9 %|63.4|
|openai/o3|8.8 %|91.2 %|100.0 %|81.5|
|meta-llama/Llama-2-7b-chat-hf|8.9 %|91.1 %|100.0 %|120.6|
|google/gemini-1.5-flash|9.0 %|91.0 %|72.2 %|62.9|
|microsoft/Phi-3-mini-128k-instruct|9.1 %|90.9 %|100.0 %|60.3|
|ibm-granite/granite-3.1-8b-instruct|9.1 %|90.9 %|100.0 %|107.4|
|google/gemini-1.5-pro-001|9.2 %|90.8 %|99.6 %|61.8|
|databricks/dbrx-instruct|9.2 %|90.8 %|100.0 %|84.9|
|CohereLabs/aya-expanse-32b|9.2 %|90.8 %|99.9 %|84.1|
|qwen/Qwen3-30B-A3B|9.4 %|90.6 %|99.9 %|69.6|
|CohereLabs/c4ai-command-r-plus|9.5 %|90.5 %|100.0 %|70.7|
|google/gemini-1.5-pro|9.8 %|90.2 %|8.3 %|84.3|
|meta-llama/Llama-3.2-3B-Instruct-Turbo|10.2 %|89.8 %|100.0 %|72.3|
|01-ai/Yi-1.5-6B-Chat|10.4 %|89.6 %|100.0 %|99.9|
|microsoft/phi-2|10.8 %|89.2 %|32.3 %|78.7|
|mistralai/Mistral-7B-Instruct-v0.3|10.9 %|89.1 %|100.0 %|98.0|
|google/text-bison-001|11.0 %|89.0 %|99.9 %|35.6|
|mistralai/Mistral-Nemo-Instruct|11.0 %|89.0 %|100.0 %|71.1|
|meta-llama/Llama-3.2-1B-Instruct|11.3 %|88.7 %|100.0 %|30.4|
|qwen/Qwen3-1.7B|11.4 %|88.6 %|100.0 %|68.6|
|mistralai/mistral-medium|11.4 %|88.6 %|98.3 %|99.2|
|meta-llama/Llama-2-13b-chat-hf|11.6 %|88.4 %|100.0 %|81.0|
|google/gemma-2-9b-it|11.8 %|88.2 %|100.0 %|71.7|
|mistralai/Mixtral-8x7B-Instruct-v0.1|12.2 %|87.8 %|99.1 %|89.3|
|qwen/Qwen2.5-3B-Instruct|12.7 %|87.3 %|100.0 %|70.3|
|ibm-granite/granite-3.2-2b-instruct|13.1 %|86.9 %|100.0 %|116.3|
|ibm-granite/granite-3.0-2b-instruct|13.5 %|86.5 %|100.0 %|80.8|
|CohereLabs/command-a-reasoning|13.7 %|86.3 %|63.5 %|97.9|
|ibm-granite/granite-3.1-2b-instruct|13.7 %|86.3 %|100.0 %|108.1|
|qwen/Qwen2-VL-2B-Instruct|14.1 %|85.9 %|100.0 %|81.7|
|anthropic/claude-2.0|14.3 %|85.7 %|100.0 %|87.0|
|deepseek-ai/DeepSeek-R1|14.7 %|85.3 %|100.0 %|77.0|
|qwen/Qwen3-235B-A22B|14.8 %|85.2 %|95.0 %|85.9|
|CohereLabs/command-chat|16.8 %|83.2 %|100.0 %|74.2|
|CohereLabs/command|17.6 %|82.4 %|100.0 %|59.0|
|google/gemma-3-1b-it|17.6 %|82.4 %|100.0 %|57.6|
|qwen/Qwen3-0.6B|17.9 %|82.1 %|99.9 %|65.6|
|qwen/QwQ-32B-Preview|19.5 %|80.5 %|100.0 %|139.5|
|google/flan-t5-large|21.9 %|78.1 %|99.4 %|21.0|
|qwen/Qwen2.5-1.5B-Instruct|25.7 %|74.3 %|100.0 %|71.4|
|tiiuae/falcon-7b-instruct|28.4 %|71.6 %|12.5 %|71.7|
|qwen/Qwen2.5-0.5B-Instruct|42.6 %|57.4 %|99.9 %|73.5|
|apple/OpenELM-3B-Instruct|45.6 %|54.4 %|99.8 %|27.0|
|google/chat-bison-001|59.6 %|40.4 %|100.0 %|221.0|


<!-- LEADERBOARD_END -->

## Model
This leaderboard uses HHEM-2.1, Vectara's commercial hallucination evaluation model, to compute the LLM rankings. You can find an open-source variant of that model, HHEM-2.1-Open on [Hugging Face](https://huggingface.co/vectara/hallucination_evaluation_model) and [Kaggle](https://www.kaggle.com/models/vectara/hallucination_evaluation_model).

## Data
See [this dataset](https://huggingface.co/datasets/vectara/leaderboard_results) for the generated summaries we used for evaluating the models.

## Prior Research
Much prior work in this area has been done. For some of the top papers in this area (factual consistency in summarization) please see here:

* [SUMMAC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization](https://aclanthology.org/2022.tacl-1.10.pdf)
* [TRUE: Re-evaluating Factual Consistency Evaluation](https://arxiv.org/pdf/2204.04991.pdf)
* [TrueTeacher: Learning Factual Consistency Evaluation with Large Language Models](https://browse.arxiv.org/pdf/2305.11171v1.pdf)
* [ALIGNSCORE: Evaluating Factual Consistency with A Unified Alignment Function](https://arxiv.org/pdf/2305.16739.pdf)
* [MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents](https://arxiv.org/pdf/2404.10774)

For a very comprehensive list, please see here - https://github.com/EdinburghNLP/awesome-hallucination-detection. The methods described in the following section use protocols established in those papers, amongst many others.

## Methodology
For a detailed explanation of the work that went into this model please refer to our blog post on the release: [Cut the Bull…. Detecting Hallucinations in Large Language Models](https://vectara.com/blog/cut-the-bull-detecting-hallucinations-in-large-language-models/).

To determine this leaderboard, we trained a model to detect hallucinations in LLM outputs, using various open source datasets from the factual consistency research into summarization models. Using a model that is competitive with the best state of the art models, we then fed 1000 short documents to each of the LLMs above via their public APIs and asked them to summarize each short document, using only the facts presented in the document. Of these 1000 documents, only 831 document were summarized by every model, the remaining documents were rejected by at least one model due to content restrictions. Using these 831 documents, we then computed the overall factual consistency rate (no hallucinations) and hallucination rate (100 - accuracy) for each model. The rate at which each model refuses to respond to the prompt is detailed in the 'Answer Rate' column. None of the content sent to the models contained illicit or 'not safe for work' content but the present of trigger words was enough to trigger some of the content filters. The documents were taken primarily from the [CNN / Daily Mail Corpus](https://huggingface.co/datasets/cnn_dailymail/viewer/1.0.0/test). We used a **temperature of 0** when calling the LLMs.

We evaluate summarization factual consistency rate instead of overall factual accuracy because it allows us to compare the model's response to the provided information. In other words, is the summary provided 'factually consistent' with the source document. Determining hallucinations is impossible to do for any ad hoc question as it's not known precisely what data every LLM is trained on. In addition, having a model that can determine whether any response was hallucinated without a reference source requires solving the hallucination problem and presumably training a model as large or larger than these LLMs being evaluated. So we instead chose to look at the hallucination rate within the summarization task as this is a good analogue to determine how truthful the models are overall. In addition, LLMs are increasingly used in RAG (Retrieval Augmented Generation) pipelines to answer user queries, such as in Bing Chat and Google's chat integration. In a RAG system, the model is being deployed as a summarizer of the search results, so this leaderboard is also a good indicator for the accuracy of the models when used in RAG systems.

## Prompt Used
> You are a chat bot answering questions using data. You must stick to the answers provided solely by the text in the passage provided. You are asked the question 'Provide a concise summary of the following passage, covering the core pieces of information described.'  &lt;PASSAGE&gt;'

When calling the API, the &lt;PASSAGE&gt; token was then replaced with the source document (see the 'source' column in [this dataset](https://huggingface.co/datasets/vectara/leaderboard_results) ). 

## API Integration Details
Below is a detailed overview of the models integrated and their specific endpoints:

### OpenAI Models
- **GPT-3.5**: Accessed using the model name `gpt-3.5-turbo` through OpenAI's Python client library, specifically via the `chat.completions.create` endpoint.
- **GPT-4, GPT-4 Turbo, GPT-4o, GPT-4o-mini**: Integrated with the model identifier `gpt-4`, `gpt-4-turbo-2024-04-09`, `gpt-4o`, `gpt-4o-mini`.
- **GPT-4.1, GPT-4.1-mini, GPT-4.1-nano**: Accessed using the model name `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`
- **GPT-4.5-preview**: Accessed using the model name `gpt-4.5-preview-2025-02-27`.
- **o1-mini, o1-preview, o1, o1-pro**: Accessed using the model name `o1-mini`, `o1-preview`, `o1`, `o1-pro`.
- **o3-mini-high, o3**: Accessed using the model name `o3-mini` (with parameter `reasoning_effort="high"`), `o3`.
- **o4-mini**: Accessed using the model name `o4-mini`.

### Llama Models
- **Llama 2 7B, 13B, and 70B**: These models of varying sizes are accessed through Anyscale hosted endpoints using model `meta-llama/Llama-2-xxb-chat-hf`, where `xxb` can be `7b`, `13b`, and `70b`, tailored to each model's capacity.
- **Llama 3 8B and 70B**: These models are accessed via Together AI `chat` endpoint and using the model `meta-llama/Llama-3-xxB-chat-hf`,  where `xxB` can be `8B` and `70B`. 
- **Llama 3.1 8B, 70B and 405B**: The models [meta-llama/Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct) and [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) are accessed via Hugging Face's checkpoint. The model `Meta-Llama-3.1-405B-Instruct` is accessed via Replicate's API using the model `meta/meta-llama-3.1-405b-instruct`.
- **Llama 3.2 1B and 3B**: The model [meta-llama/Meta-Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.2-1B-Instruct) is accessed via Hugging Face's checkpoint. The model `Meta-Llama-3.2-3B-Instruct` is accessed via Together AI `chat` endpoint using model `meta-llama/Llama-3.2-3B-Instruct-Turbo`.
- **Llama 3.2 Vision 11B and 90B**:The models `Llama-3.2-11B-Vision-Instruct` and `Llama-3.2-90B-Vision-Instruct` are accessed via Together AI `chat` endpoint using model `meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo` and `meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo`.
- **Llama 3.3 70B**: The model [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) is accessed via Hugging Face's checkpoint.
- **Llama 4 Maverick and Scout**: The llama 4 [Maverick](https://openrouter.ai/meta-llama/llama-4-maverick) and [Scout](https://openrouter.ai/meta-llama/llama-4-scout) are accessed via OpenRouter API.

### Cohere Models
- **Cohere Command R**: Employed using the model `command-r-08-2024` and the `/chat` endpoint.
- **Cohere Command R Plus**: Employed using the model `command-r-plus-08-2024` and the `/chat` endpoint.
- **Aya Expanse 8B, 32B**: Accessed using models `c4ai-aya-expanse-8b` and `c4ai-aya-expanse-32b`.
- **Cohere Command A**: Employed using the model `command-a-03-2025` and the `/chat` endpoint.
For more information about Cohere's models, refer to their [website](https://docs.cohere.com/docs/models).

### Anthropic Model
- **Claude 2**: Invoked the model using `claude-2.0` for the API call.
- **Claude 3 Opus**: Invoked the model using `claude-3-opus-20240229` for the API call.
- **Claude 3 Sonnet**: Invoked the model using `claude-3-sonnet-20240229` for the API call.
- **Claude 3.5 Sonnet**: Invoked the model using `claude-3-5-sonnet-20241022` for the API call.
- **Claude 3.5 Haiku**: Invoked the model using `claude-3-5-haiku-20241022` for the API call.
- **Claude 3.7 Sonnet/Sonnet-Thinking**: Invoked the model using `claude-3-7-sonnet-20250219` for the API call. 
Details on each model can be found on their [website](https://docs.anthropic.com/claude/docs/models-overview).

### Mistral AI Models
- **Mixtral 8x7B**: The [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) model is accessed via Hugging Face's API.
- **Mixtral 8x22B**: Accessed via Together AI's API using the model `mistralai/Mixtral-8x22B-Instruct-v0.1` and the `chat` endpoint.
- **Mistral Pixtral Large**: Accessed via Mistral AI's API using the model `pixtral-large-latest`.
- **Mistral Large2**: Accessed via Mistral AI's API using the model `mistral-large-latest`.
- **Mistral Small2**: Accessed via Mistral AI's API using the model `mistral-small-latest`.
- **Mistral Small3**: The [Mistral Small3](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) model is accessed by being loaded from Hugging Face's checkpoint.
- **Mistral-Small-3.1-24B-Instruct**: The [Mistral Small 3.1](mistralai/Mistral-Small-3.1-24B-Instruct-2503) model is accessed by being loaded from Hugging Face's checkpoint.
- **Mistral-7B-Instruct-v0.3**: The [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) model is accessed by being loaded from Hugging Face's checkpoint.
- **Mistral-Nemo-Instruct** The [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) model is accessed via Hugging Face's checkpoint.
- **Mistral Ministral 3B**: Accessed via Mistral AI's API using the model `ministral-3b-latest`.
- **Mistral Ministral 8B**: Accessed via Mistral AI's API using the model `ministral-8b-latest`.

### Google Closed-Source Models via Vertex AI
- **Google Palm 2**: Implemented using the `text-bison-001` model, respectively.
- **Gemini Pro**: Google's `gemini-pro` model is incorporated for enhanced language processing, accessible on Vertex AI.
- **Gemini 1.5 Pro, Flash, Pro 002, Flase 002**: Accessed using model `gemini-1.5-pro-001`, `gemini-1.5-flash-001`, `gemini-1.5-pro-002`, `gemini-1.5-flash-002` on Vertex AI. 
- **Gemini 2.0 Flash Exp, Flash, Flash Lite, Flash Thinking Exp, Pro Exp**: Accessed using model `gemini-2.0-flash-exp`, `gemini-2.0-flash-001`, `gemini-2.0-flash-lite-preview-02-05`, `gemini-2.0-flash-thinking-exp`, `gemini-2.0-pro-exp-02-05` on Vertex AI. 
- **Gemini 2.5 Pro Exp, Flash Preview**: Accessed using model `gemini-2.5-pro-exp-03-25`, `gemini-2.5-flash-preview-04-17` on Vertex AI. 

### Google Open-Source Models on Hugging Face
- **Google flan-t5-large**: The [flan-t5-large](https://huggingface.co/google/flan-t5-large) model is accessed via Hugging Face's API.
- **Google gemma-7b-it**: The [gemma-7b-it](https://huggingface.co/google/gemma-7b-it) model is accessed via Hugging Face's API. 
- **Google gemma-1.1-7b-it** : The [gemma-1.1-7b-it](https://huggingface.co/google/gemma-1.1-7b-it) model is accessed by being loaded from Hugging Face's checkpoint. 
- **Google gemma-1.1-2b-it** : The [gemma-1.1-2b-it](https://huggingface.co/google/gemma-1.1-2b-it) model is accessed via being loaded from Hugging Face's checkpoint.
- **Google gemma-2-9b-it** : The [gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) model is accessed via being loaded from Hugging Face's checkpoint.
- **Google gemma-2-2b-it** : The [gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) model is accessed via being loaded from Hugging Face's checkpoint.
- **Google gemma-3-1b/4b/12b/27b-it** : The [gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it), [gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it), [gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it), [gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) models are accessed via being loaded from Hugging Face's checkpoint.

For an in-depth understanding of each model's version and lifecycle, especially those offered by Google, please refer to [Model Versions and Lifecycles](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/model-versioning) on Vertex AI.

### Amazon Models
- **Amazon Titan Express**: The [model](https://aws.amazon.com/bedrock/titan/) is accessed on Amazon Bedrock with model identifier of `amazon.titan-text-express-v1`.
- **Amazon Nova V1 Pro, Lite, Micro**: The Amazon Nova V1 [Pro](https://openrouter.ai/amazon/nova-pro-v1), [Lite](https://openrouter.ai/amazon/nova-lite-v1), [Micro](https://openrouter.ai/amazon/nova-micro-v1) are accessed via OpenRouter API with parameter `"temperature": 0`.

### Microsoft Models
- **Microsoft Phi-2**: The [phi-2](https://huggingface.co/microsoft/phi-2) model is accessed via Hugging Face's API.
- **Microsoft Orca-2-13b**: The [Orca-2-13b](https://huggingface.co/microsoft/Orca-2-13b) model is accessed via Hugging Face's API.
- **Microsoft WizardLM-2-8x22B**: Accessed via Together AI's API using the model `microsoft/WizardLM-2-8x22B` and the `chat` endpoint.  
- **Microsoft Phi-3-mini-4k**: The [phi-3-mini-4k](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) model is accessed via Hugging Face's checkpoint.
- **Microsoft Phi-3-mini-128k**: The [phi-3-mini-128k](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) model is accessed via Hugging Face's checkpoint.
- **Microsoft Phi-3.5-mini-instruct**: The [phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) model is accessed via Hugging Face's checkpoint.
- **Microsoft Phi-3.5-MoE-instruct**: The [phi-3.5-MoE-instruct](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct) model is accessed via Hugging Face's checkpoint.
- **Microsoft Phi-4/Phi-4-Mini**: The [phi-4](https://huggingface.co/microsoft/phi-4) and [phi-4-mini](https://huggingface.co/microsoft/Phi-4-mini-instruct) models are accessed via Hugging Face's checkpoint.

### TII Models on Hugging Face
- **tiiuae/falcon-7b-instruct**: The [falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) model is accessed via Hugging Face's API.

### Intel Model on Hugging Face
- **Intel/neural-chat-7b-v3-3**: The [Intel/neural-chat-7b-v3-3](https://huggingface.co/Intel/neural-chat-7b-v3-3) model is accessed via Hugging Face's checkpoint. 

### Databricks Model
- **Databricks/dbrx-instruct**: Accessed via Together AI's API using the model `databricks/dbrx-instruct` and the `chat` endpoint. 

### Snowflake Model
- **Snowflake/snowflake-arctic-instruct**: Accessed via Replicate's API using the model `snowflake/snowflake-arctic-instruct`.

### Apple Model
- **Apple/OpenELM-3B-Instruct**: The [OpenELM-3B-Instruct](https://huggingface.co/apple/OpenELM-3B-Instruct) model is accessed via being loaded from Hugging Face's checkpoint. The prompt for this model is the original prompt plus ''\n\nA concise summary is as follows:''

### 01-AI Models
- **01-AI/Yi-1.5-Chat 6B, 9B, 34B**: The models [01-ai/Yi-1.5-6B-Chat](https://huggingface.co/01-ai/Yi-1.5-6B-Chat), [01-ai/Yi-1.5-9B-Chat](https://huggingface.co/01-ai/Yi-1.5-9B-Chat), and [01-ai/Yi-1.5-34B-Chat](https://huggingface.co/01-ai/Yi-1.5-34B-Chat) are accessed via Hugging Face's checkpoint.

### Zhipu AI Model
- **Zhipu-AI/GLM-4-9B-Chat**: The [GLM-4-9B-Chat](https://huggingface.co/THUDM/glm-4-9b-chat) is accessed via Hugging Face's checkpoint.

### Qwen Models
- **Qwen2-72B-Instruct, -VL-Instruct 2B, 7B**: Qwen2-72B-Instruct is accessed via Together AI `chat` endpoint with model name `Qwen/Qwen2-72B-Instruct`. The models [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) and [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) are accessed via Hugging Face's checkpoints.
- **Qwen2.5-Instruct 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B, Max**: The models [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct), [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct), [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct), and [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) are accessed via Hugging Face's checkpoints. The model [Qwen2.5-max](https://openrouter.ai/qwen/qwen-max) are accessed via OpenRouter API with parameter `"temperature": 0`.
- **Qwen-QwQ-32B-Preview**: The model [Qwen/QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview) is accessed via Hugging Face's checkpoint.
- **Qwen3-0.6B, 1.7B, 4B, 8B, 14B, 32B**: The models [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) are accessed via Hugging Face's checkpoints with `enable_thinking=False`.

### AI21 Model
- **AI21-Jamba-1.5-Mini**: The [Jamba-1.5-Mini](https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini) model is accessed via Hugging Face's checkpoint.
- **AI21 Jamba-1.6-Large / Mini**: The [Jamba-1.6-Large](https://openrouter.ai/ai21/jamba-1.6-large) and [Jamba-1.6-Mini](https://openrouter.ai/ai21/jamba-1.6-mini) are accessed via OpenRouter API with endpoint `ai21/jamba-1.6-large` and `ai21/jamba-1.6-mini`.

### DeepSeek Model
- **DeepSeek V2.5**: Accessed via DeepSeek's API using `deepseek-chat` model and the `chat` endpoint.
- **DeepSeek V3**: Accessed via DeepSeek's API using `deepseek-chat` model and the `chat` endpoint.
- **DeepSeek V3-0324**: Accessed via DeepSeek's API using `deepseek-chat` model and the `chat` endpoint.
- **DeepSeek R1**: Accessed via DeepSeek's API using `deepseek-reasoner` model and the `reasoner` endpoint.

### IBM Models
- **Granite-3.0-Instruct 2B, 8B**: The models [ibm-granite/granite-3.0-8b-instruct](https://huggingface.co/ibm-granite/granite-3.0-8b-instruct) and [ibm-granite/granite-3.0-2b-instruct](https://huggingface.co/ibm-granite/granite-3.0-2b-instruct) are accessed via Hugging Face's checkpoints.
- **Granite-3.1-Instruct 2B, 8B**: The models [ibm-granite/granite-3.1-8b-instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct) and [ibm-granite/granite-3.1-2b-instruct](https://huggingface.co/ibm-granite/granite-3.1-2b-instruct) are accessed via Hugging Face's checkpoints.
- **Granite-3.2-Instruct 2B, 8B**: The models [ibm-granite/granite-3.2-8b-instruct](https://huggingface.co/ibm-granite/granite-3.2-8b-instruct) and [ibm-granite/granite-3.2-2b-instruct](https://huggingface.co/ibm-granite/granite-3.2-2b-instruct) are accessed via Hugging Face's checkpoints. Summaries are generated with `thinking=True`.

### XAI Model
- **Grok**: Accessed via xAI's API using the model `grok-beta` and the `chat/completions` endpoint.
- **Grok-2, 2-Vision**: The [grok-2](https://openrouter.ai/x-ai/grok-2-1212) and [grok-2-vision] are accessed via OpenRouter API with endpoint `x-ai/grok-2-1212` and `x-ai/grok-2-vision-1212`.
- **Grok-3-beta, 3-mini-beta**: Accessed via xAI's API using the model `grok-3-beta` and `grok-3-mini-beta`.

### AllenAI Models
- **OLMo-2 7B, 13B**: The models [allenai/OLMo-2-1124-7B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct) and [allenai/OLMo-2-1124-13B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct) are accessed via Hugging Face's checkpoints.

### InternLM Models
- **InternLM3-8B-Instruct**: The model [internlm/internlm3-8b-instruct](https://huggingface.co/internlm/internlm3-8b-instruct) are accessed via HuggingFace's checkpoint.


## Frequently Asked Questions
* **Qu.** Why are you are using a model to evaluate a model?
* **Answer** There are several reasons we chose to do this over a human evaluation. While we could have crowdsourced a large human scale evaluation, that's a one time thing, it does not scale in a way that allows us to constantly update the leaderboard as new APIs come online or models get updated. We work in a fast moving field so any such process would be out of data as soon as it published. Secondly, we wanted a repeatable process that we can share with others so they can use it themselves as one of many LLM quality scores they use when evaluating their own models. This would not be possible with a human annotation process, where the only things that could be shared are the process and the human labels. It's also worth pointing out that building a model for detecting hallucinations is **much easier** than building a generative model that never produces hallucinations. So long as the hallucination evaluation model is highly correlated with human raters' judgements, it can stand in as a good proxy for human judges. As we are specifically targetting summarization and not general 'closed book' question answering, the LLM we trained does not need to have memorized a large proportion of human knowledge, it just needs to have a solid grasp and understanding of the languages it support (currently just english, but we plan to expand language coverage over time).

* **Qu.** What if the LLM refuses to summarize the document or provides a one or two word answer?
* **Answer** We explicitly filter these out. See our [blog post](https://vectara.com/blog/cut-the-bull-detecting-hallucinations-in-large-language-models/) for more information. You can see the 'Answer Rate' column on the leaderboard that indicates the percentage of documents summarized, and the 'Average Summary Length' column detailing the summary lengths, showing we didn't get very short answers for most documents.

* **Qu.** What version of model XYZ did you use?
* **Answer** Please see the API details section for specifics about the model versions used and how they were called, as well as the date the leaderboard was last updated. Please contact us (create an issue in the repo) if you need more clarity. 

* **Qu.** Can't a model just score a 100% by providing either no answers or very short answers?
* **Answer** We explicitly filtered out such responses from every model, doing the final evaluation only on documents that all models provided a summary for. You can find out more technical details in our [blog post]([https://vectara.com/blog/cut-the-bull-detecting-hallucinations-in-large-language-models/) on the topic. See also the 'Answer Rate' and 'Average Summary Length' columns in the table above.

* **Qu.** Wouldn't an extractive summarizer model that just copies and pastes from the original summary score 100% (0 hallucination) on this task?
* **Answer** Absolutely as by definition such a model would have no hallucinations and provide a faithful summary. We do not claim to be evaluating summarization quality, that is a separate and **orthogonal** task, and should be evaluated independently. We are **not** evaluating the quality of the summaries, only the **factual consistency** of them, as we point out in the [blog post](https://vectara.com/cut-the-bull-detecting-hallucinations-in-large-language-models/).

* **Qu.** This seems a very hackable metric, as you could just copy the original text as the summary
* **Answer** That's true but we are not evaluating arbitrary models on this approach, e.g. like in a Kaggle competition. Any model that does so would perform poorly at any other task you care about. So I would consider this as quality metric that you'd run alongside whatever other evaluations you have for your model, such as summarization quality, question answering accuracy, etc. But we do not recommend using this as a standalone metric. None of the models chosen were trained on our model's output. That may happen in future but as we plan to update the model and also the source documents so this is a living leaderboard, that will be an unlikely occurrence. That is however also an issue with any LLM benchmark. We should also point out this builds on a large body of work on factual consistency where many other academics invented and refined this protocol. See our references to the SummaC and True papers in this [blog post](https://vectara.com/blog/cut-the-bull-detecting-hallucinations-in-large-language-models/), as well as this excellent compilation of resources - https://github.com/EdinburghNLP/awesome-hallucination-detection to read more.

* **Qu.** This does not definitively measure all the ways a model can hallucinate
* **Answer** Agreed. We do not claim to have solved the problem of hallucination detection, and plan to expand and enhance this process further. But we do believe it is a move in the right direction, and provides a much needed starting point that everyone can build on top of.

* **Qu.** Some models could hallucinate only while summarizing. Couldn't you just provide it a list of well known facts and check how well it can recall them?
* **Answer** That would be a poor test in my opinion. For one thing, unless you trained the model you don't know the data it was trained on, so you can't be sure the model is grounding its response in real data it has seen on or whether it is guessing. Additionally, there is no clear definition of 'well known', and these types of data are typically easy for most models to accurately recall. Most hallucinations, in my admittedly subjective experience, come from the model fetching information that is very rarely known or discussed, or facts for which the model has seen conflicting information. Without knowing the source data the model was trained on, again it's impossible to validate these sort of hallucinations as you won't know which data fits this criterion. I also think its unlikely the model would only hallucinate while summarizing. We are asking the model to take information and transform it in a way that is still faithful to the source. This is analogous to a lot of generative tasks aside from summarization (e.g. write an email covering these points...), and if the model deviates from the prompt then that is a failure to follow instructions, indicating the model would struggle on other instruction following tasks also.

* **Qu.** This is a good start but far from definitive
* **Answer** I totally agree. There's a lot more that needs to be done, and the problem is far from solved. But a 'good start' means that hopefully progress will start to be made in this area, and by open sourcing the model, we hope to involve the community into taking this to the next level.

## Coming Soon
* We will also be adding a leaderboard on citation accuracy. As a builder of RAG systems, we have noticed that LLMs tend to mis-attribute sources sometimes when answering a question based on supplied search results. We'd like to be able to measure this so we can help mitigate it within our platform.
* We will also look to expand the benchmark to cover other RAG tasks, such as multi-document summarization.
* We also plan to cover more languages than just english. Our current platform covers over 100 languages, and we want to develop hallucination detectors with comparable multi-lingual coverage.

<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5f53f560-5ba6-4e73-917b-c7049e9aea2c" />
