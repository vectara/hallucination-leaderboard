# Hallucination Leaderboard

Public LLM leaderboard computed using Vectara's Hallucination Evaluation Model. This evaluates how often an LLM introduces hallucinations when summarizing a document. We plan to update this regularly as our model and the LLMs get updated over time.

Last updated on November 1st, 2023

|Model|Accuracy|Hallucination Rate|Answer Rate|Average Summary Length (Words)|
|----|----:|----:|----:|----:|
|GPT 4|97.0 %|3.0 %|100.0 %|81.1|
|GPT 4 Turbo|97.0 %|3.0 %|100.0 %|94.3|
|GPT 3.5 Turbo|96.5 %|3.5 %|99.6 %|84.1|
|Llama 2 70B|94.9 %|5.1 %|99.9 %|84.9|
|Llama 2 7B|94.4 %|5.6 %|99.6 %|119.9|
|Llama 2 13B|94.1 %|5.9 %|99.8 %|82.1|
|Cohere-Chat|92.5 %|7.5 %|98.0 %|74.4|
|Cohere |91.5 % |8.5 % |99.8 % |59.8 |
|Anthropic Claude 2 |91.5 % |8.5 % |99.3 % |87.5 |
|Mistral 7B |90.6 % |9.4 % |98.7 % |96.1 |
|Google Palm 2|87.9 % |12.1 % |92.4 % |36.2|
|Google Palm 2 Chat|72.8 % |27.2 % |88.8 % |221.1|

**Note** on GPT4 Turbo. While the above figures show it to be comparable to GPT4, this is due to us filtering out some documents that some of the models refuse to summarize. When comparing to GPT 4 on all summaries (both GPT4 models summarize all documents) the turbo model is around 0.3% worse than GPT4, but still better than GPT 3.5 Turbo.

<table>
  <tr>
    <td style="text-align: center; vertical-align: middle;">
      <img src="img/candle.png" width="50" height="50">
    </td>
    <td style="text-align: left; vertical-align: middle;">
      In loving memory of Simon Mark Hughes...
    </td>
  </tr>
</table>

## Model
You can find the model used to compute this leaderboard open sourced for commercial use on hugging face: https://huggingface.co/vectara/hallucination_evaluation_model along with instructions how to use the model.

## Data
See [leaderboard-summaries.csv](https://github.com/vectara/hallucination-leaderboard/blob/main/leaderboard_summaries.csv) for the generated summaries we used to evaluate the models with.

## Prior Research
Much prior work in this area has been done. For some of the top papers in this area (factual consistency in summarization) please see here:

* [SUMMAC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization](https://aclanthology.org/2022.tacl-1.10.pdf)
* [TRUE: Re-evaluating Factual Consistency Evaluation](https://arxiv.org/pdf/2204.04991.pdf)
* [TrueTeacher: Learning Factual Consistency Evaluation with Large Language Models](https://browse.arxiv.org/pdf/2305.11171v1.pdf)
* [ALIGNSCORE: Evaluating Factual Consistency with A Unified Alignment Function](https://arxiv.org/pdf/2305.16739.pdf)

For a very comprehensive list, please see here - https://github.com/EdinburghNLP/awesome-hallucination-detection. The methods described in the following section use protocols established in those papers, amongst many others.

## Methodology
For a detailed explanation of the work that went into this model please refer to our blog post on the release: [Cut the Bull…. Detecting Hallucinations in Large Language Models](https://vectara.com/cut-the-bull-detecting-hallucinations-in-large-language-models/).

To determine this leaderboard, we trained a model to detect hallucinations in LLM outputs, using various open source datasets from the factual consistency research into summarization models. Using a model that is competitive with the best state of the art models, we then fed 1000 short documents to each of the LLMs above via their public APIs and asked them to summarize each short document, using only the facts presented in the document. Of these 1000 documents, only 831 document were summarized by every model, the remaining documents were rejected by at least one model due to content restrictions. Using these 831 documents, we then computed the overall accuracy (no hallucinations) and hallucination rate (100 - accuracy) for each model. The rate at which each model refuses to respond to the prompt is detailed in the 'Answer Rate' column. None of the content sent to the models contained illicit or 'not safe for work' content but the present of trigger words was enough to trigger some of the content filters. The documents were taken primarily from the [CNN / Daily Mail Corpus](https://huggingface.co/datasets/cnn_dailymail/viewer/1.0.0/test). We used a **temperature of 0** when calling the LLMs.

We evaluate summarization accuracy instead of overall factual accuracy because it allows us to compare the model's response to the provided information. In other words, is the summary provided 'factualy consistent' with the source document. Determining hallucinations is impossible to do for any ad hoc question as it's not known precisely what data every LLM is trained on. In addition, having a model that can determine whether any response was hallucinated without a reference source requires solving the hallucination problem and presumably training a model as large or larger than these LLMs being evaluated. So we instead chose to look at the hallucination rate within the summarization task as this is a good analogue to determine how truthful the models are overall. In addition, LLMs are increasingly used in RAG (Retrieval Augmented Generation) pipelines to answer user queries, such as in Bing Chat and Google's chat integration. In a RAG system, the model is being deployed as a summarizer of the search results, so this leaderboard is also a good indicator for the accuracy of the models when used in RAG systems.

## Prompt Used
> You are a chat bot answering questions using data. You must stick to the answers provided solely by the text in the passage provided. You are asked the question 'Provide a concise summary of the following passage, covering the core pieces of information described.'  &lt;PASSAGE&gt;'

When calling the API, the &lt;PASSAGE&gt; token was then replaced with the source document (see the 'source' column in [leaderboard-summaries.csv](https://github.com/vectara/hallucination-leaderboard/blob/main/leaderboard_summaries.csv) ). 

## API Details
For GPT 3.5 we used the model name ```gpt-3.5-turbo``` in their API, ```gpt-4``` for GPT4, '''gpt-4-1106-preview''' for GPT 4 Turbo (as per open AI's docs) and we used the ```chat.completions.create``` endpoint from the python client library. For the 3 Llama models, we used the Anyscale hosted endpoints for each model. For the Cohere models, we used their ```/generate``` endpoint for *Cohere*, and ```/chat``` for *Cohere-Chat*. For Anthropic, we used the largest ```claude 2``` model they offer through their API. For the Miustral 7B model, we used the  [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) model, hosted via Hugging Face's API. For Google Palm 2 we used the ```text-bison-001``` model, and for Google Palm 2 Chat we used ```chat-bison-001```.

## Frequently Asked Questions
* **Qu.** Why are you are using a model to evaluate a model?
* **Answer** There are several reasons we chose to do this over a human evaluation. While we could have crowdsourced a large human scale evaluation, that's a one time thing, it does not scale in a way that allows us to constantly update the leaderboard as new APIs come online or models get updated. We work in a fast moving field so any such process would be out of data as soon as it published. Secondly, we wanted a repeatable process that we can share with others so they can use it themselves as one of many LLM quality scores they use when evaluating their own models. This would not be possible with a human annotation process, where the only things that could be shared are the process and the human labels. It's also worth pointing out that building a model for detecting hallucinations is **much easier** than building a generative model that never produces hallucinations. So long as the hallucination evaluation model is highly correlated with human raters' judgements, it can stand in as a good proxy for human judges. As we are specifically targetting summarization and not general 'closed book' question answering, the LLM we trained does not need to have memorized a large proportion of human knowledge, it just needs to have a solid grasp and understanding of the languages it support (currently just english, but we plan to expand language coverage over time).

* **Qu.** What if the LLM refuses to summarize the document or provides a one or two word answer?
* **Answer** We explicitly filter these out. See out [blog post](https://vectara.com/cut-the-bull-detecting-hallucinations-in-large-language-models/) for more information. You can see the 'Answer Rate' column on the leaderboard that indicates the percentage of documents summarized, and the 'Average Summary Length' column detailing the summary lengths, showing we didn't get very short answers for most documents.

* **Qu.** What version of model XYZ did you use?
* **Answer** Please see the API details section for specifics about the model versions used and how they were called, as well as the date the leaderboard was last updated. Please contact us (create an issue in the repo) if you need more clarity. 

* **Qu.** What about xAI's Grok LLM?
* **Answer** Currently (as of 11/14/2023) Grok is not publicly available and we do not have access. Those with early access I suspect are probably legally forbidden from doing this sort of evaluation on the model. Once the model is available via a public API we will look to add it, along with any other LLMs that are popular enough.

* **Qu.** Can't a model just score a 100% by providing either no answers or very short answers?
* **Answer** We explicitly filtered out such responses from every model, doing the final evaluation only on documents that all models provided a summary for. You can find out more technical details in our [blog post]([https://vectara.com/cut-the-bull-detecting-hallucinations-in-large-language-models/) on the topic. See also the 'Answer Rate' and 'Average Summary Length' columns in the table above.

* **Qu.** Wouldn't an extractive summarizer model that just copies and pastes from the original summary score 100% (0 hallucination) on this task?
* **Answer** Absolutely as by definition such a model would have no hallucinations and provide a faithful summary. We do not claim to be evaluating summarization quality, that is a separate and **orthogonal** task, and should be evaluated independently. We are **not** evaluating the quality of the summaries, only the **factual consistency** of them, as we point out in the [blog post](https://vectara.com/cut-the-bull-detecting-hallucinations-in-large-language-models/).

* **Qu.** This seems a very hackable metric, as you could just copy the original text as the summary
* **Answer.** That's true but we are not evaluating arbitrary models on this approach, e.g. like in a Kaggle competition. Any model that does so would perform poorly at any other task you care about. So I would consider this as quality metric that you'd run alongside whatever other evaluations you have for your model, such as summarization quality, question answering accuracy, etc. But we do not recommend using this as a standalone metric. None of the models chosen were trained on our model's output. That may happen in future but as we plan to update the model and also the source documents so this is a living leaderboard, that will be an unlikely occurance. That is however also an issue with any LLM benchmark. We should also point out this builds on a large body of work on factual consistency where many other academics invented and refined this protocol. See our references to the SummaC and True papers in this [blog post](https://vectara.com/cut-the-bull-detecting-hallucinations-in-large-language-models/), as well as this excellent compilation of resources - https://github.com/EdinburghNLP/awesome-hallucination-detection to read more.

* **Qu.** This does not definitively measure all the ways a model can hallucinate
* **Answer.** Agreed. We do not claim to have solved the problem of hallucination detection, and plan to expand and enhance this process further. But we do believe it is a move in the right direction, and provides a much needed starting point that everyone can build on top of.

* **Qu.** Some models could hallucinate only while summarizing. Couldn't you just provide it a list of well known facts and check how well it can recall them?
* **Answer.** That would be a poor test in my opinion. For one thing, unless you trained the model you don't know the data it was trained on, so you can't be sure the model is grounding its response in real data it has seen on or whether it is guessing. Additionally, there is no clear defintion of 'well known', and these types of data are typically easy for most models to accurately recall. Most hallucinations, in my admittedly subjective experience, come from the model fetching information that is very rarely known or discussed, or facts for which the model has seen conflicting information. Without knowing the source data the model was trained on, again it's impossible to validate these sort of hallucinations as you won't know which data fits this criterion. I also think its unlikely the model would only hallucinate while summarizing. We are asking the model to take information and transform it in a way that is still faithful to the source. This is analogous to a lot of generative tasks aside from summarization (e.g. write an email convering these points...), and if the model deviates from the prompt then that is a failure to follow instructions, indicating the model would struggle on other instruction following tasks also.

* **Qu.** This is a good start but far from definitive
* **Answer.** I totally agree. There's a lot more that needs to be done, and the problem is far from solved. But a 'good start' means that hopefully progress will start to be made in this area, and by open sourcing the model, we hope to involve the community into taking this to the next level.

## Coming Soon
* We will also be adding a leaderboard on citation accuracy. As a builder of RAG systems, we have noticed that LLMs tend to mis-attribute sources sometimes when answering a question based on supplied search results. We'd like to be able to measure this so we can help mitigate it within our platform.
* We will also look to expand the benchmark to cover other RAG tasks, such as multi-document summarization.
* We also plan to cover more languages than just english. Our current platform covers over 100 languages, and we want to develop hallucination detectors with comparable multi-lingual coverage.
