# TODO: Create children for all models on the HHEM LB
# TODO: Above is being done as needed
'''
Model List
    - Anthropic 4
    - Google Gemini-2.0-Flash-001
    - Google Gemini-2.0-Pro-Exp
    - OpenAI o3-mini-high
    - Vectara Mockingbird-2-Echo
    - Google Gemini-2.5-Pro-Exp-0325
    - Google Gemini-2.0-Flash-Lite-Preview
    - OpenAI GPT-4.5-Preview
    - Zhipu AI GLM-4-9B-Chat	
    - Google Gemini-2.0-Flash-Exp
    - Google Gemini-2.5-Flash-Preview
    - OpenAI-o1-mini
    - OpenAI GPT-4o
    - Amazon Nova-Micro-V1
    - OpenAI GPT-4o-mini
    - OpenAI GPT-4-Turbo
    - Google Gemini-2.0-Flash-Thinking-Exp
    - Amazon Nova-Lite-V1
    - OpenAI GPT-4
    - Amazon Nova-Pro-V1
    - OpenAI GPT-3.5-Turbo
    - XAI Grok-2
    - OpenAI GPT-4.1-nano
    - OpenAI GPT-4.1 DONE
    - XAI Grok-3-Beta
    - OpenAI GPT-4.1-mini
    - Qwen3-14B
    - AI21 Jamba-1.6-Large
    - OpenAI o1-Pro
    - OpenAI o1
    - DeepSeek-V2.5
    - Microsoft Orca-2-13b
    - Microsoft Phi-3.5-MoE-instruct
    - Intel Neural-Chat-7B-v3-3
    - Qwen3-4B
    - Google Gemma-3-12B-Instruct
    - Qwen2.5-7B-Instruct
    - Qwen3-32B
    - AI21 Jamba-1.5-Mini
    - XAI Grok-2-Vision
    - Qwen2.5-Max
    - Google Gemma-3-27B-Instruct
    - Qwen2.5-32B-Instruct
    - Snowflake-Arctic-Instruct
    - Qwen3-8B
    - Microsoft Phi-3-mini-128k-instruct
    - Mistral Small3
    - XAI Grok-3-Mini-Beta
    - OpenAI o1-preview
    - Google Gemini-1.5-Flash-002
    - Microsoft Phi-4-mini-instruct
    - Google Gemma-3-4B-Instruct
    - Qwen3-0.6B
    - 01-AI Yi-1.5-34B-Chat
    - Llama-3.1-405B-Instruct
    - DeepSeek-V3
    - Microsoft Phi-3-mini-4k-instruct
    - Mistral-Large2
    - Llama-3.3-70B-Instruct
    - Qwen2-VL-7B-Instruct
    - Qwen2.5-14B-Instruct
    - Qwen2.5-72B-Instruct
    - Llama-3.2-90B-Vision-Instruct
    - Qwen3-1.7B
    - Claude-3.7-Sonnet
    - Claude-3.7-Sonnet-Think
    - Cohere Command-A
    - OpenAI o4-mini
    - AI21 Jamba-1.6-Mini
    - Meta Llama-4-Maverick
    - XAI Grok
    - Anthropic Claude-3-5-sonnet
    - Meta Llama-4-Scout
    - Qwen2-72B-Instruct
    - Microsoft Phi-4
    - Mixtral-8x22B-Instruct-v0.1
    - Anthropic Claude-3-5-haiku
    - 01-AI Yi-1.5-9B-Chat
    - Cohere Command-R
    - Llama-3.1-70B-Instruct
    - Google Gemma-3-1B-Instruct
    - Llama-3.1-8B-Instruct
    - Cohere Command-R-Plus
    - Mistral-Small-3.1-24B-Instruct
    - Llama-3.2-11B-Vision-Instruct
    - Llama-2-70B-Chat-hf
    - IBM Granite-3.0-8B-Instruct
    - Google Gemini-1.5-Pro-002
    - Google Gemini-1.5-Flash
    - Mistral-Pixtral
    - Microsoft phi-2
    - OpenAI o3
    - Google Gemma-2-2B-it
    - Qwen2.5-3B-Instruct
    - Llama-3-8B-Chat-hf
    - Mistral-Ministral-8B
    - Google Gemini-Pro
    - 01-AI Yi-1.5-6B-Chat
    - Llama-3.2-3B-Instruct
    - DeepSeek-V3-0324
    - Mistral-Ministral-3B
    - databricks dbrx-instruct
    - Qwen2-VL-2B-Instruct
    - Cohere Aya Expanse 32B
    - IBM Granite-3.1-8B-Instruct
    - Mistral-Small2
    - IBM Granite-3.2-8B-Instruct
    - IBM Granite-3.0-2B-Instruct
    - Mistral-7B-Instruct-v0.3
    - Google Gemini-1.5-Pro
    - Anthropic Claude-3-opus
    - Google Gemma-2-9B-it
    - Llama-2-13B-Chat-hf
    - AllenAI-OLMo-2-13B-Instruct
    - AllenAI-OLMo-2-7B-Instruct
    - Mistral-Nemo-Instruct
    - Llama-2-7B-Chat-hf
    - Microsoft WizardLM-2-8x22B
    - Cohere Aya Expanse 8B
    - Amazon Titan-Express
    - Google PaLM-2
    - DeepSeek-R1
    - Google Gemma-7B-it
    - IBM Granite-3.1-2B-Instruct
    - Qwen2.5-1.5B-Instruct
    - Qwen-QwQ-32B-Preview
    - Anthropic Claude-3-sonnet
    - IBM Granite-3.2-2B-Instruct
    - Google Gemma-1.1-7B-it
    - Anthropic Claude-2
    - Google Flan-T5-large
    - Mixtral-8x7B-Instruct-v0.1
    - Llama-3.2-1B-Instruct
    - Apple OpenELM-3B-Instruct
    - Qwen2.5-0.5B-Instruct
    - Google Gemma-1.1-2B-it
    - TII falcon-7B-instruct
'''
from abc import ABC, abstractmethod
from src.logging.Logger import logger
from tqdm import tqdm
from src.config import OUTPUT_DIR
import os
import time
import re

MODEL_FAILED_TO_RETURN_OUTPUT = "MODEL FAILED TO RETURN ANY OUTPUT"
MODEL_RETURNED_NON_STRING_TYPE_OUTPUT = (
    "DID NOT RECIEVE A STRING TYPE FROM OUTPUT"
)
EMPTY_SUMMARY = (
    "THIS SUMMARY IS EMPTY, THIS IS THE DEFAULT VALUE A SUMMARY "
    "VARIABLE GETS. A REAL SUMMARY WAS NOT ASSIGNED TO THIS VARIABLE."
)
INCOMPLETE_THINK_TAG = "FOUND <think> WITH NO CLOSING </think>"

SUMMARY_ERRORS = [
    MODEL_FAILED_TO_RETURN_OUTPUT,
    MODEL_RETURNED_NON_STRING_TYPE_OUTPUT,
    EMPTY_SUMMARY,
    INCOMPLETE_THINK_TAG
]

class AbstractLLM(ABC):
    """
    Abstract Class

    Attributes:
        model_name (str): Name of the model
        prompt (str): Summary prompt
        company (str): Company of model
        temperature (float): set to 0.0 to compare deterministic output
        max_tokens (int): number of tokens for models
        min_throttle_time (float): minimum amount of time a request must run to
            avoid throttling 
        model_output_dir (str): path to output directory

    Methods:
        summarize_articles(articles): Summarizes the list of articles
        summarize_clean_wait(article): Summarizes article waits if needed 
            and cleans it
        try_to_summarize_one_article(article): exception handler method
        summarize_one_article(article): Requests summary of input
            article from LLM
        prepare_article_for_llm(article): Injects prompt and slightly reformats
            article text
        clean_raw_summary(raw_summary): cleans summary
        remove_thinking_text(raw_summary): removes thinking output
        get_model_identifier(model_name, date_code): get the model id expected
            by the company
        get_model_name(): returns name of model
        get_company(): get company of model
        get_model_out_dir(): get the output directory dedicated for this model
        get_date_code(): get date code
        get_temperature(): get temperature
        get_max_tokens(): get max tokens
        set_temperature(temp, reason): sets temperature
        summarize(prepared_text): Requests LLM to summarize the given text
        setup(): setup model for runtime use
        teardown(): teardown model when no longer needed for runtime use
    """

    def __init__(
            self,
            model_name: str, 
            date_code: str,
            temperature = 0.0,
            max_tokens = 1024,
            company="NullCompany",
            min_throttle_time=0
        ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        if date_code:
            self.date_code = date_code
        else:
            self.date_code = ""
        self.min_throttle_time = min_throttle_time
        self.company = company
        self.model_name = model_name
        self.prompt = ("You are a chat bot answering questions using data."
            "You must stick to the answers provided solely by the text in the "
            "passage provided. You are asked the question 'Provide a concise "
            "summary of the following passage, covering the core pieces of "
            "information described.'"
        )

        output_dir = OUTPUT_DIR
        self.model_output_dir = f"{output_dir}/{self.company}/{self.model_name}"
        os.makedirs(self.model_output_dir, exist_ok=True)

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_t):
        self.teardown()

    def summarize_articles(self, articles: list[str]) -> list[str]:
        """
        Takes in a list of articles, iterates through the list. Returns list of
        the summaries

        Args:
            articles (list[str]): List of strings where the strings are human
            written news articles

        Returns:
            list[str]: List of articles generated by the LLM
        """
        summaries = []
        for article in tqdm(articles, desc="Article Loop"):
            summary = self.summarize_clean_wait(article)
            summaries.append(summary)
        return summaries

    def summarize_clean_wait(self, article: str) -> str:
        """
        Given an article, requests a summary, halts until the minimum time for
        a request is met then cleans the output such that it only contains the 
        summary

        Args:
            article (str): Article text
        
        Returns:
            str: Output from LLM that only contains summary
        """

        start_time = time.time()
        raw_summary = self.try_to_summarize_one_article(article)
        summary = self.clean_raw_summary(raw_summary)
        elapsed_time = time.time() - start_time
        remaining_time = self.min_throttle_time - elapsed_time
        if remaining_time > 0:
            time.sleep(remaining_time)

        return summary

    def try_to_summarize_one_article(self, article: str) -> str:
        """
        Tries to request the model to summarize an Article. Logs warnings if it
        fails but continues the program with dummy output indicative of the 
        failure

        Args:
            Article (str): Article to be summarized

        Returns:
            str: Summary of article or dummy string output
        
        """
        llm_summary = EMPTY_SUMMARY

        try:
            llm_summary = self.summarize_one_article(article)
        except Exception as e:
            logger.warning((
                f"Model call failed for {self.model_name}: {e} "
            ))
            return MODEL_FAILED_TO_RETURN_OUTPUT

        if not isinstance(llm_summary, str):
            bad_output = llm_summary
            logger.warning((
                f"{self.model_name} returned unexpected output. Expected a "
                f"string but got {type(bad_output).__name__}. "
                f"Replacing output."
            ))
            return MODEL_RETURNED_NON_STRING_TYPE_OUTPUT
        return llm_summary

    def summarize_one_article(self, article: str) -> str:
        """
        Takes in a string representing a human written article, injects a prompt
        and feeds into the LLM to generate a summary.

        Args:
            article (str): String that is a human written news article
        
        Returns:
            str: A summary of the article generated by the LLM

        """
        prepared_llm_input = self.prepare_article_for_llm(article)

        llm_summary = self.summarize(prepared_llm_input)

        return llm_summary

    def prepare_article_for_llm(self, article: str) -> str:
        """
        Combines prompt and article for input into an LLM

        Args:
            text (str): Content text for LLM

        Returns:
            str: Prompt + content text
        """
        prepared_text = f"{self.prompt} '{article}'"
        return prepared_text

    def clean_raw_summary(self, raw_summary: str) -> str:
        """
        Cleans the output summary to only contains relevant summary data

        Args:
            raw_summary (str): raw_summary output by LLM

        Returns:
            str: string that only contains summary data
        """
        summary = self.remove_thinking_text(raw_summary)
        if summary in SUMMARY_ERRORS:
            return summary
        return summary

    def remove_thinking_text(self, raw_summary: str) -> str:
        """
        Removes any thinking tags and content in between them. If a summary does
        not have a closing thinking tag it will be considered as an incomplete 
        summary and return an error string instead.

        Args:
            raw_summary (str): raw summary from LLM

        returns:
            str: summary without thinking data or invalid summary text

        """
        if '<think>' in raw_summary and '</think>' not in raw_summary:
            logger.warning(f"<think> tag found with no </think>. This is indicative of an incomplete response from an LLM. Raw Summary: {raw_summary}")
            return INCOMPLETE_THINK_TAG

        summary = re.sub(
            r'<think>.*?</think>\s*', '',
            raw_summary, flags=re.DOTALL
        )
        return summary

    def get_model_identifier(self, model_name: str, date_code: str) -> str:
        """
        Combines model_name and its date_code if the date_code isn't an emtpy
        string otherwise model_name

        Args:
            model_name (str): base model name
            date_code (str): date code of model

        Returns:
            str: full model identifier
        
        """
        model = f"{model_name}"
        if date_code != "":
            model = f"{model_name}-{date_code}"
        return model

    def get_model_name(self):
        """
        Returns name of model

        Args:
            None

        Returns:
            str: Name of model
        """
        return self.model_name

    def get_company(self):
        """
        Returns company name of model

        Args:
            None

        Returns:
            str: Company name of model
        """
        return self.company

    def get_model_out_dir(self):
        """
        Returns path to model output directory

        Args:
            None

        Returns:
            str: Path to model output directory
        """
        return self.model_output_dir

    def get_date_code(self) -> str:
        """
        Gets date code

        Args:
            None
        
        Returns:
            str: date code
        """
        return self.date_code

    def get_temperature(self) -> float:
        """
        Gets temperature

        Args:
            None

        Returns
            float: temperature
        """
        return self.temperature

    def get_max_tokens(self) -> int:
        """
        Get max tokens

        Args:
            None

        Returns:
            int: max tokens
        """
        return self.max_tokens

    def set_temperature(self, temp: float, reason="no reason given"):
        """
        Sets temperature, optionally can provide a reason.

        Args:
            temp (float): temperature
            reason (str): reason for changing temp
        """
        logger.warning(
            f"Temperature for {self.model_name} was changed from "
            f"{self.temperature} to {temp} because: {reason}"
        )
        print(
            f"Temperature for {self.model_name} was changed from "
            f"{self.temperature} to {temp} because: {reason}"
        )
        self.temperature = temp

    @abstractmethod
    def summarize(self, prepared_text: str) -> str:
        """
        Requests LLM to generate a summary given the input

        Args:
            prepared_text (str): Prompt prepared text

        Returns:
            str: Generated LLM summary
        """
        return None

    @abstractmethod
    def setup(self):
        """
        Setup model for use

        Args:
            None
        Returns:
            None
        """
        return None

    @abstractmethod
    def teardown(self):
        """
        Teardown model

        Args:
            None
        Returns:
            None
        """
        return None
