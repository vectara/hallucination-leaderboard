# TODO: Create children for all models on the HHEM LB
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
import os

MODEL_FAILED_TO_RETURN_OUTPUT = "MODEL FAILED TO RETURN ANY OUTPUT"
MODEL_RETURNED_NON_STRING_TYPE_OUTPUT = "DID NOT RECIEVE A STRING TYPE FROM OUTPUT"


class AbstractLLM(ABC):
    """
    Abstract Class

    Attributes:
        model_name (str): Name of the model
        prompt (str): Summary prompt
        company (str): Company of model
        temperature (float): set to 0.0 to compare deterministic output
        max_tokens (int): number of tokens for models


    Methods:
        summarize_articles(articles): Requests summary for a given
            list of articles
        try_to_summarize_one_article(article): exception handler method
        summarize_one_article(article): Requests summary of input
            article from LLM
        prepare_article_for_llm(article): Injects prompt and slightly reformats
            article text
        get_model_name(): returns name of model
        summarize(prepared_text): Requests LLM to summarize the given text
        setup(): setup model for runtime use
        teardown(): teardown model when no longer needed for runtime use
        get_name(): get name of model
        get_company(): get company of model
        get_model_out_dir(): get the output directory dedicated for this model
    """
    def __init__(self, model_name: str, company="NullCompany"):
        self.max_tokens = 1024
        self.temperature = 0.0
        self.company = company
        self.model_name = model_name
        '''Do we need a pad token at start?'''
        self.prompt = ("You are a chat bot answering questions using data."
            "You must stick to the answers provided solely by the text in the "
            "passage provided. You are asked the question 'Provide a concise "
            "summary of the following passage, covering the core pieces of "
            "information described.'"
        )

        output_dir = os.getenv("OUTPUT_DIR")
        model_output_dir = f"{output_dir}/{self.company}/{self.model_name}"
        self.model_output_dir = model_output_dir
        os.makedirs(model_output_dir, exist_ok=True)

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
            summary = self.try_to_summarize_one_article(article)
            summaries.append(summary)
        return summaries

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
        llm_summary = None

        try:
            llm_summary = self.summarize_one_article(article)
        except Exception as e:
            logger.log((
                f"~WARNING~ Model call failed for {self.model_name}: {e} "
            ))
            return MODEL_FAILED_TO_RETURN_OUTPUT

        if not isinstance(llm_summary, str):
            bad_output = llm_summary
            logger.log((
                f"~WARNING~ {self.model_name} returned unexpected output. Expected a "
                f"string but got {type(bad_output).__name__}. "
                f"Replacing output."
            ))
            return MODEL_RETURNED_NON_STRING_TYPE_OUTPUT
        return llm_summary


    def summarize_one_article(self, article: str) -> str:
        """
        Takes in a string representing a human written article, injects a prompt
        at the start and feeds into the LLM to generate a summary.

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
