"""Vectara model implementations for hallucination evaluation.

This module provides the LLM implementation for Vectara's RAG (Retrieval
Augmented Generation) platform, supporting API-based inference via the
Vectara API with corpus-based search and generation. Includes a custom
HTTP client for communicating with the Vectara platform.

Classes:
    VectaraClient: Custom HTTP client for Vectara API communication.
    VectaraConfig: Configuration model for Vectara model settings.
    VectaraSummary: Output model for Vectara summarization results.
    ClientMode: Enum for API client execution modes.
    LocalMode: Enum for local model execution modes.
    VectaraLLM: Main LLM class implementing AbstractLLM for Vectara models.

Attributes:
    COMPANY: Provider identifier string ("vectara").
    client_mode_group: Mapping of models to supported API client modes.
    local_mode_group: Mapping of models to local execution modes (empty).
"""

import os
from typing import Literal
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

import http.client
import json

COMPANY = "vectara"
"""str: Provider identifier used for API key lookup and model registration."""

class VectaraClient:
    """Custom HTTP client for Vectara API communication.

    Provides a simple interface for querying Vectara corpora with RAG
    capabilities including search, reranking, and generation. Uses the
    Vectara v2 API with HTTPS connections.

    Attributes:
        api_key: The Vectara API key for authentication.
        corpus_key: The identifier for the corpus to query.
        conn: HTTPS connection to the Vectara API endpoint.
        headers: HTTP headers including content type and API key.
    """

    def __init__(self, api_key: str, corpus_key: str):
        """Initialize the Vectara client with authentication credentials.

        Args:
            api_key: The Vectara API key for authentication.
            corpus_key: The identifier for the corpus to query.
        """
        self.api_key = api_key
        self.corpus_key = corpus_key
        self.conn = http.client.HTTPSConnection("api.vectara.io")
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'x-api-key': self.api_key
        }

    def summarize(self, prompt: str) -> str:
        """Query the Vectara corpus and generate a summary.

        Sends a query to the Vectara API with search, reranking, and generation
        configuration. Uses lexical interpolation, the Rerank_Multilingual_v1
        reranker, and the mockingbird-2.0 generation preset.

        Args:
            prompt: The query text to search and summarize.

        Returns:
            The generated summary from Vectara, or "No Summary Returned."
            if the response lacks a summary field.
        """
        payload = json.dumps({
            "query": prompt,
            "search": {
                # "metadata_filter": "doc.topic = 'authentication' and doc.platform = 'kubernetes'",
                "lexical_interpolation": 0.005,
                "limit": 50,
                "context_configuration": {
                    "sentences_before": 2,
                    "sentences_after": 2,
                    "start_tag": "<em>",
                    "end_tag": "</em>"
                },
                "reranker": {
                    "type": "customer_reranker",
                    "reranker_name": "Rerank_Multilingual_v1",
                    "limit": 50,
                    "include_context": True
                }
            },
            "generation": {
                "generation_preset_name": "mockingbird-2.0",
                # "max_used_search_results": 10,
                # "citations": {
                #     "style": "markdown",
                #     "url_pattern": "https://vectara.com/documents/{doc.id}",
                #     "text_pattern": "{doc.title}"
                # }
            },
            "save_history": True,
            "intelligent_query_rewriting": True
        })

        self.conn.request(
            "POST",
            f"/v2/corpora/{self.corpus_key}/query",
            payload,
            self.headers
        )
        res = self.conn.getresponse()
        data = res.read()
        result = json.loads(data.decode("utf-8"))

        summary = result.get("summary", "No Summary Returned.")
        return summary

class VectaraConfig(BasicLLMConfig):
    """Configuration model for Vectara models.

    Extends BasicLLMConfig with Vectara-specific settings for model selection
    and API configuration. Supports various summarization models including
    the mockingbird-2.0 generation preset.

    Attributes:
        company: Provider identifier, fixed to "vectara".
        model_name: Name of the Vectara model/preset to use. Options include
            manual_short_summary, manual_long_summary, and mockingbird-2.0.
        date_code: Optional version/date identifier for the model.
        execution_mode: Where to run inference, currently only "api" supported.
        endpoint: API endpoint type ("chat" for conversational format).
    """

    company: Literal["vectara"] = "vectara"
    model_name: Literal[
        "manual_short_summary",
        "manual_long_summary",
        "mockingbird-2.0"
    ]
    date_code: str = ""
    execution_mode: Literal["api"] = "api"
    endpoint: Literal["chat", "response"] = "chat"

class VectaraSummary(BasicSummary):
    """Output model for Vectara summarization results.

    Extends BasicSummary with endpoint tracking for result provenance.

    Attributes:
        endpoint: The API endpoint type used for generation, if applicable.
    """

    endpoint: Literal["chat", "response"] | None = None

    class Config:
        """Pydantic configuration to ignore extra fields during parsing."""

        extra = "ignore"

class ClientMode(Enum):
    """Execution modes for API client inference.

    Defines how the model should be invoked when using the Vectara API
    via the custom VectaraClient.

    Attributes:
        CHAT_DEFAULT: Use the VectaraClient.summarize method for RAG queries.
        RESPONSE_DEFAULT: Use the completion/response API endpoint.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()


class LocalMode(Enum):
    """Execution modes for local model inference.

    Defines how the model should be invoked when running locally.
    Currently unused as Vectara models are accessed via API only.

    Attributes:
        CHAT_DEFAULT: Use chat template formatting for input.
        RESPONSE_DEFAULT: Use direct completion without chat template.
        UNDEFINED: Mode not defined or not supported.
    """

    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

# client_mode_group: Mapping of model names to their supported API client modes.
# Each model maps endpoint types to ClientMode enum values indicating how to
# invoke the Vectara API via the VectaraClient.
client_mode_group = {
    "mockingbird-2.0": {
        "chat": ClientMode.CHAT_DEFAULT
    }
}

# local_mode_group: Mapping of model names to their supported local execution modes.
# Empty dict indicates Vectara models are accessed via API only.
local_mode_group = {}


class VectaraLLM(AbstractLLM):
    """LLM implementation for Vectara models.

    Provides text summarization using Vectara's RAG platform via the custom
    VectaraClient. Supports corpus-based search with reranking and generation
    using presets like mockingbird-2.0.

    Attributes:
        endpoint: The API endpoint type (e.g., "chat").
        execution_mode: Where inference runs (currently only "api" supported).
        full_config: Complete configuration object for reference.
    """

    def __init__(self, config: VectaraConfig):
        """Initialize the Vectara LLM with the given configuration.

        Args:
            config: Configuration object specifying model and API settings.
        """
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.full_config = config

    def summarize(self, prepared_text: str) -> str:
        """Generate a summary of the provided text.

        Uses the VectaraClient to query the configured corpus and generate
        a RAG-based summary. The client handles search, reranking, and
        generation using the mockingbird-2.0 preset.

        Args:
            prepared_text: The preprocessed text to summarize.

        Returns:
            The generated summary text, or an error placeholder if generation fails.

        Raises:
            Exception: If neither client nor local_model is initialized.
        """
        summary = SummaryError.EMPTY_SUMMARY
        if self.client:
            match client_mode_group[self.model_name][self.endpoint]:
                case ClientMode.CHAT_DEFAULT:
                    summary = self.client.summarize(prepared_text)
        elif self.local_model: 
            pass
        else:
            raise Exception(
                ModelInstantiationError.MISSING_SETUP.format(
                    class_name=self.__class__.__name__
                )
            )
        return summary

    def setup(self):
        """Initialize the VectaraClient for model inference.

        Creates a VectaraClient instance configured with the API key from
        the VECTARA_API_KEY environment variable and a default corpus key.

        Raises:
            AssertionError: If the API key environment variable is not set.
            Exception: If the model does not support the configured execution mode.
        """
        if self.execution_mode == "api":
            if self.model_name in client_mode_group:
                api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
                assert api_key is not None, (
                    f"{COMPANY} API key not found in environment variable "
                    f"{COMPANY.upper()}_API_KEY"
                )
                self.client = VectaraClient(api_key=api_key, corpus_key="my-corpus")
            else:
                raise Exception(
                    ModelInstantiationError.CANNOT_EXECUTE_IN_MODE.format(
                        model_name=self.model_name,
                        company=self.company,
                        execution_mode=self.execution_mode
                    )
                )
        elif self.execution_mode == "local":
            pass

    def teardown(self):
        """Clean up resources after inference is complete.

        Releases any held resources from the client or local model.
        Currently a no-op as cleanup is handled automatically.
        """
        if self.client:
            pass
        elif self.local_model:
            pass

    def close_client(self):
        """Close the VectaraClient connection.

        Currently a no-op as the HTTP connection is managed by the client
        and does not require explicit cleanup.
        """
        pass