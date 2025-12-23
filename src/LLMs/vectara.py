import os
from typing import Literal
from enum import Enum, auto

from . AbstractLLM import AbstractLLM
from .. data_model import BasicLLMConfig, BasicSummary, BasicJudgment
from .. data_model import ModelInstantiationError, SummaryError

import http.client
import json

COMPANY = "vectara"

class VectaraClient:
    def __init__(self, api_key: str, corpus_key: str):
        self.api_key = api_key
        self.corpus_key = corpus_key
        self.conn = http.client.HTTPSConnection("api.vectara.io")
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'x-api-key': self.api_key
        }

    def summarize(self, prompt: str) -> str:
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
    """Extended config for vectara-specific properties"""
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
    endpoint: Literal["chat", "response"] | None = None

    class Config:
        extra = "ignore"

class ClientMode(Enum):
    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

class LocalMode(Enum):
    CHAT_DEFAULT = auto()
    RESPONSE_DEFAULT = auto()
    UNDEFINED = auto()

client_mode_group = {
    "mockingbird-2.0":{
        "chat": ClientMode.CHAT_DEFAULT
    }
}

local_mode_group = {}


class VectaraLLM(AbstractLLM):
    """
    Class for models from vectara
    """
    def __init__(self, config: VectaraConfig):
        super().__init__(config)
        self.endpoint = config.endpoint
        self.execution_mode = config.execution_mode
        self.full_config = config

    def summarize(self, prepared_text: str) -> str:
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
        if self.client:
            pass
        elif self.local_model:
            pass

    def close_client(self):
        pass