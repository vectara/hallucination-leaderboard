"""HHEM (Hughes Hallucination Evaluation Model) implementations.

This module provides multiple implementations of the HHEM model for evaluating
factual consistency between a premise (source text) and hypothesis (generated text).
It includes both local model inference and API-based approaches.

Classes:
    HHEMOutput: Pydantic model for standardized prediction output.
    HHEM_2_1_open: Open-source HHEM 2.1 using Flan-T5 foundation.
    HHEM_2_3: HHEM 2.3 using Llama-3.2-3B foundation with pipeline interface.
    HHEM_2_3_PROD: Production-style HHEM 2.3 with direct model inference.
    HHEM_2_3_API: HHEM 2.3 via Vectara's REST API.

Example:
    >>> from src.HHEM_2_x import HHEM_2_1_open
    >>> model = HHEM_2_1_open()
    >>> result = model.predict("The capital of France is Paris.", "Paris is in France.")
    >>> print(result.score, result.label)
"""

from typing import List, Literal, Dict
import re
import string
import requests
import os
import json
from typing import Any
import time
from . Logger import logger

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from transformers.tokenization_utils_base import BatchEncoding

def clean_string(s: str) -> str:
    """Clean and normalize text for improved HHEM evaluation accuracy.

    Performs text preprocessing including whitespace normalization, punctuation
    cleanup, and ensuring proper sentence termination. This standardization
    helps improve consistency in HHEM score calculations.

    Args:
        s: The input string to clean.

    Returns:
        The cleaned string with normalized whitespace, corrected punctuation
        spacing, and guaranteed terminal punctuation.

    Example:
        >>> clean_string("  Hello   world  .")
        'Hello world.'
    """
    s = s.strip()
    # remove citations in the form of [1], [2], etc.
    # s = re.sub(r'\[\d+\]', '', s)
    # for any square brackets, remove the brackets but keep the content
    # s = re.sub(r'\[(.*?)\]', r'\1', s)
    # remove special characters
    # s = re.sub(r'[^\w\s]', '', s)
    # remove extra whitespace
    s = re.sub(r'\s+', ' ', s)
    # remove spaces before any punctuation
    s = re.sub(r'\s([.,!?])', r'\1', s)
    # if the string does not end with a punctuation, add a period
    if not any([s.endswith(p) for p in string.punctuation]):
        s = s + '.'
    return s

class HHEMOutput(BaseModel):
    """Standardized output model for HHEM predictions.

    Provides a consistent interface for all HHEM model variants, containing
    both the raw consistency score and the thresholded binary label.

    Attributes:
        score: Factual consistency score between 0 and 1, where higher values
            indicate greater consistency. Used for ROC curve analysis and
            fine-grained evaluation.
        label: Binary classification label (0 = inconsistent/hallucinated,
            1 = consistent/factual) derived from applying a threshold to the score.
    """

    score: float
    label: Literal[0, 1]

class HHEM_2_1_open():
    """Open-source HHEM 2.1 model for hallucination detection.

    Uses the Vectara hallucination evaluation model built on Google's Flan-T5-small
    foundation. This is a lightweight, open-source option suitable for environments
    where the larger Llama-based models are not available.

    Attributes:
        PROMPT_TEMPLATE: Template string for formatting premise-hypothesis pairs.
        classifier: HuggingFace text classification pipeline.
    """

    def __init__(self):
        """Initialize the HHEM 2.1 open model.

        Loads the vectara/hallucination_evaluation_model checkpoint with the
        google/flan-t5-small tokenizer and creates a text classification pipeline.
        """
        self.PROMPT_TEMPLATE = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"

        CHECKPOINT = "vectara/hallucination_evaluation_model"
        FOUNDATION = "google/flan-t5-small"

        tokenizer = AutoTokenizer.from_pretrained(FOUNDATION)

        self.classifier = pipeline("text-classification", model=CHECKPOINT, tokenizer=tokenizer, trust_remote_code=True)

    def __str__(self):
        """Return the model identifier string."""
        return "HHEM-2.1-open"

    def predict(self, premise: str, hypothesis: str) -> HHEMOutput:
        """Evaluate factual consistency between premise and hypothesis.

        Args:
            premise: The source/reference text to check against.
            hypothesis: The generated text to evaluate for consistency.

        Returns:
            HHEMOutput containing the consistency score and binary label.
        """
        texts_prompted: List[str] = [self.PROMPT_TEMPLATE.format(text1=premise, text2=hypothesis)]

        full_scores = self.classifier(texts_prompted, top_k=None) # List[List[Dict[str, float]]]

        # Optional: Extract the scores for the 'consistent' label
        simple_scores = [score_dict['score'] for score_for_both_labels in full_scores for score_dict in score_for_both_labels if score_dict['label'] == 'consistent']

        threshold = 0.5
        preds = [0 if s < threshold else 1 for s in simple_scores]

        return HHEMOutput(score=simple_scores[0], label=preds[0])

class HHEM_2_3():
    """HHEM 2.3 model using HuggingFace pipeline interface.

    Uses the Vectara HHEM 2.3 checkpoint built on Meta's Llama-3.2-3B foundation.
    Automatically selects between CPU (float32) and CUDA (bfloat16 with flash
    attention) based on hardware availability.

    Attributes:
        PROMPT_TEMPLATE: Template string for formatting premise-hypothesis pairs.
        classifier: HuggingFace text classification pipeline.
    """

    def __init__(self):
        """Initialize the HHEM 2.3 model with automatic device selection.

        Loads the vectara/hhem-2.3 checkpoint with the meta-llama/Llama-3.2-3B
        tokenizer. Configures optimal dtype and attention implementation based
        on whether CUDA is available.
        """
        self.PROMPT_TEMPLATE = "Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"

        CHECKPOINT = "vectara/hhem-2.3"
        FOUNDATION = "meta-llama/Llama-3.2-3B"
        DEVICE= "cuda" if torch.cuda.is_available() else "cpu"
        model_load_options = {
            "cpu": {
                "torch_dtype": torch.float32,
                "use_flash_attention_2": False,
            },
            "cuda": {
                "torch_dtype": torch.bfloat16,
                "use_flash_attention_2": True,
            }
        }

        tokenizer = AutoTokenizer.from_pretrained(FOUNDATION)

        self.classifier = pipeline("text-classification", model=CHECKPOINT, tokenizer=tokenizer, device=DEVICE)

    def __str__(self):
        """Return the model identifier string."""
        return "HHEM-2.3"

    def predict(self, premise: str, hypothesis: str) -> HHEMOutput:
        """Evaluate factual consistency between premise and hypothesis.

        Args:
            premise: The source/reference text to check against.
            hypothesis: The generated text to evaluate for consistency.

        Returns:
            HHEMOutput containing the consistency score and binary label.
        """
        texts_prompted: List[str] = [self.PROMPT_TEMPLATE.format(text1=premise, text2=hypothesis)]

        full_scores = self.classifier(texts_prompted, top_k=None) # List[List[Dict[str, float]]]

        simple_scores = [score_dict['score'] for score_for_both_labels in full_scores for score_dict in score_for_both_labels if score_dict['label'] == 'LABEL_1']

        threshold = 0.5
        preds = [0 if s < threshold else 1 for s in simple_scores]

        return HHEMOutput(score=simple_scores[0], label=preds[0])

class HHEM_2_3_PROD():
    """Production-style HHEM 2.3 with direct model inference.

    Mirrors the inference logic used in production environments while supporting
    local GPU execution. Uses direct model calls instead of the pipeline interface
    for finer control over tokenization and inference. Includes text preprocessing
    via clean_string for improved consistency.

    Attributes:
        PROMPT_TEMPLATE: Template string for formatting premise-hypothesis pairs.
        CHECKPOINT: HuggingFace model checkpoint identifier.
        FOUNDATION: Base model used for tokenization.
        DEVICE: Inference device ('cuda' or 'cpu').
        tokenizer: Llama tokenizer instance.
        model: Sequence classification model in evaluation mode.
    """

    def __init__(self):
        """Initialize the production-style HHEM 2.3 model.

        Loads the model with flash attention and automatic device mapping.
        Sets the model to evaluation mode for deterministic inference.
        """
        self.PROMPT_TEMPLATE = "Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"

        self.CHECKPOINT = "vectara/hhem-2.3"
        self.FOUNDATION = "meta-llama/Llama-3.2-3B"
        self.DEVICE= "cuda" if torch.cuda.is_available() else "cpu"
        model_load_options = {
            "cpu": {
                "torch_dtype": torch.float32,
                "attn_implementation": "flash_attention_2",
                "device_map": "auto"
            },
            "cuda": {
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2",
                "device_map": "auto"
            }
        }
        self.tokenizer = AutoTokenizer.from_pretrained(self.FOUNDATION)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.CHECKPOINT, **model_load_options[self.DEVICE])
        self.model.eval()

    def __str__(self):
        """Return the model identifier string."""
        return "HHEM-2.3-PROD"

    def predict(self, premise: str, hypothesis: str) -> HHEMOutput:
        """Evaluate factual consistency with production-style preprocessing.

        Applies text cleaning to both inputs before evaluation for improved
        consistency with production behavior.

        Args:
            premise: The source/reference text to check against.
            hypothesis: The generated text to evaluate for consistency.

        Returns:
            HHEMOutput containing the consistency score and binary label.
        """
        premise = clean_string(premise)
        hypothesis = clean_string(hypothesis)
        prompted_text = self.PROMPT_TEMPLATE.format(text1=premise, text2=hypothesis)

        tokenized_inputs: BatchEncoding = self.tokenizer(prompted_text, return_tensors='pt', truncation=False, padding=False)
        tokenized_inputs = tokenized_inputs.to(self.DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**tokenized_inputs)

        logits = outputs.logits[0, :]
        logits = torch.softmax(logits, dim=-1)
        hhem_pred = logits.argmax(dim=-1).item()
        hhem_score = logits[1].item()

        return HHEMOutput(score=hhem_score, label=hhem_pred)

class HHEM_2_3_API():
    """HHEM 2.3 client using Vectara's REST API.

    Provides hallucination detection via Vectara's hosted API service, eliminating
    the need for local GPU resources. Implements exponential backoff retry logic
    for robust production use.

    Attributes:
        PROMPT_TEMPLATE: Template string for formatting (unused in API mode but
            kept for interface consistency).
        max_retries: Maximum number of retry attempts for failed API calls.
        retry_delay: Base delay in seconds for exponential backoff.
        api_key: Vectara API key loaded from VECTARA_HHEM_API_KEY environment variable.

    Raises:
        AssertionError: If VECTARA_HHEM_API_KEY environment variable is not set.
    """

    def __init__(self):
        """Initialize the HHEM API client.

        Loads the API key from the VECTARA_HHEM_API_KEY environment variable
        and configures retry parameters.

        Raises:
            AssertionError: If the API key environment variable is not set.
        """
        self.PROMPT_TEMPLATE = "Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
        self.max_retries = 6
        self.retry_delay = 1
        self.api_key = os.getenv(f"VECTARA_HHEM_API_KEY")
        assert self.api_key is not None, (
            f"VECTARA_HHEM_API_KEY not found in environment variable "
        )

    def __str__(self):
        """Return the model identifier string."""
        return "HHEM-2.3-API"

    def try_to_get_hhem_score(self, article: str, summary: str) -> dict[str, Any]:
        """Attempt to get HHEM score with exponential backoff retry.

        Args:
            article: The source/premise text.
            summary: The generated/hypothesis text to evaluate.

        Returns:
            The HHEM consistency score from the API.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return self.get_hhem_score(article, summary)

            except Exception as e:
                last_exception = e
                print(f"[HHEM Score] Attempt {attempt}/{self.max_retries} failed: {e}")
                logger.warning(
                    f"HHEM Score attempt {attempt}/{self.max_retries} failed: {e}"
                )

                if attempt < self.max_retries:
                    backoff = self.retry_delay * (2 ** (attempt - 1))
                    time.sleep(backoff)

        logger.critical(
            f"Failed to get HHEM score after {self.max_retries} attempts: {last_exception}"
        )
        raise RuntimeError(
            f"Failed to get HHEM score after {self.max_retries} attempts: {last_exception}"
        )

    def get_hhem_score(self, article: str, summary: str) -> dict[str, Any]:
        """Make a single API request to evaluate factual consistency.

        Args:
            article: The source/premise text.
            summary: The generated/hypothesis text to evaluate.

        Returns:
            The HHEM consistency score from the API response.

        Raises:
            RuntimeError: If the API request fails or response parsing fails.
        """
        payload = {
            "generated_text": summary,
            "source_texts": [article]
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-api-key": self.api_key
        }
        
        try:
            response = requests.post(
                "https://api.vectara.io/v2/evaluate_factual_consistency",
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()['score']
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"HHEM API request failed: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse HHEM response: {e}")

    def predict(self, premise: str, hypothesis: str) -> HHEMOutput:
        """Evaluate factual consistency via the Vectara API.

        Applies text cleaning to both inputs before sending to the API
        for consistency with local model behavior.

        Args:
            premise: The source/reference text to check against.
            hypothesis: The generated text to evaluate for consistency.

        Returns:
            HHEMOutput containing the consistency score and binary label.
        """
        premise = clean_string(premise)
        hypothesis = clean_string(hypothesis)
        threshold = 0.5
        hhem_score = self.try_to_get_hhem_score(premise, hypothesis)
        hhem_pred = 0 if hhem_score < threshold else 1

        return HHEMOutput(score=hhem_score, label=hhem_pred)

if __name__ == "__main__":

    test_data = ("The sky is blue", "The universe is blue")
    
    hhem_2_1_open = HHEM_2_1_open()
    print (hhem_2_1_open.predict(*test_data))

    hhem_2_3 = HHEM_2_3()
    print (hhem_2_3.predict(*test_data))
