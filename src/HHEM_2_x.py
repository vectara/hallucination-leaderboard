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
    """
    Clean up the string to improve HHEM's performance.
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
    score: float # we need score for ROC curve
    label: Literal[0,1]

class HHEM_2_1_open():
    def __init__(self):
        self.PROMPT_TEMPLATE = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"

        CHECKPOINT = "vectara/hallucination_evaluation_model"
        FOUNDATION = "google/flan-t5-small"

        tokenizer = AutoTokenizer.from_pretrained(FOUNDATION)

        self.classifier = pipeline("text-classification", model=CHECKPOINT, tokenizer=tokenizer, trust_remote_code=True)

    def __str__(self):
        return "HHEM-2.1-open"

    def predict(self, premise: str, hypothesis: str) -> HHEMOutput:
        texts_prompted: List[str] = [self.PROMPT_TEMPLATE.format(text1=premise, text2=hypothesis)]

        full_scores = self.classifier(texts_prompted, top_k=None) # List[List[Dict[str, float]]]

        # Optional: Extract the scores for the 'consistent' label
        simple_scores = [score_dict['score'] for score_for_both_labels in full_scores for score_dict in score_for_both_labels if score_dict['label'] == 'consistent']

        threshold = 0.5
        preds = [0 if s < threshold else 1 for s in simple_scores]

        return HHEMOutput(score=simple_scores[0], label=preds[0])

class HHEM_2_3():
    def __init__(self):
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
        return "HHEM-2.3"

    def predict(self, premise: str, hypothesis: str) -> HHEMOutput:
        texts_prompted: List[str] = [self.PROMPT_TEMPLATE.format(text1=premise, text2=hypothesis)]

        full_scores = self.classifier(texts_prompted, top_k=None) # List[List[Dict[str, float]]]

        simple_scores = [score_dict['score'] for score_for_both_labels in full_scores for score_dict in score_for_both_labels if score_dict['label'] == 'LABEL_1']

        threshold = 0.5
        preds = [0 if s < threshold else 1 for s in simple_scores]

        return HHEMOutput(score=simple_scores[0], label=preds[0])

class HHEM_2_3_PROD():
    """GPU ran but using identical logic as production environment"""
    def __init__(self):
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
        return "HHEM-2.3-PROD"

    def predict(self, premise: str, hypothesis: str) -> HHEMOutput:
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
    """Get HHEM results from Vectara API"""
    def __init__(self):
        self.PROMPT_TEMPLATE = "Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
        self.max_retries = 6
        self.retry_delay = 1

    def __str__(self):
        return "HHEM-2.3-API"

    def try_to_get_hhem_score(self, article: str, summary: str) -> dict[str, Any]:
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
        api_key = os.getenv(f"VECTARA_HHEM_API_KEY")
        assert api_key is not None, (
            f"VECTARA_HHEM_API_KEY not found in environment variable "
        )
        
        payload = {
            "generated_text": summary,
            "source_texts": [article]
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-api-key": api_key
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
