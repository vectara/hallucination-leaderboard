from typing import List, Literal, Dict
import re
import string

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
    def __init__(self):
        self.PROMPT_TEMPLATE = "Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"

        self.CHECKPOINT = "vectara/hhem-2.3"
        self.FOUNDATION = "meta-llama/Llama-3.2-3B"
        self.DEVICE= "cuda" if torch.cuda.is_available() else "cpu"
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

if __name__ == "__main__":

    test_data = ("The sky is blue", "The universe is blue")
    
    hhem_2_1_open = HHEM_2_1_open()
    print (hhem_2_1_open.predict(*test_data))

    hhem_2_3 = HHEM_2_3()
    print (hhem_2_3.predict(*test_data))
