from typing import List, Literal
from pydantic import BaseModel
# import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

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

if __name__ == "__main__":

    test_data = ("The sky is blue", "The universe is blue")
    
    hhem_2_1_open = HHEM_2_1_open()
    print (hhem_2_1_open.predict(*test_data))

    hhem_2_3 = HHEM_2_3()
    print (hhem_2_3.predict(*test_data))
