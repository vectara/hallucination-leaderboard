from typing import List, Literal, Tuple

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

from hdm2 import HallucinationDetectionModel

import sys
import io
from contextlib import redirect_stdout

# idk if right
class HDM2Output(BaseModel):
    score: float # we need score for ROC curve
    label: Literal[0,1]
    reason: List[Tuple[str, float]]

class HDM2():
    def __init__(self):
        # self.PROMPT_TEMPLATE = "Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
        self.prompt = """
            Your task is to provide a concise and factual summary for the given passage.

            Rules
            1. Summarize using only the information in the given passage. Do not infer. Do not use your internal knowledge.
            2. Do not provide a preamble or explanation, output only the summary.
            3. Summaries should never exceed 20 percent of the passage's length.
            4. Maintain a neutral tone.

            If you are unable to summarize the passage due to missing, unreadable, irrelevant or insufficient content, respond only with:
            "I am unable to summarize this passage."
            Here is the passage:
            {article}
        """
        self.classifier = HallucinationDetectionModel()

    def __str__(self):
        return "HDM-2"

    def predict(self, premise: str, hypothesis: str) -> HDM2Output:
        texts_prompted: List[str] = [self.prompt]

        f = io.StringIO()  # buffer to catch prints

        with redirect_stdout(f):
            results = self.classifier.apply(self.prompt, premise, hypothesis)

        simple_score = 1.0 - results['adjusted_hallucination_severity']

        hallucinated_sentences = []

        if results['candidate_sentences']:
            is_ck_hallucinated = False
            for sentence_result in results['ck_results']:
                if sentence_result['prediction'] == 1:  # 1 indicates hallucination
                    hallucinated_sentence = sentence_result['text']
                    hallucinated_sentence_score = sentence_result['hallucination_probability']
                    hallucinated_sentences.append((hallucinated_sentence, hallucinated_sentence_score))
                    is_ck_hallucinated = True
            if not is_ck_hallucinated:
                hallucinated_sentences = [("No hallucinated sentences detected.", 0.0)]
        else:
            hallucinated_sentences = [("No hallucinated sentences detected.", 0.0)]

        threshold = 0.5
        if simple_score < threshold:
            pred = 0
        else:
            pred = 1
        # pred = 0 if s < threshold else 1 for s in simple_score

        return HDM2Output(score=simple_score, label=pred, reason=hallucinated_sentences)

    # def predict(self, premise: str, hypothesis: str) -> HDM2Output:
    #     texts_prompted: List[str] = [self.PROMPT_TEMPLATE.format(text1=premise, text2=hypothesis)]

    #     full_scores = self.classifier(self.prompt, premise, hypothesis) # List[List[Dict[str, float]]]

    #     simple_scores = [score_dict['score'] for score_for_both_labels in full_scores for score_dict in score_for_both_labels if score_dict['label'] == 'LABEL_1']

    #     threshold = 0.5
    #     preds = [0 if s < threshold else 1 for s in simple_scores]

    #     return HDM2Output(score=simple_scores[0], label=preds[0], reason=?)

if __name__ == "__main__":

    test_data = ("The sky is blue", "The universe is blue")
    
    hdm_2 = HDM2()
    print (hdm_2.predict(*test_data))