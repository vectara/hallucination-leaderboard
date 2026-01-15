"""Wrapper for the HDM-2 (Hallucination Detection Model) external library.

This module provides an interface to HDM-2, an open-source hallucination
detection model from a separate research group. HDM-2 is NOT part of the
HHEM (Hughes Hallucination Evaluation Model) family developed by Vectara;
it is included for comparison purposes in experiments.

HDM-2 provides sentence-level hallucination detection with explanations,
identifying which specific sentences in a summary are hallucinated and
their associated confidence scores.

Classes:
    HDM2Output: Pydantic model for structured prediction results.
    HDM2: Wrapper class for the HallucinationDetectionModel.

Note:
    Requires the external `hdm2` package to be installed. The model
    outputs are suppressed during inference to reduce console noise.

Example:
    >>> hdm2 = HDM2()
    >>> result = hdm2.predict(
    ...     premise="The sky is blue due to Rayleigh scattering.",
    ...     hypothesis="The sky appears blue because of light scattering."
    ... )
    >>> print(result.score, result.label)
"""

import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import List, Literal, Tuple

import torch
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

from hdm2 import HallucinationDetectionModel

class HDM2Output(BaseModel):
    """Structured output from HDM-2 hallucination detection.

    Contains the overall factual consistency score, binary classification
    label, and sentence-level explanations identifying hallucinated content.

    Attributes:
        score: Factual consistency score from 0.0 to 1.0, where higher
            values indicate greater consistency with the source (inverse
            of hallucination severity).
        label: Binary classification label. 1 indicates factually
            consistent (score >= 0.5), 0 indicates hallucinated.
        reason: List of (sentence, probability) tuples identifying
            hallucinated sentences and their hallucination probabilities.
            Contains ("No hallucinated sentences detected.", 0.0) if
            no hallucinations are found.
    """

    score: float
    label: Literal[0, 1]
    reason: List[Tuple[str, float]]

class HDM2():
    """Wrapper class for the external HDM-2 hallucination detection model.

    Provides a consistent interface for hallucination detection that matches
    the HHEM model API, enabling side-by-side comparison in experiments.
    Wraps the external HallucinationDetectionModel from the hdm2 package.

    Attributes:
        prompt: Task prompt template used by HDM-2 for context. Contains
            summarization instructions with an {article} placeholder.
        classifier: Instance of the underlying HallucinationDetectionModel.

    Note:
        HDM-2 is developed by an external research group and is included
        for experimental comparison with HHEM models. It is not part of
        the Vectara HHEM family.
    """

    def __init__(self):
        """Initialize the HDM-2 model wrapper.

        Creates an instance of HallucinationDetectionModel from the hdm2
        package and configures the task prompt for summarization evaluation.
        """
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

    def __str__(self) -> str:
        """Return the model identifier string.

        Returns:
            The string "HDM-2" for logging and display purposes.
        """
        return "HDM-2"

    def predict(self, premise: str, hypothesis: str) -> HDM2Output:
        """Evaluate factual consistency between a source and summary.

        Runs HDM-2 inference to determine whether the hypothesis (summary)
        is factually consistent with the premise (source document). Returns
        a structured result with overall score, binary label, and sentence-
        level hallucination explanations.

        Args:
            premise: The source document or passage (ground truth).
            hypothesis: The generated summary to evaluate for hallucinations.

        Returns:
            HDM2Output containing:
                - score: Factual consistency score (1.0 - hallucination_severity)
                - label: 1 if score >= 0.5 (consistent), 0 otherwise
                - reason: List of hallucinated sentences with probabilities

        Note:
            Empty hypotheses are replaced with a placeholder string before
            evaluation. Model stdout/stderr is suppressed during inference.
        """
        if hypothesis == "":
            hypothesis = "EMPTY SUMMARY GIVEN BY MODEL"

        f = io.StringIO()

        with redirect_stdout(f), redirect_stderr(f):
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

        return HDM2Output(score=simple_score, label=pred, reason=hallucinated_sentences)

if __name__ == "__main__":

    test_data = ("The sky is blue", "The universe is blue")
    
    hdm_2 = HDM2()
    print (hdm_2.predict(*test_data))