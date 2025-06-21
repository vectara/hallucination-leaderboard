from pydantic import BaseModel
from typing import Optional

class SourceArticle(BaseModel):
    """
    Representation of an Article record from LB dataset

    Fields
        article_id (int): unique id of article
        text (str): content text of article
        dataset (str): dataset that article belongs to (TODO: VERIFY TRUE)
    """
    article_id: int
    text: str
    dataset: str

    class Keys:
        ARTICLE_ID = "article_id"
        TEXT = "text"
        DATASET = "dataset"

class Summary(BaseModel):
    #TODO: Doc
    """
    Representation of a Summary of an Article
    
    Fields:
        timestamp (str): Date summary was produced
        llm (str): unique llm identifier, matches the label the respective
            company gave it
        article_id (int): unique id of article
        summary (str): llm generated summary of the text associated to article_id
    """
    timestamp: str
    llm: str
    date_code: str
    article_id: int
    summary: str
    summary_uid: str

    class Keys:
        TIMESTAMP = "timestamp"
        LLM = "llm"
        DATE_CODE = "date_code"
        ARTICLE_ID = "article_id"
        SUMMARY = "summary"
        SUMMARY_UID = "summary_uid"

class Judgement(BaseModel):
    """
    Representation of Judgements/Metrics for the Summary of an Article

    Fields:
        timestamp (str): date the metrics were performed
        article_id (int): id of the article summarized
        hhem_version (str): version of hhem applied for hhem score
        hhem_score (float): Hughes Hallucination Evaluation Metric (HHEM)
        valid (bool): Validity of the summary, defined in is_valid_summary
        summary_words (int): word count of summary
    """
    timestamp: str
    article_id: int
    hhem_version: str
    hhem_score: float
    valid: bool
    summary_words: int

    class Keys:
        TIMESTAMP = "timestamp"
        ARTICLE_ID = "article_id"
        HHEM_VERSION = "hhem_version"
        HHEM_SCORE = "hhem_score"
        VALID = "valid"
        SUMMARY_WORDS = "summary_words"

class Stats(BaseModel):
    """
    Representation of Stats for the Summaries of the Article Dataset. These are
    aggregated Judgements/Metrics

    Fields:
        timestamp (str): date the stats were performed
        llm (str): llm that performed the summarization
        hallucination_rate (float): hallucination rate on all summaries
        confidence_interval (float): variation in the hallucination rate
        answer_rate (float): For all summaries what percentage of them were
            valid
        avg_summary_length (float): Average summary length for all valid
            summaries
    """
    timestamp: str
    llm: str
    hallucination_rate: float
    confidence_interval: float
    answer_rate: float
    avg_summary_length: float

    class Keys:
        TIMESTAMP = "timestamp"
        LLM = "llm"
        HALLUCINATION_RATE = "hallucination_rate"
        CONFIDENCE_INTERVAL = "confidence_interval"
        ANSWER_RATE = "answer_rate"
        AVG_SUMMARY_LENGTH = "avg_summary_length"

    
    
