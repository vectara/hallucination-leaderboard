from pydantic import BaseModel

# The data model is redundant but hopefully it will make things in the future easier

# All summaries evaluated together should have the same timestamp in `summaries.json` and `judgements.json`

# A `summaries.json` file is a list of Summary objects
class Summary(BaseModel):
    timestamp: str
    llm: str
    article_id: int
    summary: str

    class Keys:
        TIMESTAMP = "timestamp"
        LLM = "llm"
        ARTICLE_ID = "article_id"
        SUMMARY = "summary"

# A `judgements.json` file is a list of Judgement objects
class Judgement(BaseModel):
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

# A `stats.json` file is a  Stats objects, aggregated from a list of Judgement objects
class Stats(BaseModel):
    timestamp: str
    llm: str
    hallucination_rate: float
    answer_rate: float
    avg_summary_length: float

    class Keys:
        TIMESTAMP = "timestamp"
        LLM = "llm"
        HALLUCINATION_RATE = "hallucination_rate"
        ANSWER_RATE = "answer_rate"
        AVG_SUMMARY_LENGTH = "avg_summary_length"

    
    
