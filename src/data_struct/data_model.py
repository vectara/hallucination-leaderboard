from pydantic import BaseModel

# The data model is redundant but hopefully it will make things in the future easier

# All summaries evaluated together should have the same timestamp in `summaries.json` and `judgements.json`

# A `summaries.json` file is a list of Summary objects
class Summary(BaseModel):
    timestamp: str
    llm: str
    article_id: int
    summary: str

# A `judgements.json` file is a list of Judgement objects
class Judgement(BaseModel):
    timestamp: str
    article_id: int
    hhem_version: str
    hhem_score: float
    valid: bool
    summary_words: int

# A `stats.json` file is a  Stats objects, aggregated from a list of Judgement objects
class Stats(BaseModel):
    timestamp: str
    llm: str
    hallucination_rate: float
    answer_rate: float
    summary_words: int

    
    
