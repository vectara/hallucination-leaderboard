from src.tests.AbstractTest import AbstractTest
import pandas as pd
from src.config import TEST_JUDGEMENTS_DATA, TEST_RESULTS_DATA, TEST_SUMMARIES_DATA
from src.data_struct.data_model import Stats, Summary, Judgement
from src.analytics.stats import (
    compute_confidence_interval, compute_hallucination_rate,
    compute_answer_rate, compute_avg_summary_length
)
from src.analytics.metrics import is_valid_summary
from src.utils.json_utils import load_json

class TestAnalytics(AbstractTest):
    def __init__(self):
        self.summaries_df = pd.read_json(TEST_SUMMARIES_DATA, lines=True)
        self.metrics_df = pd.read_json(TEST_JUDGEMENTS_DATA, lines=True)
        self.stat_answers = Stats.model_validate(load_json(TEST_RESULTS_DATA))

    def __str__(self):
        return "TestAnalytics"
        
    def run_tests(self):
        self.test_valid_summary()
        self.test_hallucination_rate()
        self.test_confidence_interval()
        self.test_answer_rate()
        self.test_avg_summary_length()

    def test_hallucination_rate(self):
        #TODO: Doc
        hr = round(compute_hallucination_rate(self.metrics_df)*100.0, 1)
        assert hr == self.stat_answers.hallucination_rate

    def test_answer_rate(self):
        #TODO: Doc
        ar = round(compute_answer_rate(self.metrics_df)*100.0, 1)
        assert ar == self.stat_answers.answer_rate

    def test_avg_summary_length(self):
        #TODO: Doc
        asl = round(compute_avg_summary_length(self.metrics_df), 1)
        assert asl == self.stat_answers.avg_summary_length

    def test_confidence_interval(self):
        #TODO: Doc
        ci = round(compute_confidence_interval(self.metrics_df)*100.0, 1)
        assert ci == self.stat_answers.confidence_interval

    def test_valid_summary(self):
        #TODO: Doc
        article_summaries = self.summaries_df[Summary.Keys.SUMMARY].tolist()
        true_valid_summ = self.metrics_df[Judgement.Keys.VALID].tolist()
        valid_summ_check = []
        for summary in article_summaries:
            valid_summ_check.append(is_valid_summary(summary))

        assert valid_summ_check == true_valid_summ