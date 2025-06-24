from src.tests.AbstractTest import AbstractTest
import pandas as pd
from src.config import (
    TEST_JUDGEMENTS_DATA, TEST_RESULTS_DATA, TEST_SUMMARIES_DATA
)
from src.LLMs.AbstractLLM import MODEL_RETURNED_NON_STRING_TYPE_OUTPUT, SUMMARY_ERRORS
from src.data_struct.data_model import Stats, Summary, Judgement
from src.analytics.stats import (
    compute_confidence_interval, compute_hallucination_rate,
    compute_answer_rate, compute_avg_summary_length
)
from src.analytics.metrics import is_valid_summary
from src.utils.json_utils import load_json

class TestAnalytics(AbstractTest):
    """
    Tests the function in metrics.py and stats.py

    Attributes:
        summaries_df (pd.DataFrame): Test summaries data
        metrics_df (pd.DataFrame): Precalculated metrics for summaries data
        stats_anwers (Stats): Precaculatd stats for summaries data
    
    Methods:
        test_hallucination_rate()
        test_answer_rate()
        test_avg_summary_length()
        test_confidence_interval()
        test_valid_summary()
    """
    def __init__(self):
        self.summaries_df = pd.read_json(TEST_SUMMARIES_DATA, lines=True)
        self.metrics_df = pd.read_json(TEST_JUDGEMENTS_DATA, lines=True)
        self.stat_answers = pd.read_json(TEST_RESULTS_DATA, lines=True)

    def __str__(self):
        return "TestAnalytics"
        
    def run_tests(self):
        self.test_valid_summary()
        self.test_hallucination_rate()
        self.test_confidence_interval()
        self.test_answer_rate()
        self.test_avg_summary_length()

    def test_hallucination_rate(self):
        """
        Test if compute_hallucination_rate returns the correct value

        Args:
            None
        Returns:
            None
        """
        grouped_metric_df = self.metrics_df.groupby(Stats.Keys.DATE_CODE)

        for date_code, subset_df in grouped_metric_df:
            hr = round(compute_hallucination_rate(subset_df)*100.0, 1)
            assert hr == self.stat_answers[self.stat_answers[Stats.Keys.DATE_CODE] == date_code][Stats.Keys.HALLUCINATION_RATE].values[0]

    def test_answer_rate(self):
        """
        Test if compute_answer_rate returns the correct value

        Args:
            None
        Returns:
            None
        """
        grouped_metric_df = self.metrics_df.groupby(Stats.Keys.DATE_CODE)

        for date_code, subset_df in grouped_metric_df:
            ar = round(compute_answer_rate(subset_df)*100.0, 1)
            assert ar == self.stat_answers[self.stat_answers[Stats.Keys.DATE_CODE] == date_code][Stats.Keys.ANSWER_RATE].values[0]

    def test_avg_summary_length(self):
        """
        Test if compute_avg_summary_length returns the correct value

        Args:
            None
        Returns:
            None
        """
        grouped_metric_df = self.metrics_df.groupby(Stats.Keys.DATE_CODE)

        for date_code, subset_df in grouped_metric_df:
            asl = round(compute_avg_summary_length(subset_df), 1)
            assert asl == self.stat_answers[self.stat_answers[Stats.Keys.DATE_CODE] == date_code][Stats.Keys.AVG_SUMMARY_LENGTH].values[0]

    def test_confidence_interval(self):
        """
        Test if compute_confidence interval returns the correct value

        Args:
            None
        Returns:
            None
        """
        grouped_metric_df = self.metrics_df.groupby(Stats.Keys.DATE_CODE)

        for date_code, subset_df in grouped_metric_df:
            ci = round(compute_confidence_interval(subset_df)*100.0, 1)
            assert ci == self.stat_answers[self.stat_answers[Stats.Keys.DATE_CODE] == date_code][Stats.Keys.CONFIDENCE_INTERVAL].values[0]

    def test_valid_summary(self):
        """
        Test if is_valid_summary is working

        Args:
            None
        Returns:
            None
        """
        article_summaries = self.summaries_df[Summary.Keys.SUMMARY].tolist()
        true_valid_summ = self.metrics_df[Judgement.Keys.VALID].tolist()
        valid_summ_check = []
        for summary in article_summaries:
            valid_summ_check.append(is_valid_summary(summary))

        assert valid_summ_check == true_valid_summ