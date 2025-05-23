from src.tests.AbstractTest import AbstractTest
from src.LLMs.AbstractLLM import AbstractLLM
import os
import csv

class TestLLM(AbstractTest):
    """
    Test critical functions that all LLMs must be able to do in order to recieve
    HHEM results

    Attributes:
        sample_article (str): Example article for testing
        model (AbstractLLM): model being tested

    Methods:
        run_tests(): Runs all tests
        test_summarize(test_article): test the summarize functionality of model
        set_model(model): setter
    """
    def __init__(self):
        super().__init__()
        data_path = os.getenv("TEST_DATA")
        self.sample_article = None
        with open(data_path) as csvfile:
            reader = csv.DictReader(csvfile)
            first_row = next(reader)
            self.sample_article = first_row["text"]
        self.model = None

    def run_tests(self):
        """
        Tests critical functionality of LLMs. Currently this is just the ability
        to summarize a document

        Args:
            None
        
        Returns:
            None
        """
        self.test_summarize(self.sample_article)

    def test_summarize(self, test_article: str):
        """
        Tests to see if a model can take in a string and output a string.

        Args:
            test_article (str): Article to be summarized by model
        
        Returns:
            None
        """
        msg = self.model.summarize(test_article)
        assert type(msg) == str

    def set_model(self, model: AbstractLLM):
        """
        Sets current model

        Args:
            model (AbstractLLM): model to be tested

        Returns:
            None
        """
        self.model = model

