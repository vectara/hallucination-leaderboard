from src.tests.AbstractTest import AbstractTest
from src.LLMs.AbstractLLM import AbstractLLM
from dotenv import load_dotenv
import os
import csv

class TestLLM(AbstractTest):
    def __init__(self):
        super().__init__()
        load_dotenv()
        data_path = os.getenv("TEST_DATA")
        self.sample_article = None
        with open(data_path) as csvfile:
            reader = csv.DictReader(csvfile)
            first_row = next(reader)
            self.sample_article = first_row["text"]
        self.model = None

    def set_model(self, model: AbstractLLM):
        self.model = model

    def run_tests(self):
        self.test_summarize(self.sample_article)

    def test_summarize(self, test_article: str):
        msg = self.model.summarize(test_article)
        assert type(msg) == str