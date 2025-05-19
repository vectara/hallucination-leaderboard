from src.logging.Logger import logger
import pandas as pd
import inspect
import os

from src.LLMs.OpenAI_GPTd4p1.GPTd4p1 import GPTd4p1

"""
Requests all LLMS to produce a summary. Summaries are only produced if no 
summary data is detected. Summaries can be forced to be regenerated. Stores the
summary data as a JSON file local to associated LLM class.

Functions:

"""

'''Need to do something about article IDs here'''
def run():
    models = [GPTd4p1]
    article_df = pd.read_csv(os.getenv("LB_DATA"))
    article_texts = article_df['text'].tolist()
    for model in models:
        summaries = []
        model_name = model.get_name()
        obj_file_path = inspect.getfile(type(model))
        obj_dir = os.path.dirname(os.path.abspath(obj_file_path))
        json_file = f"summaries_{model_name}.json"
        if summaries_json_exists(obj_dir, json_file):
            continue
        else:
            summaries = model.summarize_articles(article_texts)
            pass

def summaries_json_exists(obj_dir, json_file):
    full_path = os.path.join(obj_dir, json_file)
    if os.path.isfile(full_path):
        return True
    else:
        return False

if __name__ == "__main__":
    run()