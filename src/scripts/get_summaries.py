from src.logging.Logger import logger
import pandas as pd
import inspect
import os
import json

from src.LLMs.OpenAI_GPTd4p1.GPTd4p1 import GPTd4p1

"""
Requests all LLMS to produce a summary. Summaries are only produced if no 
summary data is detected. Summaries can be forced to be regenerated. Stores the
summary data as a JSON file local to associated LLM class.

Functions:

"""

'''Need to do something about article IDs here'''
def run():
    logger.log("Starting to generate summaries")

    models = [GPTd4p1()]
    article_df = pd.read_csv(os.getenv("LB_DATA"))

    for model in models:
        model_name = model.get_name()

        logger.log(f"Generating summaries for {model_name}")

        obj_file_path = inspect.getfile(type(model))
        obj_dir = os.path.dirname(os.path.abspath(obj_file_path))
        json_file = f"summaries_{model_name}.json"
        json_path = os.path.join(obj_dir, json_file)

        if summaries_json_exists(json_path):
            logger.log("Summaries json file exists, moving to next model")
            continue
        else:
            logger.log("Summaries json file does not exist, generating...")
            generate_and_save_summaries(model, article_df, json_path)
    
    logger.log("Finished generating and saving summaries")


def generate_and_save_summaries(model, article_df, json_path):
    article_texts = article_df['text'].tolist()
    summaries = model.summarize_articles(article_texts)
    summary_records = create_summary_records(summaries, article_df)
    save_to_json(json_path, summary_records)


def save_to_json(json_path, summary_records):
    logger.log("Saving json file")
    with open(json_path, "w") as f:
        json.dump(summary_records, f, indent=4)

def create_summary_records(summaries, article_df):
    article_texts = article_df['text'].tolist()
    article_ids = article_df['article_id'].tolist()
    article_datasets = article_df['dataset'].tolist()
    model_summary_dict = [
        {"article_id": a_id, "summary": summ,
            "source_article": source, "dataset": ds}
        for a_id, summ, source, ds in zip(
            article_ids, summaries, article_texts, article_datasets
        )
    ]
    return model_summary_dict


def summaries_json_exists(full_path):
    """
    
    """
    if os.path.isfile(full_path):
        return True
    else:
        return False

if __name__ == "__main__":
    run()