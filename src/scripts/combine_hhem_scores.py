# File Not Currently Used
from src.logging.Logger import logger
import os
from tqdm import tqdm
from src.utils.json_utils import save_to_json, json_exists, load_json
from src.LLMs.AbstractLLM import AbstractLLM
"""
Combines HHEM scores for all LLMs that have HHEM scores into a single JSON file
usable for the app leaderboard.

Functions:
    run(models)

"""

def run(models: list[AbstractLLM]):
    """
    Loads hhem_scores_model.json files if it exists and adds it a larger 
    JSON file that includes scores for all models.

    Args:
        models (list[AbstractLLM]): models 
    
    Returns:
        None
    """
    logger.info("Starting combine HHEM scores")

    combined_hhem_scores = {}
    out_dir = os.getenv("OUTPUT_DIR")
    combined_file_path = f"{out_dir}/combined_hhem_scores.json"

    for model in tqdm(models):
        model_name = model.get_model_name()
        model_out_dir = model.get_model_out_dir()

        logger.info(f"Gathering {model_name} HHEM data")

        hhem_json_file = f"hhem_scores_{model_name}.json"
        hhem_json_path = os.path.join(model_out_dir, hhem_json_file)

        if json_exists(hhem_json_path):

            logger.info(f"HHEM score JSON found for {model_name}")

            json_data = load_json(hhem_json_path)
            if model_name not in combined_hhem_scores:
                combined_hhem_scores[model_name] = json_data
            else:
                logger.warning((
                    f"{model_name} was found to already have an entry, "
                    "skipping. This message occured likely because this model"
                    "is being processed twice or has the same name as "
                    "another model."
                ))
        else:
            logger.info(
                f"HHEM JSON score not found for {model_name}, skipping model"
            )
            continue
    logger.info("Saving combined HHEM scores")
    save_to_json(combined_file_path, combined_hhem_scores)
    logger.info("Finished combining and saving HHEM scores")

if __name__ == "__main__":
    pass