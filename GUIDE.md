# How to run and expand this code base

## Overview

The HHEM (Hughes Hallucination Evaluation Model) Leaderboard is an evaluation system that measures hallucination rates across different LLMs. It works by:

1. **Summarization**: Each LLM generates summaries of source articles from a standardized dataset.
2. **Judgment**: HHEM scores each summary for factual consistency against the source article (0-1 scale, higher = more consistent).
3. **Aggregation**: Results are compiled into statistics (hallucination rate, answer rate, avg word count) per model.

The codebase is organized around **providers** (companies like OpenAI, Anthropic, etc.), each with their own LLM implementation file. Adding a new model means either adding to an existing provider or creating a new provider file.

## The Basics

This section serves as an introduction to using the basic and critical features of the leaderboard. Largely to add a model, test it, and then run the experiment associated with results on the public leaderboard.

#### Files

```plaintext
src/
├── config.py
├── data_model.py
├── LLMs/
│   ├── AbstractLLM.py
│   ├── Anthropic.py
│   ├── OpenAI.py
│   ├── ...
├── pipeline/
│   ├── summarize.py
│   ├── judge.py
│   ├── aggregate.py
├── main.py
├── ...
```

#### Initial Preparation

This step does not need to be followed exactly but helps simplify later on for repeated building of the engine.

- Optional
```bash
mkdir LB
cd LB
python3 -m venv lb_env
source lb_env/bin/activate
```

- Required
```bash
git clone https://github.com/vectara/hallucination-leaderboard.git
cd hallucination-leaderboard
git checkout lb-engine
```

**Branch workflow:** The `lb-engine` branch is the main development branch for running evaluations. For adding new models or making changes, you can either work directly on `lb-engine` or create a feature branch from it (e.g., `git checkout -b add-new-model`). Merge back to `lb-engine` when complete.

You will need the .env file containing all the keys, likely this is already shared with you. This is placed in the project root.

You will need the leadersboard_dataset_v2.csv, likely this is already shared with you. This is placed inside the datasets directory.

#### Installation

Before you run you need to install all necessary packages. Start in the root directory.

Order here matters, make sure you do these in sequence:

```bash
pip install -r requirements.txt
```

**Optional:** If you need to run models locally on GPU (not just API calls), also install:

```bash
pip install -r requirements-gpu.txt
```

Then install the package itself:

```bash
pip install -e .
```
**WARNING**

It's possible requirements.txt will be updated in the future and include the custom leaderboard package. This will cause an error when you try to install requirements.txt. In this case the following need to be removed from requirements.txt before beginning the installation sequence.

`-e git+https://github.com/vectara/hallucination-leaderboard@c2ef203e475b8a5a3d39724f8f8518a5734b1e66#egg=hhem_leaderboard`

#### Huggingface login

Currently this is not an issue when using HHEM API which is standard practice now but this is a note for later if you do end up using HHEM on local GPU. Skip this section if you aren't

If you are having trouble running HHEM on GPU its likely because you haven't linked your huggingface account.

`huggingface-cli login`

Follow the instructions to login.

Should work fine after you succesfully login.

If you are still running issues then it's likely you dont have permissions or access to the HHEM repo on huggingface. Go find someone that can give you access.


#### Add New Company and Models

**Quick Overview — Files to modify:**
- [ ] `src/LLMs/{company}.py` — new provider file
- [ ] `src/LLMs/__init__.py` — add import + registry entry
- [ ] `src/config.py` — add config import + test config entry

Copy the __example_company.py code and create a new file preferably with the name of the company. If there are unusual characters, then the python file name can be whatever is a proper file name that still clearly refers to the company.

Within the python file reference and satisfy all TODO comments. Some TODO comments may not need to be satisfied in which case remove the TODO comment and leave as is. Company name within the file
should replicate its official name on huggingface or as close to it as possible.

If the company's models are accessed through multiple backend providers (e.g., Together AI, Fireworks, HuggingFace), configure the `api_type` field. See the `api_type` TODOs in the template. The rule is:
- Use `"default"` only if the company has their own native API
- If models are only available through third-party providers, list those explicitly with no default value

Once the new python file is complete, update `src/LLMs/__init__.py` with two changes:
1. Add an import statement for your new classes (LLM, Config, Summary) at the top of the file, placed alphabetically among the other imports.
2. Add an entry to the `MODEL_REGISTRY` dictionary with your company's classes, placed alphabetically within the dictionary.

Lastly import the companies config object(e.g. VectaraConfig) into `src/config.py`. Place the object alphabetically.

#### Add Models to Existing Company

Decide if the models execution mode is local(cpu/gpu) or api based.

Search for the company python file within `src/LLMs/`. Within the class find the `class COMPANY_NAMEConfig(BasicLLMConfig)` object and add the model to the field model_name. 

Next you should understand how your model should be ran to get a summary. When you do, inspect the summarize method and at the respective execution modes conditional branch. If there is no case that matches how your model needs to be executed then you will need to add a new case with a new and descriptive enum.

Find either the client_mode_group(You're using an API) or local_mode_group(You are running the model locally) dictionary. Add your model to the dictionary. If there existed a case that matched how your model should run, then simply record that enum value. If there wasn't a case then use then one you should have made in the previous step.

#### Testing your model

Before doing a live run its recommended to test your model to make sure it works and the output doesn't have any troublesome artifacts. config.py contains various experimental setups including the test and live runs. Scan for the EvalConfig object with field eval_name assigned to "test"(Should be the first entry in the list). Then under per_LLM_configs add the new model to the list of model configs alphabetically by company.

**Important:** Before running the test, verify that your new model config is the ONLY uncommented entry in the `per_LLM_configs` list. Comment out any other active configs to avoid running unnecessary evaluations.

**Note on model names:** Model names are case-sensitive and must match exactly as defined in the provider's config class. If you use an incorrect model name, the error message will suggest similar valid names to help you find the correct spelling.

The test experiment also serves as an example on how to use your model for other experiments so when you're finished just comment out your new addition. 

To begin the test, run the following command `hhem-leaderboard --eval_name test`

Inspect output_test to find your new model's output. 

Test case 1 is a red flag if it fails. It is expected to have a response.

Make sure the response only contains summary text. If there are extra artifacts around the summary add additional code to handle their removal. Then run the test again to verify.

Test case 2 is simply informative of behavior. Should either say "I am unable to summarize this passage." or some output related to it not having anything to summarize. Failure of this case is rare and may indicate the model has issue with small texts or weak reasoning to this edge case.

Test case 3 is a yellow flag if failed. Tests if the model can handle the largest context it can expect. Failure of this case may indicate a sub optimal answer rate.

#### Live LB Run

Similar to before we need to adjust the config.py file but this time adjust the EvalConfig object with field eval_name assigned to "live" (Should be the second list entry). Within "live" find the per_LLM_configs field and remove all other models currently there. Add the models you want to run, save the file, and run the following command.

`hhem-leaderboard --eval_name live`

Once complete run the following git commands

`git add .`
`git commit -a -m "adding {model_name-date_code}`
`git push`

Github actions will automatically update both the README and the plot, make sure to pull those changes.

This process can be done manually by `update_readme.py` scripted located in the root directory.

Once a model is completed succesfully move it below the completed models comment so we have an internal record of the models we've run so far and their exact settings.

## Advanced Info

#### Classes 

Under `src/LLMs/`, for each LLM provider, there is a corresponding file (`src/LLMs/Anthropic.py`, `src/LLMs/OpenAI.py`, etc.) that contains the implementation of the LLM. Each provider has three classes in heritied from the `AbstractLLM`, `BasicLLMConfig`, and `BasicSummary` classes: 

| Class | Parent | Description |
|-------|--------|-------------|
| `{Provider_name}LLM` | `AbstractLLM` in `src/LLMs/AbstractLLM.py` | The class for all LLMs of the provider. Must have the `summarize`, `setup`, and `teardown` methods. |
| `{Provider_name}Config` | `BasicLLMConfig` in `src/data_model.py` | The parameters for using LLMs of the provider to summarize an article. Because the generation of summaries is provider-specific (e.g., different tags for thinking), their members differ while common ones such as `model_name`, `company`, `temperature` and `max_tokens` are inherited from `BasicLLMConfig`. |
| `{Provider_name}Summary` | `BasicSummary` in `src/data_model.py` | The class for summaries generated by LLMs of the provider. It usually contains provider-specific fields. |

#### The configuration file

All setting of evaluations are stored in `src/config.py` which contains one variable `eval_configs` that is a list of `EvalConfig` (defined in `src/data_model.py`) objects. Briefly, an `EvalConfig` object includes but is not limited to the following fields:

- `eval_name`: The name of the evaluation.
- `eval_date`: The date of the evaluation.
- `hhem_version`: The version of HHEM to use.
- `pipeline`: Elements of the evaluation pipeline, which is a list of strings that can only take values from the set `{"summarize", "judge", "aggregate"}`. (Default: `['summarize', 'judge', 'aggregate']` all three steps)
- `overwrite_summaries`: Whether to overwrite the existing summaries. When `True`, current behavior is that summaries that match the model name, date code, and summary date will be removed. (Default: `False`)
- `source_article_path`: The file path to the source articles to be summarized by LLMs under the evaluation. (Default: `datasets/test_articles.csv` which is the test data.)
- `common_LLM_config`: Summarization configurations for all LLMs in this evaluation. (Default: `BasicLLMConfig` with default values)
- `per_LLM_configs`: LLMs covered in this evaluation, each of which is an `{Provider_name}Config` object of the corresponding `{Provider_name}Config` class.

A `BasicLLMConfig` object includes but is not limited to the following fields (not all are required):

- `model_name`: The name of the LLM. 
- `company`: The company that provides the LLM.
- `date_code`: The date code of the LLM. (Default: `None`)
- `prompt`: The prompt to be used for summarization. (Default: a default prompt defined in `src/data_model.py`)
- `temperature`: The temperature to be used for summarization. (Default: `0.0`)
- `max_tokens`: The maximum number of tokens to be used for summarization. (Default: `4096`)
- `min_throttle_time`: The minimum time to wait between requests. (Default: `0.1`)
- `thinking_tokens`: The number of tokens allocated for thinking. (Default: `None`)
- `execution_mode`: The execution mode of the LLM. (Default: `None`)
- `api_type`: The backend API provider to use. Use `"default"` for the company's native API. If only third-party providers are available, use explicit provider names like `"together"`, `"fireworks"`, `"huggingface"`. (Default: varies by provider)

#### Order of supersedes in LLM configs

There are many places that a user can specify the parameters of an LLM when it summarizes an article. The order of supersedes is determined by the following order (top to bottom): 

1. Those in `LLM_Configs` in an `EvalConfig` object in `src/config.py` -- specific for an LLM in an evaluation run. 
2. Those not in `LLM_Configs` in an `EvalConfig` object in `src/config.py` -- default for all LLMs in all evaluation runs.
3. The default values in `{Provider_name}Config` class.
4. The default values in `BasicLLMConfig` class.

#### The pipeline

There are three steps in the pipeline defined in `src/pipeline/{summarize, judge, aggregate}.py`: 

1. `summarize`: Generate summaries of for all articles in the dataset specified in `source_article_path` in the `EvalConfig` object. The `source_article_path` is a path to a CSV file of four columns: `article_id`, `text`, `dataset`. See `datasets/test_articles.csv` for an example.
2. `judge`: Get the HHEM score along with the validity and word count of each summary produced by the LLMs specified in `LLM_Configs`.
3. `aggregate`: Aggregate the results to get the hallucination rate, average word count, and answer rate of each LLM specified in `LLM_Configs`.

You do not have to do all three steps together. You can put them in different `EvalConfig` objects and run them in different runs. 