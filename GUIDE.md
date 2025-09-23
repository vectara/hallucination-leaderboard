# How to run and expand this code base

DO NOT PUSH judgments.jsonl to the repo. It will allow people to chat on the leaderboard.

TODO: 
1. Allow interrupting the program and resume from the same point.
2. Supporting selecting to run HHEM on CPU. 
3. Do we still need test files?
4. update the app code to read from new stats files or shall we have a script that pushes to HF datasets repo?
5. Add a script to generate leaderboard ranking markdown. 
6. Do we still need `model_fullname` ?

## Quick Guide

This section serves as a quick refresher for those familiar with the code base or want an abridged version.

Make sure you are in the correct environment, have the proper libraries installed, and the installation command has been run. Run always in root.

#### Add New Company and Models

Copy the example_company.py code and create a new file preferably with the name of the company. If there are unusual characters then the python file name can be whatever is a proper file name that still clearly refers to the company.

Within the python file reference and satisfy all TODO comments.

Once the new python file is complete update the model registry in `src/LLMs/__init__.py` with the new company objects you created in the company python file. Place the company alphabetically within the dictionary.

Lastly it's time for a test run. config.py contains various experimental setups including the test and live runs. Scan for the EvalConfig object with field eval_name assigned to "test". Then under per_LLM_configs add the new model using the companys config object you created in its respective company class alphabetically by company. Repeat the previous step for all the models you added.

The test run object also serves as an example for how to add models for other tests so leave them there as a reference and overall list of all possible models we can run. Comment out other company objects if that hasn't been done yet and do a test run with only the models you added by inputting the command. 

`hhem-leaderboard --eval_name test`

Inspect the console output for any issues. If it runs to completion then inspect the output_test directory for the saved summaries. We have two test cases. A null case and a real summary case. The first article you should see a real summary. The second case you should see "I cannot do it.". If both are satisfied then you successfully added the company and its models.


#### Add Models to Existing Company

Decide if the models execution mode is local(cpu/gpu) or api based.

Search for the company python file within `src/LLMs/`. Within the class find the `class COMPANY_NAMEConfig(BasicLLMConfig)` object and add the model to the field model_name. 

Next you should understand how your model should be ran to get a summary. When you do inspect the summarize method and at the respective execution modes conditional branch. If there is no case that matches how your model needs to be executed then you will need to add a new case with a new unique id number(Just go +1 from the last number shown). If there is a valid case then record the case number for later.

Find either the client_mode_group or local_mode_group dictionary. Add your model to the dictionary. If there existed a case that matched how your model should run then simply record that number. If there wasn't a case then use then one you should have made in the previous step.

Lastly it's time for a test run. config.py contains various experimental setups including the test and live runs. Scan for the EvalConfig object with field eval_name assigned to "test". Then under per_LLM_configs add the new model using the companys config object you created in its respective company class alphabetically by company. Repeat the previous step for all the models you added.

The test run object also serves as an example for how to add models for other tests so leave them there as a reference and overall list of all possible models we can run. Comment out other company objects if that hasn't been done yet and do a test run with only the models you added by inputting the command. 

`hhem-leaderboard --eval_name test`

Inspect the console output for any issues. If it runs to completion then inspect the output_test directory for the saved summaries. We have two test cases. A null case and a real summary case. The first article you should see a real summary. The second case you should see "I cannot do it.". If both are satisfied then you successfully added the company and its models.


#### Live LB Run

Similar to before we need to adjust the config.py file but this time adjust the EvalConfig object with field eval_name assigned to "live". Within live find the per_LLM_configs field and remove all other models currently there. Add the models you want to run, save the file, and run the following command.

`hhem-leaderboard --eval_name live`

Once complete run the following git commands

`git add .`
`git commit -a -m "adding {model_name-date_code}`
`git push`

The LB will automatically update the leaderboard once it see the file changes.

## Installation

Suppose you are in the root directory of the project.

```bash
pip install -e .
```

Then, you can use the command `hhem-leaderboard` to run the evaluation. See details below. 

## Files

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

## Classes 

Under `src/LLMs/`, for each LLM provider, there is a corresponding file (`src/LLMs/Anthropic.py`, `src/LLMs/OpenAI.py`, etc.) that contains the implementation of the LLM. Each provider has three classes in heritied from the `AbstractLLM`, `BasicLLMConfig`, and `BasicSummary` classes: 

| Class | Parent | Description |
|-------|--------|-------------|
| `{Provider_name}LLM` | `AbstractLLM` in `src/LLMs/AbstractLLM.py` | The class for all LLMs of the provider. Must have the `summarize`, `setup`, and `teardown` methods. |
| `{Provider_name}Config` | `BasicLLMConfig` in `src/data_model.py` | The parameters for using LLMs of the provider to summarize an article. Because the generation of summaries is provider-specific (e.g., different tags for thinking), their members differ while common ones such as `model_name`, `company`, `temperature` and `max_tokens` are inherited from `BasicLLMConfig`. |
| `{Provider_name}Summary` | `BasicSummary` in `src/data_model.py` | The class for summaries generated by LLMs of the provider. It usually contains provider-specific fields. |

## The configuration file

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

### Order of supersedes in LLM configs

There are many places that a user can specify the parameters of an LLM when it summarizes an article. The order of supersedes is determined by the following order (top to bottom): 

1. Those in `LLM_Configs` in an `EvalConfig` object in `src/config.py` -- specific for an LLM in an evaluation run. 
2. Those not in `LLM_Configs` in an `EvalConfig` object in `src/config.py` -- default for all LLMs in all evaluation runs.
3. The default values in `{Provider_name}Config` class.
4. The default values in `BasicLLMConfig` class.

## The pipeline

There are three steps in the pipeline defined in `src/pipeline/{summarize, judge, aggregate}.py`: 

1. `summarize`: Generate summaries of for all articles in the dataset specified in `source_article_path` in the `EvalConfig` object. The `source_article_path` is a path to a CSV file of four columns: `article_id`, `text`, `dataset`. See `datasets/test_articles.csv` for an example.
2. `judge`: Get the HHEM score along with the validity and word count of each summary produced by the LLMs specified in `LLM_Configs`.
3. `aggregate`: Aggregate the results to get the hallucination rate, average word count, and answer rate of each LLM specified in `LLM_Configs`.

You do not have to do all three steps together. You can put them in different `EvalConfig` objects and run them in different runs. 

## How to reproduced results

All previous settings are stored in `src/config.py`. To reproduced results, you can just run the command:

```bash
hhem-leaderboard --eval_name <eval_name>
```
where `<eval_name>` is the name of the evaluation you want to re-run. 

## How to contribute by adding a new LLM

### If the LLM is from a provider already supported

To support a new LLM from a provider already supported, just two steps: 

1. Expand the `{Provider_name}LLM` class in `src/LLMs/{Provider_name}.py` to support summarizing using the new LLM. 
2. Create a new entry in `src/config.py` under `eval_configs` with the new LLM.

### If the LLM is from a new provider

To support a new provider, 
1. Create a new file in `src/LLMs/` with the provider's name. Add at least the following classes: 
    - `{Provider_name}Config`: The configuration for the new LLM.
    - `{Provider_name}Summary`: The summary for the new LLM.
    - `{Provider_name}LLM`: The LLM for the new provider which must have the `summarize`, `setup`, and `teardown` methods.
2. Add the new provider to the `MODEL_REGISTRY` in `src/LLMs/__init__.py`.
3. Perform the steps for "If the LLM is from a provider that is already supported" above.