# How to run and expand this code base

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

#### Installation

Before you run you need to import all necessary packages. Start in the root directory.

Order here matters, make sure you do these in sequence

```bash
pip install -r requirements.txt
```

Then

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

Copy the __example_company.py code and create a new file preferably with the name of the company. If there are unusual characters then the python file name can be whatever is a proper file name that still clearly refers to the company.

Within the python file reference and satisfy all TODO comments. Some TODO comments may not need to be satisfied in which case remove the TODO comment and leave as is. Company name within the file 
should replicate its official name on huggingface or as close to it as possible.

Once the new python file is complete update the model registry in `src/LLMs/__init__.py` with the new company objects you created in the company python file. Place the company alphabetically within the dictionary.

Lastly import the companies config object(e.g. VectaraConfig) into `src/config.py`. Place the object alphabetically.

#### Add Models to Existing Company

Decide if the models execution mode is local(cpu/gpu) or api based.

Search for the company python file within `src/LLMs/`. Within the class find the `class COMPANY_NAMEConfig(BasicLLMConfig)` object and add the model to the field model_name. 

Next you should understand how your model should be ran to get a summary. When you do inspect the summarize method and at the respective execution modes conditional branch. If there is no case that matches how your model needs to be executed then you will need to add a new case with a new unique id number(Just go +1 from the last number shown). If there is a valid case then record the case number for later.

Find either the client_mode_group or local_mode_group dictionary. Add your model to the dictionary. If there existed a case that matched how your model should run then simply record that number. If there wasn't a case then use then one you should have made in the previous step.

#### Testing your model

Before doing a live run its recommended to test your model to make sure it works and the output doesn't have any troublesome artifacts. config.py contains various experimental setups including the test and live runs. Scan for the EvalConfig object with field eval_name assigned to "test"(Should be the first entry in the list). Then under per_LLM_configs add the new model to the list of model configs alphabetically by company. Make sure other models are commented out before testing your own model.

The test experiment also serves as an example on how to use your model for other experiments so when you're finished just comment out your new addition. 

To begin the test, run the following command `hhem-leaderboard --eval_name test`

Inspect ouput_test to find your new models output. 

Test case 1 is a red flag if it fails. It is expected to have a response.

Make sure the response only contains summary text. If there are extra artifacts around the summary add additional code to handle their removal. Then run the test again to verify.

Test case 2 is a yellow flag if it fails. Should either say "I am unable to summarize this passage." or some output related to it not having anything to summarize. Failure of this case hasn't happened but if it did happen may indicate the model is weak.

Test case 3 is a yellow flag if failed. Tests if the model can handle the largest context it can expect. Failure of this case may indicate a sub optimal answer rate if this case is failed.

Make sure it only contains summary text, if there are extra artifacts around the summary add additional code to handle their removal. Then run the test again to verify.

#### Live LB Run

Similar to before we need to adjust the config.py file but this time adjust the EvalConfig object with field eval_name assigned to "live"(Should be the second list entry). Within live find the per_LLM_configs field and remove all other models currently there. Add the models you want to run, save the file, and run the following command.

`hhem-leaderboard --eval_name live`

Once complete run the following git commands

`git add .`
`git commit -a -m "adding {model_name-date_code}`
`git push`

Github actions will automatically update both the README and the plot, make sure to pull those changes.

This process can be done manually by `update_readme.py` scripted located in the root directory.



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