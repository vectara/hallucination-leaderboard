# Running the Program

DO NOT PUSH judgements.jsonl to the repo. It will allow people to chat on the leaderboard.

TODO: 
1. Allow interrupting the program and resume from the same point.
2. Supporting selecting to run HHEM on CPU. 

### Configuration
Before running the program you need to define CONFIG in config.py to include the settings and models you want.


#### File Location
```plaintext
src/
└── config.py
```

#### File

Temperature and max_tokens should stay the default of 0.0 and 1024. If a model for whatever reason cannot use the defaults then define the individual model temperature and max_token fields to be as close as possible. If a model does not allow you to adjust it then record the value the company uses for the model.

```python
from src.config_model import ExecutionMode, InteractionMode
from src.constants import (
    GET_SUMM, GET_JUDGE, GET_RESULTS,
    TEST_DATA_PATH, LB_DATA_PATH
)

# Runtime config
# ADJUST THIS GLOBAL VARIABLE
CONFIG = {
    # Defines protocols to perform, done in seqence. Not all sequences are valid
    "pipeline": [GET_SUMM, GET_JUDGE, GET_RESULTS],
    # Specifies if summary jsonl should be overwritten. Judgemennts and results are always overwritten
    "overwrite": True,
    # Specifies the article dataset to generate summaries for
    "input_file": LB_DATA_PATH,
    # Sets the temperature for all models to this value if it can be set
    "temperature": 0.0, 
    # Sets the max_tokens for all models to this value if it can be set
    "max_tokens": 1024,
    # NO EFFECT ON PROGRAM ATM. Necessary for confidence intervals. Specifies how many times to repeat the summary generation process for all articles.
    "simulation_count": 1,
    # NO EFFECT ON PROGRAM ATM. Necessary for confidence intervals. Samples a subset of the summaries for all simulations. Should be strictly less than simulation count
    "sample_count": 1,
    # List LLMs to evaluate
    "LLMs_to_eval":
    [
        {
            # Company model is from
            "company": "anthropic",
            "params": {
                # model_name: Company defined model name
                "model_name": "claude-opus-4",
                # date_code: Optional, defaults to "". Company defined date code
                "date_code": "20250514",
                # interaction_mode: doesn't directly change functionality, but is useful for record keeping, as it can influence output behavior.
                # Must be either InteractionMode.CHAT or InteractionMode.COMPLETION
                # InteractionMode.CHAT indicates that the model is responding to the summarization request in a chat-based format
                # InteractionMode.COMPLETION indicates that the model is completing the summarization request as a single prompt completion
                "interaction_mode": InteractionMode.CHAT
                # execution_mode: indicates if this model is expected to run using an API or locally. Can be only ExecutionMode.CLIENT or ExecutionMode.LOCAL
                # ExecutionMode.CLIENT indicates an API is being used for summary request
                # ExectionMode.LOCAL indicates the model is being hosted and ran on the same machine as the program
                "execution_mode": ExecutionMode.CLIENT
                # temperature: Optional defaults to global config value.
                "temperature": 0.0,
                # max_tokens: Optional, defaults to global config value
                "max_tokens": 1024,
                # thinking_tokens: Optionals, defaults to 0. Should always be set to 0 but if a model has to have thinking tokens set it to the minimal number required
                "thinking_tokens": 0,
                # min_throttle_time: Optional, defaults to 0.0 seconds. This is the minimum amount of time the summary request process must take before it moves on to the next. If it finishes before that time the program will halt until the required time has passed.
                "min_throttle_time": 0.0
            }
        }
        ,
        {
            "company": "openai",
            "params": {
                "model_name": "gpt-4.1",
            }
        }
    ]
}
```

#### Configuration superseding order

1. LLM-agnostic configuration in `config.py`
2. LLM-specific configuration in `config.py` under `LLM_Configs`
3. Default values in individual LLM classes

### Program

Main method of using the program is

```bash
python3 main.py
```

This will start the program and run with the settings and models specified by CONFIG within config.py.

Optionally CLI instructions can run the program,  For example below does the same as the sample CONFIG. Runs the protocols in sequence and overwrites previous summary data.

```bash
python3 main.py get_summ_judge_results --overwrite
```

Using the CLI will always supercede config pipeline and overwrite settings but still use the same models and defined parameters.

Use the following command for more information on how to use the CLI and general program information

```bash
python3 main.py --help
```

#### Valid Pipelines
Below is all possible pipelines when using the program

Config Approach
```python
# Given a dataset, generates summaries and saves to summaries.jsonl
[GET_SUMM],
# Given a summary.jsonl generates judgements/metrics and saves to judgements.jsonl
[GET_JUDGE],
# Given a judgement.jsonl generates stats for each unique date_code present and saves to stats.jsonl
[GET_RESULTS],
# Performs GET_SUMM then GET_JUDGE
[GET_SUMM, GET_JUDGE],
# Performs GET_JUDGE then GET_RESULTS 
[GET_JUDGE, GET_RESULTS],
# Performs GET_SUMM > GET_JUDGE >  GET_RESULTS
[GET_SUMM, GET_JUDGE, GET_RESULTS],
```

CLI Approach
```bash
python3 main.py get_summaries
python3 main.py get_judgements
python3 main.py get_results
python3 main.py get_summ_judge
python3 main.py get_judge_results
python3 main.py get_summ_judge_results
```

### Output File Structure

```plaintext
output/
└── company/
    └── model_name/
        ├── judgments.jsonl
        ├── stats.json
        └── summaries.jsonl
```

# Adding New Models

It's assumed the model you are adding is NOT operating in thinking mode. If there isn't a way to use the model without thinking the model set the think_tokens parameter to the minimum value.

### New Company

If the model you want to run does not have a company.py file for it a new file needs to be made. 

```py
from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY, MODEL_REGISTRY
import os
from src.config_model import ExecutionMode, InteractionMode
from src.exceptions import (
    ClientOrLocalNotInitializedError,
    ClientModelProtocolBranchNotFound,
    LocalModelProtocolBranchNotFound
)

# ADD IMPORTS IF NEEDED

COMPANY = "company_name"
class Company(AbstractLLM):
    """
    Class for models from company_name

    Attributes:
    """

    client_models = ["api_model_name"]
    local_models = ["local_model_name"]

    model_category1 = ["api_model_name"]
    model_category2 = ["local_model_name"]

    def __init__(
            self,
            model_name: str,
            execution_mode: ExecutionMode,
            interaction_mode: InteractionMode,
            date_code: str,
            temperature: float,
            max_tokens: int,
            thinking_tokens: int,
            min_throttle_time: float
    ):
        super().__init__(
            model_name,
            execution_mode,
            interaction_mode,
            date_code,
            temperature,
            max_tokens,
            thinking_tokens,
            min_throttle_time,
            company=COMPANY
        )
        self.model = self.get_model_identifier(model_name, date_code)

    def summarize(self, prepared_text: str) -> str:
        summary = EMPTY_SUMMARY
        if self.client_is_defined():
            if self.model_name in self.model_category1: # Set if False: if only using local
                # ADD NEW CODE HERE
                summary = # SUMMARY AS TYPE STRING FROM MODEL
            else:
                raise ClientModelProtocolBranchNotFound(self.model_name)
        elif self.local_model_is_defined():
            if self.model_name in self.model_category2: # Set if False: if only using client
                # ADD NEW CODE HERE
                summary = # SUMMARY AS TYPE STRING FROM MODEL
            else:
                raise LocalModelProtocolBranchNotFound(self.model_name)
        else:
            raise ClientOrLocalNotInitializedError(self.model_name)
        return summary

    def setup(self):
        if self.valid_client_model():
            api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
            self.client = # ADD NEW CODE HERE
        elif self.valid_local_model():
            self.local_model = # ADD NEW CODE HERE

    def teardown(self):
        if self.client_is_defined():
            self.close_client() # close_client() is an abstract method if no need to close just define as do nothing method
        elif self.local_model_is_defined():
            self.default_local_model_teardown()

MODEL_REGISTRY[COMPANY] = Company

```
Add your new model to the global CONFIG variable and run the test script to verify that the model does indeed return a string.

```bash
python3 test_script.py
```

### New Model to Existing Company

If the company.py file already exists you need to add code with the correct protocol for the model. If the protocol matches the protocol for another model, add it to the same list as the matching model. If it doesn't exist create a new list with the new model and then define its protocol under a new conditional branch in summarize.

```python
from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY, MODEL_REGISTRY
import os
from src.config_model import ExecutionMode, InteractionMode
from src.exceptions import (
    ClientOrLocalNotInitializedError,
    ClientModelProtocolBranchNotFound,
    LocalModelProtocolBranchNotFound
)

# ADD IMPORTS IF NEEDED

COMPANY = "company_name"
class Company(AbstractLLM):
    """
    Class for models from company_name

    Attributes:
    """

    client_models = ["api_model_name", "new_api_model_name2", "new_api_model_name3"]
    local_models = ["local_model_name"]

    model_category1 = ["api_model_name", "new_api_model_name2"]
    model_category2 = ["local_model_name"]
    model_category3 = ["new_api_model_name3"]

    def __init__(
            self,
            model_name: str,
            execution_mode: ExecutionMode,
            interaction_mode: InteractionMode,
            date_code: str,
            temperature: float,
            max_tokens: int,
            thinking_tokens: int,
            min_throttle_time: float
    ):
        super().__init__(
            model_name,
            execution_mode,
            interaction_mode,
            date_code,
            temperature,
            max_tokens,
            thinking_tokens,
            min_throttle_time,
            company=COMPANY
        )
        self.model = self.get_model_identifier(model_name, date_code)

    def summarize(self, prepared_text: str) -> str:
        summary = EMPTY_SUMMARY
        if self.client_is_defined():
            if self.model_name in self.model_category1:
                # Protocol to run models in category 1
                summary = # SUMMARY AS TYPE STRING FROM MODEL
            elif self.model_name in self.model_category3:
                # ADD NEW CODE HERE
                summary = # SUMMARY AS TYPE STRING FROM MODEL
            else:
                raise ClientModelProtocolBranchNotFound(self.model_name)
        elif self.local_model_is_defined():
            if self.model_name in self.model_category2:
                # Protocol to run models in category 2
                summary = # SUMMARY AS TYPE STRING FROM MODEL
            else:
                raise LocalModelProtocolBranchNotFound(self.model_name)
        else:
            raise ClientOrLocalNotInitializedError(self.model_name)
        return summary

    def setup(self):
        if self.valid_client_model():
            api_key = os.getenv(f"{COMPANY.upper()}_API_KEY")
            self.client = # Client
        elif self.valid_local_model():
            self.local_model = # Local Model

    def teardown(self):
        if self.client_is_defined():
            self.close_client()
        elif self.local_model_is_defined():
            self.default_local_model_teardown()

MODEL_REGISTRY[COMPANY] = Company

```

Same as before add your new model to the global CONFIG variable and run the test script to verify that the model does indeed return a string.

```bash
python3 test_script.py
```
