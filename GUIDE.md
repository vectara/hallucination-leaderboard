# How to run and expand this code base

DO NOT PUSH judgments.jsonl to the repo. It will allow people to chat on the leaderboard.

TODO: 
1. Allow interrupting the program and resume from the same point.
2. Supporting selecting to run HHEM on CPU. 
3. Do we still need test files?
4. update the app code to read from new stats files or shall we have a script that pushes to HF datasets repo?

## How to reproduced

## HOw to support new LLM from a provider already supported

## How to support new provider

## Usage 

1. Go to `src/config.py` and add/edit the configuration for an evaluation that you want. 
2. Run `hhem-leaderboard --eval_name <eval_name>` to run the evaluation. 

## To evaluate an LLM from a provider that is alreayd supported 


#### Configuration superseding order

1. LLM-agnostic configuration in `config.py`
2. LLM-specific configuration in `config.py` under `LLM_Configs`
3. Default values in individual LLM classes

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
