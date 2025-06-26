# Running the Program

### Configuration
Before running the program you need to define CONFIG in config.py to include the settings and models you want.


#### File Location
```plaintext
src/
└── config.py
```

#### File

```python
# ... Other Config

# Runtime config
# ADJUST THIS GLOBAL VARIABLE
CONFIG = {
    # Defines protocols to perform, done in seqence. Not all sequences are valid
    "pipeline": [GET_SUMM, GET_JUDGE, GET_RESULTS],
    # Specifies if summary jsonl should be overwritten before appending new data. Judgemennts and results are always overwritten before appending
    "overwrite": True,
    # Specifies the article dataset to generate summaries for
    "input_file": DATA_PATH,
    # Sets the temperature for all models to this value if it can be set
    "temperature": 0.0, 
    # Sets the max_tokens for all models to this value if it can be set
    "max_tokens": 1024,
    # Necssary for confidence intervals. Specifies how many times to repeat the summary generation process for all articles
    "simulation_count": 5,
    # Necessary for confidence intervals. Samples a subset of the summaries for all simulations. Should be strictly less than simulation count
    "sample_count": 2,
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
                # temperature: Optional defaults to 0.0
                "temperature": 0.0,
                # max_tokens: Optional, defaults to 1024
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

# ... Other Config
```

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
from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY
from src.LLMs.model_registry import register_model
import os
from src.data_struct.config_model import ExecutionMode

COMPANY = "company_name"
@register_model(COMPANY)
class CompanyName(AbstractLLM):
    """
    Class for models from company_name

    Attributes:
    """

    local_models = ["local_model"]
    client_models = ["api_model"]

    model_category1 = ["api_model"]
    model_category2 = ["local_model"]

    def __init__(
            self,
            model_name: str,
            exeuction_mode: ExecutionMode,
            date_code: str,
            temperature: float,
            max_tokens: int,
            thinking_tokens: int,
            min_throttle_time: float
    ):
        super().__init__(
            model_name,
            execution_mode
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
        if self.client and self.model_name in self.model_category1:
            # Protocol to run API models in model_category1
            # NEW CODE HERE
        elif self.local_model and self.model_name in self.model_category2:
            # Protocol to run local models in model_category2
            # NEW CODE HERE
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        return summary

    def setup(self):
        if self.valid_client_model():
            # How to set up client
            api_key = os.getenv("API_KEY")
            self.client = # NEW CODE HERE
        elif self.valid_local_model():
            # How to set up model
            self.local_model = # NEW CODE HERE

    def teardown(self):
        if self.valid_client_model():
            # How to teardown client if needed
            # NEW CODE HERE
        elif self.valid_local_model():
            # How to teardown local model if needed
            # NEW CODE HERE
```
Add your new model to the global CONFIG variable and run the test script to verify that the model does indeed return a string.

```bash
python3 test_script.py
```

### New Model to Existing Company

If the company.py file already exists you need to add code with the correct protocol for the model. If the protocol matches the protocol for another model, add it to the same list as the matching model. If it doesn't exist create a new list with the new model and then define its protocol under a new conditional branch in summarize.

```python
from src.LLMs.AbstractLLM import AbstractLLM, EMPTY_SUMMARY
import os
from src.LLMs.model_registry import register_model
from src.data_struct.config_model import ExecutionMode

COMPANY = "company_name"
@register_model(COMPANY)
class CompanyName(AbstractLLM):
    """
    Class for models from company_name

    Attributes:
    """

    local_models = ["local_model"]
    client_models = ["api_model", "new_api_model2", "new_api_model3"]

    model_category1 = ["api_model", "new_api_model2"]
    model_category2 = ["local_model"]
    model_category3 = ["new_api_model3"]

    def __init__(
            self,
            model_name: str,
            execution_mode: ExecutionMode,
            date_code: str,
            temperature: float,
            max_tokens: int,
            thinking_tokens: int,
            min_throttle_time: float
    ):
        super().__init__(
            model_name,
            execution_mode,
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
        if self.client and self.model_name in self.model_category1:
            # Protocol to run API models in model_category1
            # Same as previous
        elif self.client and self.model_name in self.model_category3:
            # Protocol to run API models in model_category3
            # NEW CODE HERE
        elif self.local_model and self.model_name in self.model_category2:
            # Protocol to run local models in model_category2
            # Same as previous
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        return summary

    def setup(self):
        if self.valid_client_model():
            # How to set up client
            api_key = os.getenv("API_KEY")
            self.client = # Same as previous
        elif self.valid_local_model():
            # How to set up model
            self.local_model = # Same as previous

    def teardown(self):
        if self.valid_client_model():
            # How to teardown client if needed
            # Same as previous
        elif self.valid_local_model():
            # How to teardown local model if needed
            # Same as previous
```

Same as before add your new model to the global CONFIG variable and run the test script to verify that the model does indeed return a string.

```bash
python3 test_script.py
```
