import dotenv
dotenv.load_dotenv()

from openai import OpenAI
client = OpenAI()

models: dict = client.models.list().model_dump()

# filter models by those beging with gpt and o

gpt_models = [model['id'] for model in models['data'] if model['id'].startswith('gpt-') or model['id'].startswith('o-')]

print(sorted(gpt_models))