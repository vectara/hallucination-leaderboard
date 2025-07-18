import dotenv

dotenv.load_dotenv()

import anthropic

client = anthropic.Anthropic()

anthropic_models = [model['id'] for model in client.models.list(limit=400).model_dump()['data']]

print (sorted(anthropic_models))