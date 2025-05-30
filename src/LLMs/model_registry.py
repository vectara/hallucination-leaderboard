MODEL_REGISTRY = {}

def register_model(company):
    def decorator(cls):
        MODEL_REGISTRY[company] = cls
        return cls
    return decorator