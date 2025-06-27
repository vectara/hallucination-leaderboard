class ClientOrLocalNotInitializedError(Exception):
    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__(f"Model Initialization Error for {model_name}: model is either not included in the class variable client_models or local_models list, or setup was not properly defined and self.client or self.local_model are still None.")


class ClientModelProtocolBranchNotFound(Exception):
    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__(f"Client model protocol branch not found for {model_name}: model is not included in a model_categoryX class variable that or the category does not have a defined conditional banch.")

class LocalModelProtocolBranchNotFound(Exception):
    def __init__(self, model_name):
        self.model_name = model_name
        super().__init__(f"Local model protocol branch not found for {model_name}: model is not included in a model_categoryX class variable that or the category does not have a defined conditional banch.")