from abc import ABC, abstractmethod

class AbstractTest(ABC):
    """
    Abstract Class for testing code

    Attributes:
        None

    Methods:
        run_tests(): run all tests
    
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def run_tests(self):
        pass