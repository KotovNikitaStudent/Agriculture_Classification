from abc import ABC, abstractmethod


class AbstractModel(ABC):
    @abstractmethod
    def __init__(self, pretrained):
        pass
    
    @abstractmethod
    def get_model(self, n_out, weight):
        pass
    