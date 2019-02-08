from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def fit(self, train_x, train_y): pass

    @abstractmethod
    def predict(self, test_x): pass

    def get_params(self, deep = True):
        return {}

    def get_grid_params(self):
        return {}
