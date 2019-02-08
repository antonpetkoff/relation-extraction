from abc import ABC, abstractmethod

import numpy as np

# from utils import update_dict
from utils import update_dict


class BaseModel(ABC):
    @abstractmethod
    def fit(self, train_x, train_y): pass

    @abstractmethod
    def predict(self, test_x): pass

    def get_params(self, deep = True):
        return {}

    def set_params(self, **params):
        for key, value in params.items():
            update_dict(self.params, key.split('__'), value)
        return self

    def get_grid_params(self):
        return {}
