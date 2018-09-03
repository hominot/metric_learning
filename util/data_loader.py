from util.class_registry import ClassRegistry


class DataLoader(object, metaclass=ClassRegistry):
    def load_dataset(self):
        pass

from metric_learning.dataset import *
