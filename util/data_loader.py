from util.class_registry import ClassRegistry


class DataLoader(object, metaclass=ClassRegistry):
    name = 'None'

    def data_path(self):
        return '/tmp/research/{}'.format(self.name)

    def prepare_files(self):
        pass

    def load_dataset(self):
        pass

from metric_learning.datasets import *
