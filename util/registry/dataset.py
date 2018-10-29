from util.registry.class_registry import ClassRegistry


class Dataset(object, metaclass=ClassRegistry):
    module_path = 'metric_learning.datasets'

    def __init__(self, conf, extra_info):
        super(Dataset, self).__init__()
        self.conf = conf
        self.extra_info = extra_info
        self.data_loader = extra_info['data_loader']

    def create_dataset(self, image_files, labels, testing=False):
        raise NotImplementedError

    def __str__(self):
        return self.conf['name']
