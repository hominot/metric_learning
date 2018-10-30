from util.registry.class_registry import ClassRegistry


class BatchDesign(object, metaclass=ClassRegistry):
    module_path = 'metric_learning.batch_designs'

    def __init__(self, conf, extra_info):
        super(BatchDesign, self).__init__()
        self.conf = conf
        self.extra_info = extra_info
        self.data_loader = extra_info['data_loader']

    def create_dataset(self, image_files, labels, testing=False):
        raise NotImplementedError

    def get_pairwise_distances(self, batch, model, distance_function):
        raise NotImplementedError

    def get_npair_distances(self, batch, model, distance_function):
        raise NotImplementedError

    def get_embeddings(self, batch, model, distance_function):
        raise NotImplementedError

    def __str__(self):
        return self.conf['batch_design']['dataset']['name']
