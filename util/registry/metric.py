from util.registry.class_registry import ClassRegistry


class Metric(object, metaclass=ClassRegistry):
    module_path = 'metric_learning.metrics'

    def __init__(self, conf, extra_info):
        super(Metric, self).__init__()
        self.conf = conf
        self.extra_info = extra_info

    def compute_metric(self, model, test_ds, num_testcases):
        raise NotImplementedError

    def __str__(self):
        return self.conf['name']
