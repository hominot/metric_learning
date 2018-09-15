from util.class_registry import ClassRegistry


class Metric(object, metaclass=ClassRegistry):
    module_path = 'metric_learning.metrics'

    def compute_metric(self, model, test_ds, *args, **kwargs):
        raise NotImplementedError
