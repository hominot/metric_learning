from util.class_registry import ClassRegistry


class LossFunction(object, metaclass=ClassRegistry):
    module_path = 'metric_learning.loss_functions'

    def loss(self, embeddings, labels, *args, **kwargs):
        raise NotImplementedError

