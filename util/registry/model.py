import tensorflow as tf

from util.registry.class_registry import ClassRegistry
from util.registry.loss_function import LossFunction


class Model(tf.keras.models.Model, metaclass=ClassRegistry):
    module_path = 'metric_learning.models'

    loss_function = None

    def __init__(self, conf, extra_info):
        super(Model, self).__init__()
        self.conf = conf
        self.extra_info = extra_info
        if 'loss' in conf:
            self.loss_function = LossFunction.create(conf['loss'])

    def loss(self, images, labels):
        embeddings = self.call(images, training=True)
        return self.loss_function.loss(embeddings, labels)
