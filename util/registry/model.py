import tensorflow as tf

from util.registry.class_registry import ClassRegistry
from util.registry.loss_function import LossFunction
from tensorflow.keras.layers import Dense


class Model(tf.keras.models.Model, metaclass=ClassRegistry):
    module_path = 'metric_learning.models'

    loss_function = None
    model = None

    def __init__(self, conf, extra_info):
        super(Model, self).__init__()
        self.conf = conf
        self.extra_info = extra_info

        self.loss_function = LossFunction.create(conf['loss']['name'], conf, extra_info)
        for k, v in self.loss_function.extra_variables.items():
            setattr(self, k, v)
        if 'dimension' in conf['model']:
            self.dense_layer = Dense(conf['model']['dimension'])

    def loss(self, images, labels, image_ids):
        embeddings = self.call(images, training=True)
        return self.loss_function.loss(embeddings, labels, image_ids)

    def __str__(self):
        return self.conf['model']['name'] + '_' + str(self.loss_function)

    def call(self, inputs, training=None, mask=None):
        ret = self.model(inputs, training=training, mask=mask)
        if 'dimension' in self.conf['model']:
            ret = self.dense_layer(ret)
        return ret
