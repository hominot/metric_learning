import tensorflow as tf

from util.registry.class_registry import ClassRegistry
from util.registry.loss_function import LossFunction
from tensorflow.keras.layers import Dense


class Model(tf.keras.models.Model, metaclass=ClassRegistry):
    module_path = 'metric_learning.models'

    loss_function = None
    model = None

    variable_names = ['model']

    def __init__(self, conf, extra_info):
        super(Model, self).__init__()
        self.conf = conf
        self.extra_info = extra_info

        self.loss_function = LossFunction.create(conf['loss']['name'], conf, extra_info)
        for k, v in self.loss_function.extra_variables.items():
            setattr(self, k, v)
            self.variable_names.append(k)
        if 'dimension' in conf['model']:
            self.embedding = Dense(conf['model']['dimension'],
                                   name='dimension_reduction')
            self.variable_names.append('embedding')

    def loss(self, images, labels, image_ids):
        embeddings = self.call(images, training=True)
        return self.loss_function.loss(embeddings, labels, image_ids)

    def __str__(self):
        return self.conf['model']['name'] + '_' + str(self.loss_function)

    def preprocess_image(self, image):
        return (image / 255. - 0.5) * 2

    def learning_rates(self):
        return {
            k: (
                self.conf['trainer'].get('lr_{}'.format(k), self.conf['trainer']['learning_rate']),
                getattr(self, k).variables if hasattr(getattr(self, k), 'variables') else getattr(self, k)
            ) for k in self.variable_names
        }

    def call(self, inputs, training=None, mask=None):
        ret = self.model(self.preprocess_image(inputs),
                         training=training,
                         mask=mask)
        if 'dimension' in self.conf['model']:
            ret = self.embedding(ret)
        if self.conf['model']['l2_normalize']:
            ret = tf.nn.l2_normalize(ret)
        return ret
