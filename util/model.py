import tensorflow as tf

from util.class_registry import ClassRegistry


class Model(tf.keras.models.Model, metaclass=ClassRegistry):
    module_path = 'metric_learning.models'
