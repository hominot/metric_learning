import tensorflow as tf
import numpy as np

from util.model import Model


layers = tf.keras.layers


class GridModel(Model):
    name = 'grid'

    def __init__(self, conf, extra_info):
        super(GridModel, self).__init__(conf, extra_info)

        self.child_model = Model.create(conf['child_model'], extra_info)
        self.grid = np.random.random([extra_info['num_labels'], conf['child_model']['k']]) * 2 - 1

    def call(self, inputs, training=None, mask=None):
        return self.child_model(inputs, training, mask)

    def loss(self, images, labels):
        embeddings = self.call(images, training=True)
        grid_points_for_labels = self.grid[labels - 1, :]
        d = tf.norm(embeddings - grid_points_for_labels, axis=1)
        return tf.reduce_mean(tf.maximum(0, d - 0.2))
