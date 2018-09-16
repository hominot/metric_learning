import tensorflow as tf

from util.loss_function import LossFunction


class GridLossFunction(LossFunction):
    name = 'grid'

    def loss(self, embeddings, labels):
        grid_points_for_labels = kwargs['grid_points'][labels - 1, :]
        d = tf.norm(embeddings - grid_points_for_labels, axis=1)
        return tf.reduce_mean(tf.maximum(0, d - 0.2))
