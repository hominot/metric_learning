import tensorflow as tf

from util.loss_function import LossFunction


class GridLossFunction(LossFunction):
    name = 'grid'

    def loss(self, embeddings, labels, *args, **kwargs):
        grid_points_for_labels = kwargs['grid_points'][labels - 1, :]
        d = tf.reduce_sum(tf.square(embeddings - grid_points_for_labels), axis=1)
        return sum(d)
