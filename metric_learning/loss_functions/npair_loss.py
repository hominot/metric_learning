import tensorflow as tf

from metric_learning.constants.distance_function import DistanceFunction
from util.registry.loss_function import LossFunction


class NPairLossFunction(LossFunction):
    name = 'npair'

    def loss(self, batch, model, dataset):
        pairwise_distances, matching_matrix = dataset.get_npair_distances(
            batch, model, self.conf['loss']['npair'],
            DistanceFunction.DOT_PRODUCT)
        embeddings = dataset.get_embeddings(
            batch, model, DistanceFunction.DOT_PRODUCT)
        regularizer = tf.reduce_mean(tf.reduce_sum(tf.square(embeddings), axis=1))
        return tf.reduce_mean(
            tf.reduce_logsumexp(-pairwise_distances, axis=1) +
            tf.boolean_mask(pairwise_distances, matching_matrix)
        ) + regularizer * self.conf['loss']['lambda']
