import tensorflow as tf

import numpy as np

from metric_learning.constants.distance_function import DistanceFunction
from util.registry.loss_function import LossFunction

from util.tensor_operations import off_diagonal_part
from util.tensor_operations import repeat_columns

tfe = tf.contrib.eager


class MarginLoss(LossFunction):
    name = 'margin'

    def loss(self, batch, model, dataset):
        images, labels = batch
        pairwise_distances, matching_labels_matrix, weights = dataset.get_raw_pairwise_distances(
            batch, model, DistanceFunction.EUCLIDEAN_DISTANCE
        )
        positive_distances = tf.boolean_mask(pairwise_distances, matching_labels_matrix)
        negative_distances = tf.boolean_mask(pairwise_distances, ~matching_labels_matrix)

        alpha = self.conf['loss']['alpha']
        beta = self.conf['loss']['beta']
        positive_loss = tf.maximum(positive_distances - beta, 0.0)
        negative_loss = tf.maximum(alpha - negative_distances, 0.0)

        pair_cnt = tf.reduce_sum(tf.cast(positive_loss > 0, tf.float32)) + tf.reduce_sum(tf.cast(negative_loss > 0, tf.float32))

        return tf.reduce_sum(positive_loss + negative_loss) / pair_cnt
