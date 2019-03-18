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
        if self.conf['loss'].get('importance_sampling') or self.conf['loss'].get('new_importance_sampling') or self.conf['loss'].get('balanced_pairs'):
            positive_weights = tf.boolean_mask(weights, matching_labels_matrix)
            negative_weights = tf.boolean_mask(weights, ~matching_labels_matrix)
            loss_value = (
                sum(positive_loss * positive_weights) +
                sum(negative_loss * negative_weights)
            ) / int(pairwise_distances.shape[0])
        else:
            loss_value = (
                sum(positive_loss) +
                sum(negative_loss)
            ) / int(pairwise_distances.shape[0])

        return loss_value