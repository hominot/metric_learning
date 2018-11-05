import tensorflow as tf

from metric_learning.constants.distance_function import DistanceFunction
from util.registry.loss_function import LossFunction
from util.tensor_operations import stable_sqrt


class ContrastiveLossFunction(LossFunction):
    name = 'contrastive'

    def loss(self, batch, model, dataset):
        alpha = self.conf['loss']['alpha']

        if self.conf['loss'].get('npair'):
            pairwise_distances, matching_labels_matrix = dataset.get_npair_distances(
                batch, model,self.conf['loss']['npair'],
                DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED)
        else:
            pairwise_distances, matching_labels_matrix, weights = dataset.get_pairwise_distances(
                batch, model, DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED)
        positive_distances = tf.boolean_mask(pairwise_distances, matching_labels_matrix)
        negative_distances = tf.boolean_mask(pairwise_distances, ~matching_labels_matrix)
        if self.conf['loss'].get('importance_sampling'):
            positive_weights = tf.boolean_mask(weights, matching_labels_matrix)
            negative_weights = tf.boolean_mask(weights, ~matching_labels_matrix)
            loss_value = (
                sum(positive_distances * positive_weights) +
                sum(tf.square(tf.maximum(0, alpha - stable_sqrt(negative_distances))) * negative_weights)
            ) / int(pairwise_distances.shape[0])
        else:
            loss_value = (
                sum(positive_distances) +
                sum(tf.square(tf.maximum(0, alpha - stable_sqrt(negative_distances))))
            ) / int(pairwise_distances.shape[0])

        return loss_value
