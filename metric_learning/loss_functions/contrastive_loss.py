import tensorflow as tf

from metric_learning.constants.distance_function import DistanceFunction
from util.registry.loss_function import LossFunction
from util.tensor_operations import stable_sqrt


class ContrastiveLossFunction(LossFunction):
    name = 'contrastive'

    def loss(self, batch, model, dataset):
        alpha = self.conf['loss']['alpha']
        pairwise_distances, matching_labels_matrix = dataset.get_pairwise_distances(
            batch, model, DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED)
        positive_distances = tf.boolean_mask(pairwise_distances, matching_labels_matrix)
        negative_distances = tf.boolean_mask(pairwise_distances, ~matching_labels_matrix)
        loss_value = (
            sum(positive_distances) +
            sum(tf.square(tf.maximum(0, alpha - stable_sqrt(negative_distances))))
        ) / int(pairwise_distances.shape[0])

        return loss_value
