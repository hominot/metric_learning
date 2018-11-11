import tensorflow as tf

from util.registry.loss_function import LossFunction
from util.tensor_operations import pairwise_matching_matrix
from util.tensor_operations import upper_triangular_part
from util.tensor_operations import repeat_columns
from util.tensor_operations import pairwise_difference
from metric_learning.constants.distance_function import DistanceFunction


class TripletLossFunction(LossFunction):
    name = 'triplet'

    def loss(self, batch, model, dataset):
        images, labels = batch
        pairwise_distances, matching_labels_matrix, weights = dataset.get_raw_pairwise_distances(
            batch, model, DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED)
        pairwise_distances = upper_triangular_part(pairwise_distances)
        matching_labels_matrix = upper_triangular_part(matching_labels_matrix)

        positive_distances = tf.boolean_mask(pairwise_distances, matching_labels_matrix)
        negative_distances = tf.boolean_mask(pairwise_distances, ~matching_labels_matrix)

        first_labels = upper_triangular_part(tf.cast(repeat_columns(labels), tf.int64))
        positive_labels = tf.boolean_mask(first_labels, matching_labels_matrix)
        negative_labels = tf.boolean_mask(first_labels, ~matching_labels_matrix)

        triplet_match = pairwise_matching_matrix(positive_labels, negative_labels)
        differences = pairwise_difference(positive_distances, negative_distances)

        alpha = self.conf['loss']['alpha']
        semi_hard_candidate_mask = triplet_match & \
                                   (differences < 0) & \
                                   (alpha + differences > 0)
        semi_hard_candidates = tf.multiply(
            tf.cast(semi_hard_candidate_mask, tf.float32),
            alpha + differences) + \
            tf.multiply(
                1 - tf.cast(semi_hard_candidate_mask, tf.float32),
                alpha)
        semi_hards = tf.reduce_min(semi_hard_candidates, axis=1)

        semi_hard_triplet_loss = tf.boolean_mask(
            semi_hards,
            (semi_hards > 1e-12) & (semi_hards < alpha - 1e-12))

        return tf.reduce_mean(semi_hard_triplet_loss)
