import tensorflow as tf

from util.registry.loss_function import LossFunction
from util.tensor_operations import pairwise_euclidean_distance_squared
from util.tensor_operations import pairwise_matching_matrix
from util.tensor_operations import upper_triangular_part
from util.tensor_operations import repeat_columns
from util.tensor_operations import pairwise_difference


class TripletLossFunction(LossFunction):
    name = 'triplet'

    def loss(self, embeddings, labels):
        pairwise_distances = upper_triangular_part(pairwise_euclidean_distance_squared(embeddings, embeddings))
        matching_labels_matrix = tf.cast(
            upper_triangular_part(tf.cast(pairwise_matching_matrix(labels, labels), tf.int64)),
            tf.bool)
        positive_distances = tf.boolean_mask(pairwise_distances, matching_labels_matrix)
        negative_distances = tf.boolean_mask(pairwise_distances, ~matching_labels_matrix)

        first_labels = upper_triangular_part(tf.cast(repeat_columns(labels), tf.int64))
        positive_labels = tf.boolean_mask(first_labels, matching_labels_matrix)
        negative_labels = tf.boolean_mask(first_labels, ~matching_labels_matrix)

        triplet_match = pairwise_matching_matrix(positive_labels, negative_labels)

        triplet_loss = tf.boolean_mask(
            tf.maximum(0., self.conf['model']['loss'].get('alpha', 1.0) + pairwise_difference(positive_distances, negative_distances)),
            triplet_match)

        return sum(triplet_loss)
