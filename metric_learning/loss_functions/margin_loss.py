import tensorflow as tf

from util.registry.loss_function import LossFunction

from util.tensor_operations import pairwise_euclidean_distance_squared
from util.tensor_operations import pairwise_matching_matrix
from util.tensor_operations import upper_triangular_part


class MarginLoss(LossFunction):
    name = 'margin'

    def loss(self, embeddings, labels):
        pairwise_distances_squared = upper_triangular_part(pairwise_euclidean_distance_squared(embeddings, embeddings))
        pairwise_distances = tf.sqrt(pairwise_distances_squared)
        matching_labels_matrix = tf.cast(
            upper_triangular_part(tf.cast(pairwise_matching_matrix(labels, labels), tf.int64)),
            tf.bool)

        positive_distances = tf.boolean_mask(pairwise_distances, matching_labels_matrix)
        negative_distances = tf.boolean_mask(pairwise_distances, ~matching_labels_matrix)

        loss_value = sum(tf.maximum(
            0, self.conf['loss']['alpha'] + positive_distances - self.conf['loss']['beta'])) + sum(
            tf.maximum(
                0, self.conf['loss']['alpha'] - negative_distances + self.conf['loss']['beta']
            )
        )

        return loss_value / int(pairwise_distances.shape[0])

    def __str__(self):
        return self.name
