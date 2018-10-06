import tensorflow as tf

from util.registry.loss_function import LossFunction
from util.tensor_operations import pairwise_euclidean_distance_squared
from util.tensor_operations import pairwise_matching_matrix
from util.tensor_operations import upper_triangular_part


class ContrastiveLossFunction(LossFunction):
    name = 'contrastive'

    def loss(self, embeddings, labels, *args, **kwargs):
        alpha = self.conf['model']['loss'].get('alpha', 8.)
        pairwise_distances = upper_triangular_part(pairwise_euclidean_distance_squared(embeddings, embeddings))
        matching_labels_matrix = tf.cast(
            upper_triangular_part(tf.cast(pairwise_matching_matrix(labels), tf.int32)),
            tf.bool)
        positive_distances = tf.boolean_mask(pairwise_distances, matching_labels_matrix)
        negative_distances = tf.boolean_mask(pairwise_distances, ~matching_labels_matrix)
        loss_value = sum(positive_distances) + sum(tf.maximum(0, alpha - negative_distances))

        return loss_value
