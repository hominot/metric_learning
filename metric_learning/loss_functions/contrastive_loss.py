import tensorflow as tf

from util.registry.loss_function import LossFunction
from util.tensor_operations import pairwise_euclidean_distance_squared
from util.tensor_operations import pairwise_matching_matrix
from util.tensor_operations import upper_triangular_part
from util.tensor_operations import stable_sqrt


class ContrastiveLossFunction(LossFunction):
    name = 'contrastive'

    def loss(self, embeddings, labels, image_ids, *args, **kwargs):
        alpha = self.conf['loss']['alpha']
        pairwise_distances = upper_triangular_part(pairwise_euclidean_distance_squared(embeddings, embeddings))
        matching_labels_matrix = tf.cast(
            upper_triangular_part(tf.cast(pairwise_matching_matrix(labels, labels), tf.int64)),
            tf.bool)
        positive_distances = tf.boolean_mask(pairwise_distances, matching_labels_matrix)
        negative_distances = tf.boolean_mask(pairwise_distances, ~matching_labels_matrix)
        loss_value = (
            sum(positive_distances) +
            sum(tf.square(tf.maximum(0, alpha - stable_sqrt(negative_distances))))
        ) / int(pairwise_distances.shape[0])

        return loss_value
