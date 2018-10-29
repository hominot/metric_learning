import tensorflow as tf

from util.registry.loss_function import LossFunction

from util.tensor_operations import pairwise_euclidean_distance_squared
from util.tensor_operations import pairwise_matching_matrix
from util.tensor_operations import off_diagonal_part
from util.tensor_operations import repeat_columns
from util.tensor_operations import stable_sqrt


class MarginLoss(LossFunction):
    name = 'margin'

    def __init__(self, conf, extra_info):
        super(MarginLoss, self).__init__(conf, extra_info)

        loss_conf = conf['loss']
        if 'num_labels' in extra_info and 'num_images' in extra_info:
            beta_class = tf.ones(extra_info['num_labels']) * loss_conf['beta']

            self.extra_variables['beta'] = tf.keras.backend.variable(value=beta_class, dtype='float32')

    def loss(self, embeddings, labels):
        pairwise_distances_squared = off_diagonal_part(pairwise_euclidean_distance_squared(embeddings, embeddings))
        pairwise_distances = stable_sqrt(pairwise_distances_squared)
        matching_labels_matrix = tf.cast(
            off_diagonal_part(tf.cast(pairwise_matching_matrix(labels, labels), tf.int64)),
            tf.bool)

        positive_distances = tf.boolean_mask(pairwise_distances, matching_labels_matrix)
        negative_distances = tf.boolean_mask(pairwise_distances, ~matching_labels_matrix)

        label_indices = off_diagonal_part(tf.cast(repeat_columns(labels), tf.int64))
        betas = tf.gather(self.extra_variables['beta'], label_indices)

        positive_betas = tf.boolean_mask(betas, matching_labels_matrix)
        negative_betas = tf.boolean_mask(betas, ~matching_labels_matrix)
        loss_value = sum(tf.maximum(
            0, self.conf['loss']['alpha'] + positive_distances - positive_betas)) + sum(
            tf.maximum(
                0, self.conf['loss']['alpha'] - negative_distances + negative_betas
            )
        )

        nu = self.conf['loss']['nu']

        return (loss_value + nu * sum(betas)) / int(pairwise_distances.shape[0])

    def __str__(self):
        return self.name
