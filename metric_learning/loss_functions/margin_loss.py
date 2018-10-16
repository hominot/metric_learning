import tensorflow as tf

from util.registry.loss_function import LossFunction

from util.tensor_operations import pairwise_euclidean_distance_squared
from util.tensor_operations import pairwise_matching_matrix
from util.tensor_operations import off_diagonal_part
from util.tensor_operations import repeat_columns


class MarginLoss(LossFunction):
    name = 'margin'

    def __init__(self, conf, extra_info):
        super(MarginLoss, self).__init__(conf, extra_info)

        loss_conf = conf['loss']
        beta = loss_conf['beta']
        beta_class = tf.ones(extra_info['num_labels']) * loss_conf['beta_class']
        beta_image = tf.ones(extra_info['num_images']) * loss_conf['beta_image']

        self.extra_variables['beta'] = tf.keras.backend.variable(value=beta, dtype='float32')
        self.extra_variables['beta_class'] = tf.keras.backend.variable(value=beta_class, dtype='float32')
        self.extra_variables['beta_image'] = tf.keras.backend.variable(value=beta_image, dtype='float32')

    def loss(self, embeddings, labels, image_ids):
        pairwise_distances_squared = off_diagonal_part(pairwise_euclidean_distance_squared(embeddings, embeddings))
        pairwise_distances = tf.sqrt(pairwise_distances_squared)
        matching_labels_matrix = tf.cast(
            off_diagonal_part(tf.cast(pairwise_matching_matrix(labels, labels), tf.int64)),
            tf.bool)

        positive_distances = tf.boolean_mask(pairwise_distances, matching_labels_matrix)
        negative_distances = tf.boolean_mask(pairwise_distances, ~matching_labels_matrix)

        loss_value = sum(tf.maximum(
            0, self.conf['loss']['alpha'] + positive_distances - self.conf['loss']['beta'])) + sum(
            tf.maximum(
                0, self.conf['loss']['alpha'] - negative_distances + self.conf['loss']['beta']
            )
        )

        label_regularizers = off_diagonal_part(repeat_columns(labels))
        image_regularizers = off_diagonal_part(repeat_columns(image_ids))
        nu = self.conf['loss']['nu']

        regularizers = self.conf['loss']['beta'] + tf.gather(self.extra_variables['beta_class'], label_regularizers) + tf.gather(self.extra_variables['beta_image'], image_regularizers)

        return (loss_value + nu * sum(regularizers)) / int(pairwise_distances.shape[0])

    def __str__(self):
        return self.name
