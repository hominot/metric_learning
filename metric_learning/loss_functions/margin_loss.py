import tensorflow as tf

from metric_learning.constants.distance_function import DistanceFunction
from util.registry.loss_function import LossFunction

from util.tensor_operations import off_diagonal_part
from util.tensor_operations import repeat_columns

tfe = tf.contrib.eager


class MarginLoss(LossFunction):
    name = 'margin'

    def __init__(self, conf, extra_info):
        super(MarginLoss, self).__init__(conf, extra_info)

        loss_conf = conf['loss']
        if 'num_labels' in extra_info and 'num_images' in extra_info:
            beta_class = tf.ones(extra_info['num_labels']) * loss_conf['beta']

            self.extra_variables['beta'] = tfe.Variable(beta_class)

    def loss(self, batch, model, dataset):
        images, labels = batch
        pairwise_distances, matching_labels_matrix, weights = dataset.get_raw_pairwise_distances(
            batch, model, DistanceFunction.EUCLIDEAN_DISTANCE
        )
        pairwise_distances = off_diagonal_part(pairwise_distances)
        matching_labels_matrix = off_diagonal_part(matching_labels_matrix)

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
