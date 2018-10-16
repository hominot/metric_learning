import tensorflow as tf

from util.registry.loss_function import LossFunction
from util.dataset import group_npairs

from util.tensor_operations import pairwise_euclidean_distance_squared
from util.tensor_operations import pairwise_matching_matrix
from util.tensor_operations import upper_triangular_part
from util.tensor_operations import pairwise_dot_product


class LatentPositionLoss(LossFunction):
    name = 'latent_position'

    def loss(self, embeddings, labels, image_ids):
        loss_conf = self.conf['loss']
        if 'npair' not in loss_conf:
            if loss_conf['parametrization'] == 'euclidean_distance':
                pairwise_distance = pairwise_euclidean_distance_squared(embeddings, embeddings)
                eta = loss_conf['alpha'] - pairwise_distance
            elif loss_conf['parametrization'] == 'dot_product':
                dot_products = pairwise_dot_product(embeddings, embeddings)
                eta = loss_conf['alpha'] + dot_products
            else:
                raise Exception

            y = pairwise_matching_matrix(labels, labels)
            signed_eta = upper_triangular_part(tf.multiply(eta, -2 * tf.cast(y, tf.float32) + 1))
            padded_signed_eta = tf.stack([tf.zeros(signed_eta.shape[0]), signed_eta])

            return tf.reduce_mean(tf.reduce_logsumexp(padded_signed_eta, axis=0))

        # npair compatible loss for fair comparison with n-tuplet loss
        npairs = group_npairs(embeddings, labels, loss_conf['npair']['n'])
        if loss_conf['parametrization'] == 'euclidean_distance':
            pairwise_distances = tf.concat(
                [pairwise_euclidean_distance_squared(first, second) for first, second in npairs],
                axis=0)
            eta = loss_conf['alpha'] * 100. - pairwise_distances
        elif loss_conf['parametrization'] == 'dot_product':
            dot_products = tf.concat(
                [pairwise_dot_product(first, second) for first, second in npairs],
                axis=0)
            eta = loss_conf['alpha'] + dot_products
        else:
            raise Exception
        y = 1. - tf.eye(loss_conf['npair']['n']) * 2.
        signed_eta = tf.reshape(tf.multiply(eta, tf.concat([y] * len(npairs), axis=0)), [-1])
        padded_signed_eta = tf.stack([tf.zeros(signed_eta.shape[0]), signed_eta])

        return tf.reduce_mean(tf.reduce_logsumexp(padded_signed_eta, axis=0))

    def __str__(self):
        return self.name + '_' + self.conf['loss']['parametrization']
