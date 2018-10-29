import tensorflow as tf

from metric_learning.constants.distance_function import DistanceFunction

from util.registry.loss_function import LossFunction


class LatentPositionLoss(LossFunction):
    name = 'latent_position'

    def loss(self, batch, model, dataset):
        loss_conf = self.conf['loss']
        pairwise_distance, y = dataset.get_pairwise_distances(
            batch, model, DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED)
        eta = loss_conf['alpha'] - pairwise_distance
        signed_eta = tf.multiply(eta, -2 * tf.cast(y, tf.float32) + 1)
        padded_signed_eta = tf.stack([tf.zeros(signed_eta.shape[0]), signed_eta])

        return tf.reduce_mean(tf.reduce_logsumexp(padded_signed_eta, axis=0))

    def __str__(self):
        return self.name + '_' + self.conf['loss']['parametrization']
