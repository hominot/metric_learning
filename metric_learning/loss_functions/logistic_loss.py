import tensorflow as tf

from metric_learning.constants.distance_function import DistanceFunction

from util.registry.loss_function import LossFunction


class LogisticLoss(LossFunction):
    name = 'logistic'

    def loss(self, batch, model, dataset):
        loss_conf = self.conf['loss']
        if self.conf['loss']['npair']:
            pairwise_distance, y = dataset.get_npair_distances(
                batch, model, self.conf['loss']['npair'],
                DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED)
            pairwise_distance = tf.reshape(pairwise_distance, [-1])
            y = tf.reshape(y, [-1])
        else:
            pairwise_distance, y = dataset.get_pairwise_distances(
                batch, model, DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED)
        eta = loss_conf['alpha'] - pairwise_distance
        signed_eta = tf.multiply(eta, -2 * tf.cast(y, tf.float32) + 1)
        padded_signed_eta = tf.stack([tf.zeros(signed_eta.shape[0]), signed_eta])

        return tf.reduce_mean(tf.reduce_logsumexp(padded_signed_eta, axis=0))
