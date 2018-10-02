import tensorflow as tf

from util.registry.loss_function import LossFunction


def pairwise_euclidean_distance_squared(embeddings):
    return tf.reduce_sum(
        tf.square(embeddings[None] - embeddings[:, None]),
        axis=2)


def pairwise_dot_product(embeddings):
    return tf.reduce_sum(
        tf.multiply(embeddings[None], embeddings[:, None]),
        axis=2)


def pairwise_matching_matrix(labels):
    return tf.cast(tf.equal(labels[None], labels[:, None]), tf.float32) * 2 - 1


def upper_triangular_part(matrix):
    a = tf.linalg.band_part(tf.ones(matrix.shape), -1, 0)
    return tf.boolean_mask(matrix, 1 - a)


class LatentPositionLoss(LossFunction):
    name = 'latent_position'

    def __init__(self, conf, extra_info):
        super(LatentPositionLoss, self).__init__(conf, extra_info)

        loss_conf = conf['model']['loss']
        alpha = loss_conf.get('alpha', 1.0)
        self.extra_variables['alpha'] = tf.keras.backend.variable(value=alpha, dtype='float32')

    def loss(self, embeddings, labels):
        loss_conf = self.conf['model']['loss']
        if loss_conf['parametrization'] == 'bias':
            pairwise_distance = pairwise_euclidean_distance_squared(embeddings)
            eta = self.conf['model']['loss'].get('alpha', 4.0) - pairwise_distance
        elif loss_conf['parametrization'] == 'dot_product':
            dot_products = pairwise_dot_product(embeddings)
            eta = self.extra_variables['alpha'] + dot_products
        else:
            raise Exception

        y = pairwise_matching_matrix(labels)
        signed_eta = upper_triangular_part(tf.multiply(eta, -y))
        padded_signed_eta = tf.stack([tf.zeros(signed_eta.shape[0]), signed_eta])

        return tf.reduce_mean(tf.reduce_logsumexp(padded_signed_eta, axis=0))

    def __str__(self):
        return self.name + '_' + self.conf['model']['loss']['parametrization']
