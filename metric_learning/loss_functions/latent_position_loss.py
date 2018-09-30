import tensorflow as tf

from util.registry.loss_function import LossFunction


layers = tf.keras.layers


class LatentPositionLoss(LossFunction):
    name = 'latent_position'

    def __init__(self, conf, extra_info):
        super(LatentPositionLoss, self).__init__(conf, extra_info)

        alpha = conf.get('alpha', 1.0)
        if conf['parametrization'] == 'bias':
            self.extra_variables['alpha'] = tf.keras.backend.variable(value=alpha, dtype='float32')
        elif conf['parametrization'] == 'unit':
            self.extra_variables['alpha'] = tf.keras.backend.variable(value=alpha, dtype='float32')
        else:
            raise Exception

    def loss(self, embeddings, labels):
        difference = tf.reshape(embeddings[None] - embeddings[:, None], [-1, int(embeddings.shape[1])])
        if self.conf['parametrization'] == 'bias':
            eta = self.extra_variables['alpha'] - tf.reduce_sum(tf.square(difference), axis=1)
        elif self.conf['parametrization'] == 'unit':
            eta = self.extra_variables['alpha'] * (1 - tf.reduce_sum(tf.square(difference), axis=1))
        else:
            raise Exception

        y = tf.reshape(tf.equal(labels[None], labels[:, None]), [-1])
        b = tf.equal(tf.reshape(tf.eye(int(embeddings.shape[0])), [-1]), 1)
        positive = sum(tf.boolean_mask(eta, y & ~b))
        negative = tf.reduce_sum(tf.log(1 + tf.exp(tf.boolean_mask(eta, ~b))))

        return -positive + negative

    def __str__(self):
        return self.name + '_' + self.conf['parametrization']
