import tensorflow as tf

from util.registry.loss_function import LossFunction


layers = tf.keras.layers


class LatentPositionLoss(LossFunction):
    name = 'latent_position'

    def __init__(self, conf, extra_info):
        super(LatentPositionLoss, self).__init__(conf, extra_info)

        if conf['method'] == 'distance':
            self.extra_variables['alpha'] = tf.keras.backend.variable(value=1.0, dtype='float32')
        else:
            self.extra_variables['alpha'] = tf.keras.backend.variable(value=-3.0, dtype='float32')

    def loss(self, embeddings, labels):
        if self.conf['method'] == 'distance':
            difference = tf.reshape(embeddings[None] - embeddings[:, None], [-1, int(embeddings.shape[1])])
            eta = self.extra_variables['alpha'] - tf.reduce_sum(tf.square(difference), axis=1)
        elif self.conf['method'] == 'projection':
            product = tf.reshape(tf.multiply(embeddings[None], embeddings[:, None]), [-1, int(embeddings.shape[1])])
            eta = self.extra_variables['alpha'] + tf.reduce_sum(product, axis=1)
        else:
            raise Exception

        y = tf.reshape(tf.equal(labels[None], labels[:, None]), [-1])
        b = tf.equal(tf.reshape(tf.eye(int(embeddings.shape[0])), [-1]), 1)
        positive = sum(tf.boolean_mask(eta, y & ~b))
        negative = tf.reduce_sum(tf.log(1 + tf.exp(tf.boolean_mask(eta, ~b))))

        return -positive + negative

    def __str__(self):
        return self.name + '_' + self.conf['method']
