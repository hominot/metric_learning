import tensorflow as tf

from util.registry.model import Model


layers = tf.keras.layers


class LatentPositionModel(Model):
    name = 'latent_position'

    def __init__(self, conf, extra_info):
        super(LatentPositionModel, self).__init__(conf, extra_info)

        self.child_model = Model.create(conf['child_model'], extra_info)
        self.alpha = tf.keras.backend.variable(value=0.0, dtype='float32')

    def call(self, inputs, training=None, mask=None):
        return self.child_model(inputs, training, mask)

    def loss(self, images, labels):
        embeddings = self.call(images, training=True)
        if self.conf['method'] == 'distance':
            difference = tf.reshape(embeddings[None] - embeddings[:, None], [-1, int(embeddings.shape[1])])
            eta = self.alpha - tf.reduce_sum(tf.square(difference), axis=1)
        elif self.conf['method'] == 'projection':
            product = tf.reshape(tf.multiply(embeddings[None], embeddings[:, None]), [-1, int(embeddings.shape[1])])
            eta = self.alpha + tf.reduce_sum(product, axis=1)
        else:
            raise Exception

        y = tf.reshape(tf.equal(labels[None], labels[:, None]), [-1])
        b = tf.equal(tf.reshape(tf.eye(int(images.shape[0])), [-1]), 1)
        positive = sum(tf.boolean_mask(eta, y & ~b))
        negative = tf.reduce_sum(tf.log(1 + tf.exp(tf.boolean_mask(eta, ~b))))

        return -positive + negative

    def __str__(self):
        return self.name + '_' + str(self.child_model)
