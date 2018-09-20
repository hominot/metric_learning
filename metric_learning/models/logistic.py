import tensorflow as tf

from util.registry.model import Model


layers = tf.keras.layers


class LogistciModel(Model):
    name = 'logistic'

    def __init__(self, conf, extra_info):
        super(LogistciModel, self).__init__(conf, extra_info)

        self.child_model = Model.create(conf['child_model'], extra_info)
        self.last_layer = layers.Dense(extra_info['num_labels'])

    def call(self, inputs, training=None, mask=None):
        return self.child_model(inputs, training, mask)

    def loss(self, images, labels):
        embeddings = self.call(images, training=True)
        response = self.last_layer(embeddings)

        positive_index = tf.transpose(
            tf.stack([
                tf.range(images.shape[0]),
                labels - 1,
            ])
        )
        loglikelihood = -tf.gather_nd(response, positive_index) + tf.log(tf.reduce_sum(tf.exp(response), axis=1))
        return tf.reduce_mean(loglikelihood)
