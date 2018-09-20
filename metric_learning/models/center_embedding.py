import tensorflow as tf

from util.registry.model import Model


layers = tf.keras.layers


class CenterEmbeddingModel(Model):
    name = 'center_embedding'

    def __init__(self, conf, extra_info):
        super(CenterEmbeddingModel, self).__init__(conf, extra_info)

        self.child_model = Model.create(conf['child_model'], extra_info)
        self.embedding = layers.Embedding(extra_info['num_labels'], conf['child_model']['k'])

    def call(self, inputs, training=None, mask=None):
        return self.child_model(inputs, training, mask)

    def loss(self, images, labels):
        embeddings = self.call(images, training=True)
        mean_embeddings = self.embedding(labels - 1)
        d = tf.norm(embeddings - mean_embeddings, axis=1)
        return tf.reduce_mean(tf.maximum(0, d - 0.2))

