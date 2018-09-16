import tensorflow as tf

from util.model import Model


layers = tf.keras.layers


class MeanEmbeddingModel(Model):
    name = 'mean_embedding'

    def __init__(self, conf, extra_info):
        super(MeanEmbeddingModel, self).__init__(conf, extra_info)

        self.child_model = Model.create(conf['child_model'])
        self.embedding = layers.Embedding(extra_info['num_labels'], conf['child_model']['k'])

    def call(self, inputs, training=None, mask=None):
        return self.child_model(inputs, training, mask)

    def loss(self, images, labels):
        embeddings = self.call(images, training=True)
        mean_embeddings = self.embedding(labels - 1)
        return sum(tf.reduce_sum(tf.square(embeddings - mean_embeddings), axis=1))
