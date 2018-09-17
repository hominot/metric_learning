import tensorflow as tf

from util.model import Model


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

        embedding_loss = 0
        for i in labels:
            for j in labels:
                if i != j:
                    e_i = self.embedding(i)
                    e_j = self.embedding(j)
                    d = tf.reduce_sum(tf.square(e_i - e_j))
                    embedding_loss += tf.maximum(0, 1 - d)
        return sum(tf.reduce_sum(tf.square(embeddings - mean_embeddings), axis=1)) + embedding_loss / 16.0
