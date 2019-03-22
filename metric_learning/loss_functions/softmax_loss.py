import tensorflow as tf

from metric_learning.constants.distance_function import DistanceFunction
from util.registry.loss_function import LossFunction
from util.tensor_operations import stable_sqrt


class SoftmaxLossFunction(LossFunction):
    name = 'softmax'

    def loss(self, batch, model, dataset):
        images, labels = batch
        embeddings = model(images, training=True)
        return tf.losses.softmax_cross_entropy(tf.one_hot(labels, embeddings.shape[1]), embeddings)
