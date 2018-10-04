import tensorflow as tf

from collections import defaultdict
from util.registry.loss_function import LossFunction


def sample_npair(images, labels, n):
    ret = []
    for batch in range(int(images.shape[0]) // (n * 2)):
        cur_labels = labels[batch * n * 2: (batch + 1) * n * 2]
        cur_images = images[batch * n * 2: (batch + 1) * n * 2]
        data_map = defaultdict(list)
        for index, label in enumerate(cur_labels):
            data_map[int(label)].append(index)

        first_images = []
        second_images = []
        for label in data_map.keys():
            first_images.append(cur_images[data_map[label][0]])
            second_images.append(cur_images[data_map[label][1]])
        ret.append((
            tf.stack(first_images),
            tf.stack(second_images),
        ))
    return ret


def compute_dot_product_exponents(first_images, second_images):
    exponents = tf.matmul(first_images, tf.transpose(second_images)) - \
           tf.reduce_sum(tf.multiply(first_images, second_images), axis=1, keepdims=True)
    off_diagonals = 1 - tf.eye(int(first_images.shape[0]))
    eta = tf.reshape(tf.boolean_mask(exponents, off_diagonals), (int(first_images.shape[0]), -1))
    return tf.concat([tf.zeros((eta.shape[0], 1)), eta], axis=1)


def compute_euclidean_distance_exponents(first_images, second_images):
    exponents = tf.reduce_sum(tf.square(first_images - second_images), axis=1, keepdims=True) - \
                tf.reduce_sum(
                    tf.square(first_images[None] - second_images[:, None]),
                    axis=2)
    off_diagonals = 1 - tf.eye(int(first_images.shape[0]))
    eta = tf.reshape(tf.boolean_mask(exponents, off_diagonals), (int(first_images.shape[0]), -1))
    return tf.concat([tf.zeros((eta.shape[0], 1)), eta], axis=1)


class NPairLossFunction(LossFunction):
    name = 'npair'

    def __init__(self, conf, extra_info):
        super(NPairLossFunction, self).__init__(conf, extra_info)
        self.compute_exponents = compute_euclidean_distance_exponents \
            if conf['model']['loss']['parametrization'] == 'euclidean_distance' \
            else compute_dot_product_exponents

    def loss(self, embeddings, labels):
        sampled_data = sample_npair(embeddings, labels, self.conf['model']['loss']['n'])
        losses = []
        for first_images, second_images in sampled_data:
            loss = tf.reduce_logsumexp(self.compute_exponents(first_images, second_images), axis=1)
            losses.append(tf.reduce_mean(loss))
        return sum(losses)
