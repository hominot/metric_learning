import tensorflow as tf
import numpy as np

from collections import defaultdict
from util.loss_function import LossFunction


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


class NPairLossFunction(LossFunction):
    name = 'npair'

    def loss(self, embeddings, labels):
        sampled_data = sample_npair(embeddings, labels, self.conf['n'])
        losses = []
        for first_images, second_images in sampled_data:
            b = 1 - np.eye(int(first_images.shape[0]))
            difference = tf.reshape(
                tf.boolean_mask(
                    tf.exp(
                        tf.matmul(first_images, tf.transpose(second_images)) - \
                        tf.reduce_sum(tf.multiply(first_images, second_images), axis=1, keepdims=True)
                    ),
                    b
                ),
                (self.conf['n'], self.conf['n'] - 1)
            )
            if self.conf.get('ovo', False):
                loss = tf.reduce_sum(tf.log(1 + difference), axis=1)
            else:
                loss = tf.log(1 + tf.reduce_sum(difference, axis=1))
            losses.append(tf.reduce_mean(loss))
        return sum(losses)
