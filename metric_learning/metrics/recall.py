from util.registry.metric import Metric
from collections import defaultdict

import tensorflow as tf


def cosine_similarity(x, y, axis):
    x_norm = x / tf.norm(x, axis=len(x.shape) - 1, keep_dims=True)
    y_norm = y / tf.norm(y, axis=len(y.shape) - 1, keep_dims=True)
    return -tf.reduce_sum(tf.multiply(x_norm, y_norm), axis=axis)


def euclidean_distance(x, y, axis):
    return tf.norm(x - y, axis=axis)


def dot_product(x, y, axis):
    return -tf.reduce_sum(tf.multiply(x, y), axis=axis)


def evaluate_accuracy(func, anchor_embeddings, positive_embeddings, negative_embeddings):
    p = func(anchor_embeddings, positive_embeddings, axis=1)
    n = tf.reduce_min(func(negative_embeddings, anchor_embeddings, axis=2), axis=0)
    return p < n


class Recall(Metric):
    name = 'recall'
    dataset = 'recall'

    def compute_metric(self, model, test_ds):
        total = 0.
        successes = defaultdict(float)
        for images, labels in test_ds:
            all_labels = []
            embeddings = model(images, training=False)
            distance_blocks = []
            for test_images, test_labels in test_ds:
                all_labels += list(test_labels.numpy())
                test_embeddings = model(test_images, training=False)
                distances = tf.reduce_sum(
                    tf.square(embeddings[None] - test_embeddings[:, None]),
                    axis=2)
                distance_blocks.append(distances)
            tf.concat(distance_blocks, axis=1)
            values, indices = tf.nn.top_k(-tf.concat(distance_blocks, axis=1), max(self.conf['k']) + 1)
            top_labels = tf.gather(tf.constant(all_labels), indices)[:, 1:]
            for k in self.conf['k']:
                score = tf.reduce_sum(tf.cast(tf.equal(tf.transpose(labels[None]), top_labels[:, 0:k]), tf.int32), axis=1)
                successes[k] += int(sum(tf.cast(score >= 1, tf.int32)))
            total += int(images.shape[0])

        ret = {'recall@{}'.format(k): success / float(total) for k, success in successes.items()}
        return ret
