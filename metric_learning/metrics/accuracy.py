from util.registry.metric import Metric
from collections import defaultdict

import random
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


class Accuracy(Metric):
    name = 'accuracy'

    metric_functions = {
        'accuracy:euclidean': euclidean_distance,
        'accuracy:cosine_similarity': cosine_similarity,
        'accuracy:dot_product': dot_product,
    }

    def compute_metric(self, model, test_ds):
        total = 0.
        success_counts = defaultdict(float)
        positive_distance = 0.
        negative_distance = 0.
        num_batches = 0
        for anchor_images, positive_images, negative_images_group in test_ds:
            if random.random() > self.conf.get('sampling_rate', 1.0):
                continue
            anchor_embeddings = model(anchor_images, training=False)
            positive_embeddings = model(positive_images, training=False)
            negative_embeddings = tf.stack([model(negative_images, training=False) for negative_images in negative_images_group])
            for metric, func in self.metric_functions.items():
                results = evaluate_accuracy(func, anchor_embeddings, positive_embeddings, negative_embeddings)
                success_counts[metric] += float(sum(tf.cast(results, tf.float32)))
            positive_distance += float(tf.reduce_mean(tf.norm(anchor_embeddings - positive_embeddings, axis=1)))
            negative_distance += float(tf.reduce_mean(tf.reshape(tf.norm(anchor_embeddings - negative_embeddings, axis=2), [-1])))

            total += int(anchor_images.shape[0])
            num_batches += 1

        ret = {metric: success / float(total) for metric, success in success_counts.items()}
        ret['positive_distance'] = positive_distance / num_batches
        ret['negative_distance'] = negative_distance / num_batches
        return ret
