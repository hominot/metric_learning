from util.registry.metric import Metric
from collections import defaultdict

import tensorflow as tf


def cosine_similarity(x, y, axis):
    x_norm = x / tf.norm(x, axis=axis, keep_dims=True)
    y_norm = y / tf.norm(y, axis=axis, keep_dims=True)
    return tf.reduce_sum(tf.multiply(x_norm, y_norm), axis=axis)


class Accuracy(Metric):
    name = 'accuracy'

    metric_functions = {
        'accuracy:euclidean': lambda x, y, axis: tf.norm(x - y, axis=axis),
        'accuracy:cosine_similarity': cosine_similarity,
        'accuracy:dot_product': lambda x, y, axis:
            tf.reduce_sum(tf.multiply(x, y), axis=axis),
    }

    def compute_metric(self, model, test_ds, *args, **kwargs):

        total = 0.
        success_counts = defaultdict(float)
        for anchor_images, positive_images, negative_images_group in test_ds:
            anchor_embeddings = model(anchor_images, training=False)
            positive_embeddings = model(positive_images, training=False)
            negative_embeddings = tf.stack([model(negative_images) for negative_images in negative_images_group])
            for metric, func in self.metric_functions.items():
                p = func(anchor_embeddings, positive_embeddings, axis=1)
                n = tf.reduce_min(tf.norm(negative_embeddings - anchor_embeddings, axis=2), axis=0)
                success_counts[metric] += float(sum(tf.cast(p < n, tf.float32)))

            total += int(anchor_images.shape[0])

        return {metric: success / float(total) for metric, success in success_counts.items()}
