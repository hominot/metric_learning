from util.registry.metric import Metric

from tqdm import tqdm

import tensorflow as tf


def cosine_similarity(x, y, axis):
    x_norm = x / tf.norm(x, axis=len(x.shape) - 1, keep_dims=True)
    y_norm = y / tf.norm(y, axis=len(y.shape) - 1, keep_dims=True)
    return -tf.reduce_sum(tf.multiply(x_norm, y_norm), axis=axis)


def euclidean_distance(x, y, axis):
    return tf.norm(x - y, axis=axis)


def evaluate_accuracy(func, anchor_embeddings, positive_embeddings, negative_embeddings):
    p = func(anchor_embeddings, positive_embeddings, axis=1)
    n = tf.reduce_min(func(negative_embeddings, anchor_embeddings, axis=2), axis=0)
    return p < n


class Accuracy(Metric):
    name = 'accuracy'
    dataset = 'identification'

    metric_functions = {
        'euclidean_distance': euclidean_distance,
        'cosine_similarity': cosine_similarity,
        'dot_product': cosine_similarity,
    }

    def compute_metric(self, model, test_ds, num_testcases):
        total = 0.
        success_count = 0.
        positive_distance = 0.
        negative_distance = 0.
        num_batches = 0
        batch_size = self.metric_conf['batch_size']
        test_ds = test_ds.batch(batch_size).prefetch(batch_size)
        for anchor_images, positive_images, negative_images_group in tqdm(
                test_ds,
                total=num_testcases // batch_size,
                desc=self.name):
            anchor_embeddings = model(anchor_images, training=False)
            positive_embeddings = model(positive_images, training=False)
            negative_embeddings = tf.stack([model(negative_images, training=False) for negative_images in negative_images_group])
            func = self.metric_functions[self.conf['loss']['parametrization']]
            results = evaluate_accuracy(func, anchor_embeddings, positive_embeddings, negative_embeddings)
            success_count += float(sum(tf.cast(results, tf.float32)))
            positive_distance += float(tf.reduce_mean(tf.norm(anchor_embeddings - positive_embeddings, axis=1)))
            negative_distance += float(tf.reduce_mean(tf.reshape(tf.norm(anchor_embeddings - negative_embeddings, axis=2), [-1])))

            total += int(anchor_images.shape[0])
            num_batches += 1

        return {
            'accuracy': success_count / total,
            'positive_distance': positive_distance / num_batches,
            'negative_distance': negative_distance / num_batches,
        }
