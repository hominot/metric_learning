from util.registry.metric import Metric
from collections import defaultdict
from tqdm import tqdm

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

    def compute_metric(self, model, dataset, num_testcases):
        images_ds, labels_ds = dataset
        total = 0.
        successes = defaultdict(float)
        batch_size = self.metric_conf['batch_size']
        ds = tf.data.Dataset.zip((images_ds, labels_ds)).batch(batch_size)
        data = []
        for images, labels in tqdm(ds, total=num_testcases // batch_size, desc='recall: embedding'):
            embeddings = model(images, training=False)
            data.append((embeddings, labels))

        for embeddings, labels in tqdm(
                data, total=len(data), desc=self.name):
            all_labels = []
            distance_blocks = []
            for test_embeddings, test_labels in data:
                all_labels += list(test_labels.numpy())
                distances = tf.reduce_sum(
                    tf.square(test_embeddings[None] - embeddings[:, None]),
                    axis=2)
                distance_blocks.append(distances)

            values, indices = tf.nn.top_k(-tf.concat(distance_blocks, axis=1), max(self.metric_conf['k']) + 1)
            top_labels = tf.gather(tf.constant(all_labels, tf.int64), indices)[:, 1:]
            for k in self.metric_conf['k']:
                score = tf.reduce_sum(tf.cast(tf.equal(tf.transpose(labels[None]), top_labels[:, 0:k]), tf.int32), axis=1)
                successes[k] += int(sum(tf.cast(score >= 1, tf.int32)))
            total += int(embeddings.shape[0])

        ret = {'recall@{}'.format(k): success / float(total) for k, success in successes.items()}
        return ret
