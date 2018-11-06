from util.registry.metric import Metric

from collections import defaultdict
from tqdm import tqdm
from util.tensor_operations import pairwise_euclidean_distance_squared
from util.tensor_operations import pairwise_cosine_similarity

import math
import tensorflow as tf


def count_singletons(all_labels):
    counts = defaultdict(int)
    for label in all_labels:
        counts[label] += 1
    return len(list(filter(lambda x: x == 1, counts.values())))


def compute_recall(data, k_list, parametrization):
    successes = defaultdict(float)
    total = 0.
    num_singletons = 0
    for i, (embeddings, labels) in enumerate(tqdm(data, total=len(data), desc='recall', dynamic_ncols=True)):
        all_labels = []
        distance_blocks = []
        for j, (test_embeddings, test_labels) in enumerate(data):
            all_labels += list(test_labels.numpy())
            if parametrization == 'dot_product':
                distances = -pairwise_cosine_similarity(embeddings, test_embeddings)
            else:
                distances = pairwise_euclidean_distance_squared(embeddings, test_embeddings)
            if i == j:
                distances = distances + tf.eye(int(distances.shape[0])) * 1e6
            distance_blocks.append(distances)

        values, indices = tf.nn.top_k(-tf.concat(distance_blocks, axis=1), max(k_list))
        top_labels = tf.gather(tf.constant(all_labels, tf.int64), indices)
        for k in k_list:
            score = tf.reduce_sum(
                tf.cast(tf.equal(
                    tf.transpose(labels[None]),
                    top_labels[:, 0:k]
                ), tf.int32), axis=1)
            successes[k] += int(sum(tf.cast(score >= 1, tf.int32)))
        total += int(embeddings.shape[0])
        num_singletons = count_singletons(all_labels)
    return {k: success / float(total - num_singletons) for k, success in successes.items()}


class Recall(Metric):
    name = 'recall'

    def compute_metric(self, model, ds, num_testcases):
        batch_size = self.metric_conf['batch_size']
        ds = ds.batch(batch_size)
        data = []
        for images, labels in tqdm(ds, total=math.ceil(num_testcases / batch_size), desc='recall: embedding', dynamic_ncols=True):
            embeddings = model(images, training=False)
            data.append((embeddings, labels))

        ret = compute_recall(data, self.metric_conf['k'], self.conf['loss']['parametrization'])
        return {'recall@{}'.format(k): score for k, score in ret.items()}
