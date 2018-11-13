from util.registry.metric import Metric

from collections import defaultdict
from tqdm import tqdm
from util.tensor_operations import compute_pairwise_distances
from metric_learning.constants.distance_function import get_distance_function

import tensorflow as tf


def count_singletons(all_labels):
    counts = defaultdict(int)
    for label in all_labels:
        counts[label] += 1
    return len(list(filter(lambda x: x == 1, counts.values())))


def compute_recall(embeddings_list, labels_list, k_list, distance_function):
    successes = defaultdict(float)
    embeddings = tf.concat(embeddings_list, axis=0)
    labels = tf.concat(labels_list, axis=0)
    pairwise_distances = compute_pairwise_distances(
        embeddings, embeddings, distance_function)
    pairwise_distances = pairwise_distances + \
                         tf.eye(int(pairwise_distances.shape[0])) * 1e6
    values, indices = tf.nn.top_k(-pairwise_distances, max(k_list))
    top_labels = tf.gather(tf.constant(labels, tf.int64), indices)
    for k in k_list:
        score = tf.reduce_sum(
        tf.cast(tf.equal(
            tf.transpose(labels[None]),
            top_labels[:, 0:k]
        ), tf.int32), axis=1)
        successes[k] += int(sum(tf.cast(score >= 1, tf.int32)))
    num_singletons = count_singletons(labels.numpy().tolist())
    return {k: success / float(int(labels.shape[0]) - num_singletons) for k, success in successes.items()}


class Recall(Metric):
    name = 'recall'

    def compute_metric(self, model, ds, num_testcases):
        embeddings_list, labels_list = self.get_embeddings(model, ds, num_testcases)

        ret = compute_recall(
            embeddings_list,
            labels_list,
            self.metric_conf['k'],
            get_distance_function(self.conf['loss']['distance_function']))
        return {'recall@{}'.format(k): score for k, score in ret.items()}
