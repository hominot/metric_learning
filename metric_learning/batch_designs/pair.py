from util.registry.batch_design import BatchDesign

from collections import defaultdict
from metric_learning.constants.distance_function import DistanceFunction
from util.tensor_operations import stable_sqrt

import numpy as np
import tensorflow as tf
import random


class PairBatchDesign(BatchDesign):
    name = 'pair'

    def get_next_batch(self, image_files, labels):
        data_map = defaultdict(list)
        data_list = list(zip(image_files, labels))
        random.shuffle(data_list)
        for image_file, label in data_list:
            data_map[label].append(image_file)
        data_map = dict(filter(lambda x: len(x[1]) >= 2, data_map.items()))

        batch_size = self.conf['batch_design']['batch_size']
        positive_ratio = self.conf['batch_design']['positive_ratio']
        num_positive_pairs = int(batch_size * positive_ratio / 2)
        num_negative_pairs = (batch_size // 2) - num_positive_pairs
        label_match = [1] * num_positive_pairs + [0] * num_negative_pairs
        random.shuffle(label_match)

        weights = [float(len(x)) for x in data_map.values()]
        p = np.array(weights) / sum(weights)

        elements = []
        for match in label_match:
            if match:
                query_label = np.random.choice(list(data_map.keys()), size=1, p=p)[0]
                a = data_map[query_label].pop()
                b = data_map[query_label].pop()
                elements.append((a, query_label))
                elements.append((b, query_label))
                if len(data_map[query_label]) < 2:
                    del data_map[query_label]
            else:
                query_label, target_label = np.random.choice(
                    list(data_map.keys()), size=2, replace=False, p=p)
                elements.append(
                    (data_map[query_label].pop(), query_label)
                )
                elements.append(
                    (data_map[target_label].pop(), target_label)
                )
                if len(data_map[query_label]) < 2:
                    del data_map[query_label]
                if len(data_map[target_label]) < 2:
                    del data_map[target_label]
        return elements

    def get_pairwise_distances(self, batch, model, distance_function):
        images, labels = batch
        embeddings = model(images, training=True)
        evens = tf.range(images.shape[0] // 2, dtype=tf.int64) * 2
        odds = tf.range(images.shape[0] // 2, dtype=tf.int64) * 2 + 1
        even_embeddings = tf.gather(embeddings, evens)
        odd_embeddings = tf.gather(embeddings, odds)
        even_labels = tf.gather(labels, evens)
        odd_labels = tf.gather(labels, odds)
        match = tf.equal(even_labels, odd_labels)
        if distance_function == DistanceFunction.EUCLIDEAN_DISTANCE:
            pairwise_distances = stable_sqrt(
                tf.reduce_sum(tf.square(even_embeddings - odd_embeddings), axis=1))
        elif distance_function == DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED:
            pairwise_distances = tf.reduce_sum(
                tf.square(even_embeddings - odd_embeddings), axis=1)
        elif distance_function == DistanceFunction.DOT_PRODUCT:
            pairwise_distances = -tf.reduce_sum(
                tf.multiply(even_embeddings, odd_embeddings), axis=1)
        else:
            raise Exception('Unknown distance function: {}'.format(distance_function))
        return pairwise_distances, match, None

    def get_npair_distances(self, batch, model, n, distance_function):
        raise NotImplementedError

    def get_embeddings(self, batch, model, distance_function):
        raise NotImplementedError
