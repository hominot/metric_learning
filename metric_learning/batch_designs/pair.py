from util.registry.batch_design import BatchDesign

from collections import defaultdict
from metric_learning.constants.distance_function import DistanceFunction
from util.tensor_operations import stable_sqrt
from util.tensor_operations import pairwise_product
from util.tensor_operations import compute_elementwise_distances

import numpy as np
import tensorflow as tf
import random


class PairBatchDesign(BatchDesign):
    name = 'pair'

    def get_next_batch(self, image_files, labels, batch_conf):
        data_map = defaultdict(list)
        data_list = list(zip(image_files, labels))
        random.shuffle(data_list)
        for image_file, label in data_list:
            data_map[label].append(image_file)
        data_map = dict(filter(lambda x: len(x[1]) >= 2, data_map.items()))

        batch_size = batch_conf['batch_size']
        positive_ratio = batch_conf['positive_ratio']
        num_positive_pairs = int(batch_size * positive_ratio / 2)
        num_negative_pairs = (batch_size // 2) - num_positive_pairs
        label_match = [1] * num_positive_pairs + [0] * num_negative_pairs

        elements = []
        for match in label_match:
            if match:
                query_label = np.random.choice(list(data_map.keys()), size=1)[0]
                a = data_map[query_label].pop()
                b = data_map[query_label].pop()
                elements.append((a, query_label))
                elements.append((b, query_label))
                if len(data_map[query_label]) < 2:
                    del data_map[query_label]
            else:
                query_label, target_label = np.random.choice(
                    list(data_map.keys()), size=2, replace=False)
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

    def get_pairwise_distances(self, batch, model, distance_function, training=True):
        images, labels = batch
        embeddings = model(images, training=training)
        evens = tf.range(images.shape[0] // 2, dtype=tf.int64) * 2
        odds = tf.range(images.shape[0] // 2, dtype=tf.int64) * 2 + 1
        even_embeddings = tf.gather(embeddings, evens)
        odd_embeddings = tf.gather(embeddings, odds)
        even_labels = tf.gather(labels, evens)
        odd_labels = tf.gather(labels, odds)
        match = tf.equal(even_labels, odd_labels)

        elementwise_distances = compute_elementwise_distances(
            even_embeddings, odd_embeddings, distance_function
        )

        num_images = model.extra_info['num_images']
        num_labels = model.extra_info['num_labels']
        num_average_images_per_label = num_images / num_labels
        label_counts = tf.gather(
            tf.constant(model.extra_info['label_counts'], dtype=tf.float32),
            labels) / num_average_images_per_label
        even_label_counts = tf.gather(label_counts, evens)
        odd_label_counts = tf.gather(label_counts, odds)
        num_labels = model.extra_info['num_labels']
        label_counts_multiplied = tf.multiply(even_label_counts, odd_label_counts)
        positive_ratio = self.conf['batch_design']['positive_ratio']

        positive_weights = positive_ratio / even_label_counts / (even_label_counts - 1 / num_average_images_per_label)
        negative_weights = (1 - positive_ratio) / (num_labels - 1) / label_counts_multiplied
        weights = positive_weights * tf.cast(match, tf.float32) + negative_weights * tf.cast(~match, tf.float32)
        return elementwise_distances, match, 1 / weights

    def get_npair_distances(self, batch, model, n, distance_function, training=True):
        raise NotImplementedError

    def get_embeddings(self, batch, model, distance_function, training=True):
        raise NotImplementedError
