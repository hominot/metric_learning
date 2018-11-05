from util.registry.batch_design import BatchDesign

from collections import defaultdict
from metric_learning.constants.distance_function import DistanceFunction
from util.tensor_operations import pairwise_euclidean_distance_squared
from util.tensor_operations import pairwise_euclidean_distance
from util.tensor_operations import pairwise_dot_product
from util.tensor_operations import pairwise_matching_matrix
from util.tensor_operations import upper_triangular_part
from util.tensor_operations import get_n_blocks
from util.tensor_operations import pairwise_product

import numpy as np
import tensorflow as tf
import random


def get_npair_distances(embeddings, n, distance_function):
    num_groups = int(embeddings.shape[0]) // 2
    evens = tf.range(num_groups, dtype=tf.int64) * 2
    odds = tf.range(num_groups, dtype=tf.int64) * 2 + 1
    even_embeddings = tf.gather(embeddings, evens)
    odd_embeddings = tf.gather(embeddings, odds)

    if distance_function == DistanceFunction.EUCLIDEAN_DISTANCE:
        pairwise_distances = pairwise_euclidean_distance(even_embeddings, odd_embeddings)
    elif distance_function == DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED:
        pairwise_distances = pairwise_euclidean_distance_squared(even_embeddings, odd_embeddings)
    elif distance_function == DistanceFunction.DOT_PRODUCT:
        pairwise_distances = -pairwise_dot_product(even_embeddings, odd_embeddings)
    else:
        raise Exception('Unknown distance function: {}'.format(distance_function))

    return (
        get_n_blocks(pairwise_distances, n),
        get_n_blocks(tf.cast(tf.eye(num_groups), tf.bool), n)
    )


class GroupedBatchDesign(BatchDesign):
    name = 'grouped'

    def get_next_batch(self, image_files, labels):
        data = list(zip(image_files, labels))
        random.shuffle(data)

        batch_size = self.conf['batch_design']['batch_size']
        group_size = self.conf['batch_design']['group_size']
        num_groups = batch_size // group_size
        data_map = defaultdict(list)
        for image_file, label in data:
            data_map[label].append(image_file)

        data_map = dict(filter(lambda x: len(x[1]) >= group_size, data_map.items()))
        sampled_labels = np.random.choice(
            list(data_map.keys()), size=num_groups, replace=False)
        grouped_data = []
        for label in sampled_labels:
            for _ in range(group_size):
                image_file = data_map[label].pop()
                grouped_data.append((image_file, label))
        return grouped_data

    def get_pairwise_distances(self, batch, model, distance_function):
        images, labels = batch
        embeddings = model(images, training=True)

        num_images = model.extra_info['num_images']
        num_labels = model.extra_info['num_labels']
        num_average_images_per_label = num_images / num_labels
        batch_size = self.conf['batch_design']['batch_size']
        group_size = self.conf['batch_design']['group_size']
        num_groups = batch_size // group_size

        if self.conf['batch_design'].get('npair'):
            evens = tf.range(num_groups, dtype=tf.int64) * 2
            even_labels = tf.gather(labels, evens)
            num_average_images_per_label = num_images / num_labels
            label_counts = tf.gather(
                tf.constant(model.extra_info['label_counts'], dtype=tf.float32),
                even_labels) / num_average_images_per_label
            positive_label_counts = tf.stack([label_counts, label_counts], axis=1)
            label_counts_multiplied = get_n_blocks(
                pairwise_product(label_counts, label_counts),
                self.conf['batch_design']['npair'])
            pairwise_distances, matching_labels_matrix = get_npair_distances(
                embeddings, self.conf['batch_design']['npair'], distance_function)
        else:
            if distance_function == DistanceFunction.EUCLIDEAN_DISTANCE:
                pairwise_distances = pairwise_euclidean_distance(embeddings, embeddings)
            elif distance_function == DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED:
                pairwise_distances = pairwise_euclidean_distance_squared(embeddings, embeddings)
            elif distance_function == DistanceFunction.DOT_PRODUCT:
                pairwise_distances = -pairwise_dot_product(embeddings, embeddings)
            else:
                raise Exception('Unknown distance function: {}'.format(distance_function))
            label_counts = tf.gather(
                tf.constant(model.extra_info['label_counts'], dtype=tf.float32),
                labels) / num_average_images_per_label
            positive_label_counts = label_counts
            matching_labels_matrix = pairwise_matching_matrix(labels, labels)
            label_counts_multiplied = pairwise_product(label_counts, label_counts)

        num_labels = model.extra_info['num_labels']
        negative_weights = (num_groups - 1) * group_size / (num_labels - 1) / label_counts_multiplied
        positive_weights = (group_size - 1) / positive_label_counts / (positive_label_counts - 1 / num_average_images_per_label)
        weights = positive_weights * tf.cast(matching_labels_matrix, tf.float32) + negative_weights * tf.cast(~matching_labels_matrix, tf.float32)

        if self.conf['batch_design'].get('npair'):
            return (
                tf.reshape(pairwise_distances, [-1]),
                tf.reshape(matching_labels_matrix, [-1]),
                tf.reshape(1 / weights, [-1]),
            )
        else:
            return (
                upper_triangular_part(pairwise_distances),
                upper_triangular_part(matching_labels_matrix),
                upper_triangular_part(1 / weights),
            )

    def get_npair_distances(self, batch, model, n, distance_function):
        if self.conf['batch_design']['group_size'] != 2:
            raise Exception('group size must be 2 in order to get npair distances')
        if (self.conf['batch_design']['batch_size'] // 2) % n != 0:
            raise Exception(
                'n does not divide the number of groups: n={}, num_groups={}'.format(
                    n, self.conf['batch_size'] // 2
                ))

        images, labels = batch
        embeddings = model(images, training=True)

        return get_npair_distances(embeddings, n, distance_function)

    def get_embeddings(self, batch, model, distance_function):
        images, _ = batch
        return model(images, training=True)
