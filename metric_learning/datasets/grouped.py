from util.registry.dataset import Dataset

from collections import defaultdict
from metric_learning.constants.distance_function import DistanceFunction
from util.tensor_operations import upper_triangular_part
from util.tensor_operations import pairwise_euclidean_distance_squared
from util.tensor_operations import pairwise_euclidean_distance
from util.tensor_operations import pairwise_dot_product
from util.tensor_operations import pairwise_matching_matrix

import numpy as np
import tensorflow as tf
import random


class GroupedDataset(Dataset):
    name = 'grouped'

    def _get_next_batch(self, image_files, labels):
        data = list(zip(image_files, labels))
        random.shuffle(data)

        group_size = self.conf['dataset']['dataset']['group_size']
        num_groups = self.conf['dataset']['dataset']['num_groups']
        data_map = defaultdict(list)
        for image_file, label in data:
            data_map[label].append(image_file)

        data_map = dict(filter(lambda x: len(x[1]) >= group_size, data_map.items()))
        weights = [float(len(x)) for x in data_map.values()]
        p = np.array(weights) / sum(weights)

        sampled_labels = np.random.choice(
            list(data_map.keys()), size=num_groups, replace=False, p=p)
        grouped_data = []
        for label in sampled_labels:
            for _ in range(group_size):
                image_file = data_map[label].pop()
                grouped_data.append((image_file, label))
        return grouped_data

    def create_dataset(self, image_files, labels, testing=False):
        grouped_data = []
        for _ in range(self.conf['dataset']['dataset']['num_batches']):
            grouped_data += self._get_next_batch(image_files, labels)
        image_files_grouped, labels_grouped = zip(*grouped_data)
        images_ds = tf.data.Dataset.from_tensor_slices(
            tf.constant(image_files_grouped)
        ).map(self.data_loader.image_parse_function)
        if 'random_flip' in self.conf['image'] and self.conf['image']['random_flip']:
            images_ds = images_ds.map(self.data_loader.random_flip)
        if 'random_crop' in self.conf['image']:
            images_ds = images_ds.map(self.data_loader.random_crop)
        labels_ds = tf.data.Dataset.from_tensor_slices(tf.constant(labels_grouped))

        return tf.data.Dataset.zip((images_ds, labels_ds)), len(image_files_grouped)

    def get_pairwise_distances(self, batch, model, distance_function):
        images, labels = batch
        embeddings = model(images, training=True)
        if distance_function == DistanceFunction.EUCLIDEAN_DISTANCE:
            pairwise_distances = upper_triangular_part(pairwise_euclidean_distance(embeddings, embeddings))
        elif distance_function == DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED:
            pairwise_distances = upper_triangular_part(pairwise_euclidean_distance_squared(embeddings, embeddings))
        elif distance_function == DistanceFunction.DOT_PRODUCT:
            pairwise_distances = upper_triangular_part(-pairwise_dot_product(embeddings, embeddings))
        else:
            raise Exception('Unknown distance function: {}'.format(distance_function))

        matching_labels_matrix = tf.cast(
            upper_triangular_part(tf.cast(pairwise_matching_matrix(labels, labels), tf.int64)),
            tf.bool)

        return pairwise_distances, matching_labels_matrix

    def get_npair_distances(self, batch, model, distance_function):
        raise Exception('grouped dataset does not support npair distances')
