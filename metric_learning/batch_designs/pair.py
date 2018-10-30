from util.registry.batch_design import BatchDesign

from collections import defaultdict
from metric_learning.constants.distance_function import DistanceFunction
from util.tensor_operations import stable_sqrt

import numpy as np
import tensorflow as tf
import random


class PairBatchDesign(BatchDesign):
    name = 'pair'

    def _create_datasets_from_elements(self, elements):
        image_files, labels = zip(*elements)
        images_ds = tf.data.Dataset.from_tensor_slices(
            tf.constant(image_files)
        ).map(self.data_loader.image_parse_function)
        if 'random_flip' in self.conf['image'] and self.conf['image']['random_flip']:
            images_ds = images_ds.map(self.data_loader.random_flip)
        if 'random_crop' in self.conf['image']:
            images_ds = images_ds.map(self.data_loader.random_crop)
        labels_ds = tf.data.Dataset.from_tensor_slices(tf.constant(labels))
        return images_ds, labels_ds

    def get_next_batch(self, image_files, labels):
        data_map = defaultdict(list)
        data_list = list(zip(image_files, labels))
        random.shuffle(data_list)
        for image_file, label in data_list:
            data_map[label].append(image_file)
        data_map = dict(filter(lambda x: len(x[1]) >= 2, data_map.items()))

        batch_size = self.conf['batch_design']['batch_size']
        positive_ratio = self.conf['batch_design']['positive_ratio']
        num_positive_pairs = int(batch_size * positive_ratio)
        num_negative_pairs = batch_size - num_positive_pairs
        label_match = [1] * num_positive_pairs + [0] * num_negative_pairs
        random.shuffle(label_match)

        weights = [float(len(x)) for x in data_map.values()]
        p = np.array(weights) / sum(weights)

        first_elements = []
        second_elements = []
        for idx in range(batch_size):
            if label_match[idx]:
                query_label = np.random.choice(list(data_map.keys()), size=1, p=p)[0]
                a = data_map[query_label].pop()
                b = data_map[query_label].pop()
                first_elements.append((a, query_label))
                second_elements.append((b, query_label))
                if len(data_map[query_label]) < 2:
                    del data_map[query_label]
            else:
                query_label, target_label = np.random.choice(
                    list(data_map.keys()), size=2, replace=False, p=p)
                first_elements.append(
                    (data_map[query_label].pop(), query_label)
                )
                second_elements.append(
                    (data_map[target_label].pop(), target_label)
                )
                if len(data_map[query_label]) < 2:
                    del data_map[query_label]
                if len(data_map[target_label]) < 2:
                    del data_map[target_label]
        return first_elements, second_elements

    def create_dataset(self, image_files, labels, testing=False):
        first_data = []
        second_data = []
        for _ in range(self.conf['batch_design']['num_batches']):
            first_elements, second_elements = self.get_next_batch(
                image_files, labels)
            first_data += first_elements
            second_data += second_elements

        return tf.data.Dataset.zip((
            self._create_datasets_from_elements(first_data),
            self._create_datasets_from_elements(second_data)
        )), len(first_data)

    def get_pairwise_distances(self, batch, model, distance_function):
        (first_images, first_labels), (second_images, second_labels) = batch
        matching_labels = tf.equal(first_labels, second_labels)
        first_embeddings = model(first_images, training=True)
        second_embeddings = model(second_images, training=True)
        if distance_function == DistanceFunction.EUCLIDEAN_DISTANCE:
            pairwise_distances = stable_sqrt(
                tf.reduce_sum(tf.square(first_embeddings - second_embeddings), axis=1))
        elif distance_function == DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED:
            pairwise_distances = tf.reduce_sum(
                tf.square(first_embeddings - second_embeddings), axis=1)
        elif distance_function == DistanceFunction.DOT_PRODUCT:
            pairwise_distances = -tf.reduce_sum(
                tf.multiply(first_embeddings, second_embeddings), axis=1)
        else:
            raise Exception('Unknown distance function: {}'.format(distance_function))
        return pairwise_distances, matching_labels

    def get_npair_distances(self, batch, model, distance_function):
        raise NotImplementedError

    def get_embeddings(self, batch, model, distance_function):
        raise NotImplementedError
