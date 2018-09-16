from util.class_registry import ClassRegistry
from util.dataset import create_dataset_from_directory
from collections import defaultdict

import os
import tensorflow as tf
import random


class DataLoader(object, metaclass=ClassRegistry):
    module_path = 'metric_learning.datasets'

    data_directory = '/tmp/research/data'
    temp_directory = '/tmp/research/temp'

    def __init__(self, conf, extra_info):
        super(DataLoader, self).__init__()
        self.conf = conf
        self.extra_info = extra_info

    def prepare_files(self):
        raise NotImplementedError

    def _image_parse_function(self, filename):
        raise NotImplementedError

    def create_dataset(self, image_files, labels):
        data = list(zip(image_files, labels))
        random.shuffle(data)
        image_files_shuffled, labels_shuffled = zip(*data)
        images_ds = tf.data.Dataset.from_tensor_slices(tf.constant(image_files_shuffled)) \
            .map(self._image_parse_function)
        labels_ds = tf.data.Dataset.from_tensor_slices(tf.constant(labels_shuffled))
        return tf.data.Dataset.zip((images_ds, labels_ds))

    def create_grouped_dataset(self, image_files, labels, group_size=2, num_groups=2):
        data = list(zip(image_files, labels))
        random.shuffle(data)

        data_map = defaultdict(list)
        for image_file, label in data:
            data_map[label].append(image_file)

        data_map = dict(filter(lambda x: len(x[1]) >= group_size, data_map.items()))
        grouped_data = []
        while len(data_map) >= num_groups:
            sampled_labels = random.sample(data_map.keys(), num_groups)
            for label in sampled_labels:
                for _ in range(group_size):
                    grouped_data.append((data_map[label].pop(), label))
                if len(data_map[label]) < group_size:
                    del data_map[label]

        image_files_grouped, labels_grouped = zip(*grouped_data)
        images_ds = tf.data.Dataset.from_tensor_slices(tf.constant(image_files_grouped)) \
            .map(self._image_parse_function)
        labels_ds = tf.data.Dataset.from_tensor_slices(tf.constant(labels_grouped))
        return tf.data.Dataset.zip((images_ds, labels_ds))

    def load_image_files(self):
        self.prepare_files()

        image_files, labels = create_dataset_from_directory(
            os.path.join(self.data_directory, self.name))
        return image_files, labels
