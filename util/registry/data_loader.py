from util.registry.class_registry import ClassRegistry
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

    def create_verification_test_dataset(self, image_files, labels):
        data = list(zip(image_files, labels))
        random.shuffle(data)
        data_map = defaultdict(list)
        for image_file, label in data:
            data_map[label].append(image_file)
        data_map = dict(filter(lambda x: len(x[1]) >= 2, data_map.items()))
        label_set = set(data_map.keys())
        anchor_images = []
        positive_images = []
        negative_images = [[]] * self.conf['test']['num_negative_examples']
        for label, images in data_map.items():
            for anchor_index, anchor_image in enumerate(data_map[label]):
                anchor_images.append(anchor_image)
                positive_index = random.choice(list(set(range(len(data_map[label]))) - {anchor_index}))
                positive_images.append(data_map[label][positive_index])
                negative_labels = random.sample(label_set - {label}, self.conf['test']['num_negative_examples'])
                for idx, negative_label in enumerate(negative_labels):
                    negative_images[idx].append(random.choice(data_map[negative_label]))
        anchor_images_ds = tf.data.Dataset.from_tensor_slices(tf.constant(anchor_images)) \
            .map(self._image_parse_function)
        positive_images_ds = tf.data.Dataset.from_tensor_slices(tf.constant(positive_images)) \
            .map(self._image_parse_function)
        negative_images_ds = [tf.data.Dataset.from_tensor_slices(tf.constant(x)).map(
            self._image_parse_function) for x in negative_images]
        return tf.data.Dataset.zip((anchor_images_ds, positive_images_ds, tuple(negative_images_ds)))

    def create_grouped_dataset(self, image_files, labels, group_size=2, num_groups=2, min_class_size=2):
        data = list(zip(image_files, labels))
        random.shuffle(data)

        data_map = defaultdict(list)
        for image_file, label in data:
            data_map[label].append(image_file)

        data_map = dict(filter(lambda x: len(x[1]) >= max(group_size, min_class_size), data_map.items()))
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
