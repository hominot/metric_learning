from util.registry.class_registry import ClassRegistry
from util.dataset import load_images_from_directory
from collections import defaultdict
from util.config import CONFIG

import os
import tensorflow as tf
import random


class DataLoader(object, metaclass=ClassRegistry):
    module_path = 'metric_learning.datasets'

    def __init__(self, conf, extra_info):
        super(DataLoader, self).__init__()
        self.conf = conf
        self.extra_info = extra_info

    def prepare_files(self):
        raise NotImplementedError

    def _image_parse_function(self, filename):
        raise NotImplementedError

    def _random_crop(self, image):
        width = self.conf['image']['random_crop']['width']
        height = self.conf['image']['random_crop']['height']
        channel = self.conf['image']['channel']
        return tf.random_crop(image, [width, height, channel])

    def _random_flip(self, image):
        return tf.image.random_flip_left_right(image)

    def _center_crop(self, image):
        crop_width = self.conf['image']['random_crop']['width']
        crop_height = self.conf['image']['random_crop']['height']
        width = self.conf['image']['width']
        height = self.conf['image']['height']
        return tf.image.crop_to_bounding_box(
            image,
            (width - crop_width) // 2,
            (height - crop_height) // 2,
            crop_width,
            crop_height)

    def _parse_and_augment(self, dataset: tf.data.Dataset):
        dataset = dataset.map(self._image_parse_function)
        if 'random_crop' in self.conf['image']:
            dataset = dataset.flat_map(self._random_crop)
        return dataset

    def _parse_and_center(self, dataset: tf.data.Dataset):
        dataset = dataset.map(self._image_parse_function)
        if 'random_crop' in self.conf['image']:
            dataset = dataset.map(self._center_crop)
        return dataset

    def create_grouped_dataset(self, image_files, labels, group_size=2, num_groups=2, min_class_size=2):
        data = list(zip(image_files, labels, range(len(image_files))))
        random.shuffle(data)

        data_map = defaultdict(list)
        for image_file, label, image_id in data:
            data_map[label].append((image_file, image_id))

        data_map = dict(filter(lambda x: len(x[1]) >= max(group_size, min_class_size), data_map.items()))
        grouped_data = []
        while len(data_map) >= num_groups:
            sampled_labels = random.sample(data_map.keys(), num_groups)
            for label in sampled_labels:
                for _ in range(group_size):
                    image_file, image_id = data_map[label].pop()
                    grouped_data.append((image_file, label, image_id))
                if len(data_map[label]) < group_size:
                    del data_map[label]

        image_files_grouped, labels_grouped, image_ids_grouped = zip(*grouped_data)
        images_ds = tf.data.Dataset.from_tensor_slices(tf.constant(image_files_grouped)).map(self._image_parse_function)
        if 'random_flip' in self.conf['image'] and self.conf['image']['random_flip']:
            images_ds = images_ds.map(self._random_flip)
        if 'random_crop' in self.conf['image']:
            images_ds = images_ds.map(self._random_crop)
        labels_ds = tf.data.Dataset.from_tensor_slices(tf.constant(labels_grouped))
        image_ids_ds = tf.data.Dataset.from_tensor_slices(tf.constant(image_ids_grouped))

        return tf.data.Dataset.zip((images_ds, labels_ds, image_ids_ds)), len(image_files_grouped)

    def load_image_files(self):
        self.prepare_files()
        image_files, labels = load_images_from_directory(
            os.path.join(CONFIG['dataset']['data_dir'], self.name))
        return image_files, labels

    def __str__(self):
        return self.conf['dataset']['name']
