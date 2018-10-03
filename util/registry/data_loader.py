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

    def _random_crop(self, image):
        width = self.conf['image']['random_crop']['width']
        height = self.conf['image']['random_crop']['height']
        channel = self.conf['image']['channel']
        return tf.random_crop(image, [width, height, channel])

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

    def _repeat_label(self, label):
        n = self.conf['image']['random_crop']['n']
        return tf.data.Dataset.from_tensor_slices([label] * n)

    def create_dataset(self, image_files, labels):
        data = list(zip(image_files, labels))
        random.shuffle(data)
        image_files_shuffled, labels_shuffled = zip(*data)
        images_ds = tf.data.Dataset.from_tensor_slices(tf.constant(image_files_shuffled)) \
            .map(self._image_parse_function)
        labels_ds = tf.data.Dataset.from_tensor_slices(tf.constant(labels_shuffled))
        return tf.data.Dataset.zip((images_ds, labels_ds))

    def create_identification_test_dataset(self, image_files, labels):
        data = list(zip(image_files, labels))
        random.shuffle(data)
        data_map = defaultdict(list)
        for image_file, label in data:
            data_map[label].append(image_file)
        data_map = dict(filter(lambda x: len(x[1]) >= 2, data_map.items()))
        label_set = set(data_map.keys())
        labels = list(data_map.keys())
        anchor_images = []
        positive_images = []

        num_negative_examples = self.conf['dataset']['test']['identification']['num_negative_examples']
        negative_images = [[]] * num_negative_examples

        while len(anchor_images) < self.conf['dataset']['test']['identification']['num_testcases']:
            anchor_label = random.choice(labels)
            anchor_image, positive_image = random.sample(data_map[anchor_label], 2)
            anchor_images.append(anchor_image)
            positive_images.append(positive_image)
            negative_labels = random.choices(list(label_set - {anchor_label}), k=num_negative_examples)
            for idx, negative_label in enumerate(negative_labels):
                negative_images[idx].append(random.choice(data_map[negative_label]))
        anchor_images_ds = tf.data.Dataset.from_tensor_slices(tf.constant(anchor_images)).map(self._image_parse_function)
        positive_images_ds = tf.data.Dataset.from_tensor_slices(tf.constant(positive_images)).map(self._image_parse_function)
        negative_images_ds = [tf.data.Dataset.from_tensor_slices(tf.constant(x)).map(self._image_parse_function)
            for x in negative_images]

        if 'random_crop' in self.conf['image']:
            anchor_images_ds = anchor_images_ds.map(self._center_crop)
            positive_images_ds = positive_images_ds.map(self._center_crop)
            negative_images_ds = [x.map(self._center_crop) for x in negative_images_ds]
        return tf.data.Dataset.zip((anchor_images_ds, positive_images_ds, tuple(negative_images_ds)))

    def create_recall_test_dataset(self, image_files, labels):
        data = list(zip(image_files, labels))
        random.shuffle(data)
        data_map = defaultdict(list)
        for image_file, label in data:
            data_map[label].append(image_file)
        data_map = dict(filter(lambda x: len(x[1]) >= 5, data_map.items()))
        test_images = []
        test_labels = []
        num_examples_per_class = self.conf['dataset']['test']['recall']['num_examples_per_class']
        labels = list(data_map.keys())
        random.shuffle(labels)
        for label in labels:
            images = data_map[label]
            test_images += images[0:min(num_examples_per_class, len(images))]
            test_labels += [label] * min(num_examples_per_class, len(images))
            if len(test_labels) >= self.conf['dataset']['test']['recall']['num_testcases']:
                break
        test_images_ds = tf.data.Dataset.from_tensor_slices(tf.constant(test_images)).map(self._image_parse_function)
        test_labels_ds = tf.data.Dataset.from_tensor_slices(tf.constant(test_labels, tf.int64))

        if 'random_crop' in self.conf['image']:
            test_images_ds = test_images_ds.map(self._center_crop)
        return tf.data.Dataset.zip((test_images_ds, test_labels_ds))

    def create_grouped_dataset(self, image_files, labels, group_size=2, num_groups=2, min_class_size=2):
        data = list(zip(image_files, labels))
        if 'random_crop' in self.conf['image']:
            data = data * self.conf['image']['random_crop']['n']
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
        images_ds = tf.data.Dataset.from_tensor_slices(tf.constant(image_files_grouped)).map(self._image_parse_function)
        if 'random_crop' in self.conf['image']:
            images_ds = images_ds.map(self._random_crop)
        labels_ds = tf.data.Dataset.from_tensor_slices(tf.constant(labels_grouped))

        return tf.data.Dataset.zip((images_ds, labels_ds))

    def load_image_files(self):
        self.prepare_files()
        image_files, labels = create_dataset_from_directory(
            os.path.join(self.data_directory, self.name))
        return image_files, labels

    def __str__(self):
        return self.conf['dataset']['name']
