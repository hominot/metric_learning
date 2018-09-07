from util.class_registry import ClassRegistry
from util.dataset import create_dataset_from_directory

import os
import tensorflow as tf


class DataLoader(object, metaclass=ClassRegistry):
    name = None
    directory = None

    def data_path(self):
        return '/tmp/research/{}'.format(self.name)

    def prepare_files(self):
        pass

    def _image_parse_function(self, filename):
        pass

    def create_dataset(self, image_files, labels):
        images = tf.data.Dataset.from_tensor_slices(tf.constant(image_files)) \
            .map(self._image_parse_function)
        labels = tf.data.Dataset.from_tensor_slices(tf.constant(labels))
        return tf.data.Dataset.zip((images, labels))

    def load_image_files(self):
        self.prepare_files()

        image_files, labels = create_dataset_from_directory(
            os.path.join(self.directory, self.name))
        return image_files, labels

from metric_learning.datasets import *
