from util.dataset import download, extract
from util.data_loader import DataLoader

import tensorflow as tf
import os


class LFWDataLoader(DataLoader):
    name = 'lfw'

    def prepare_files(self):
        data_directory = os.path.join(self.data_directory, self.name)
        if tf.gfile.Exists(data_directory):
            count = 0
            for root, dirnames, filenames in os.walk(data_directory):
                for filename in filenames:
                    if filename.endswith('.jpg'):
                        count += 1
            if count == 13233:
                return
        filepath = download(
            'http://vis-www.cs.umass.edu/lfw/lfw.tgz',
            os.path.join(self.temp_directory, self.name)
        )
        extract(filepath, self.data_directory)

    def _image_parse_function(self, filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [250, 250])
        image_normalized = image_resized / 255.
        image_normalized = tf.reshape(image_normalized, [250, 250, 3])
        return image_normalized
