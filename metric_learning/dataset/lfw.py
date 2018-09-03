from util.dataset import download, extract
from util.data_loader import DataLoader
import os
import tensorflow as tf


def create_image_dataset(file_paths, labels):
    def _parse_function(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [250, 250])
        image_normalized = image_resized / 255.
        image_normalized = tf.reshape(image_normalized, [250, 250, 3])
        return image_normalized

    file_paths = tf.data.Dataset.from_tensor_slices(tf.constant(file_paths))\
        .map(_parse_function)
    labels = tf.data.Dataset.from_tensor_slices(tf.constant(labels))

    return tf.data.Dataset.zip((file_paths, labels))


class LFWDataLoader(DataLoader):
    name = 'lfw'

    def load_dataset(self):
        filepath = download('http://vis-www.cs.umass.edu/lfw/lfw.tgz')
        extract(filepath)
        label_map = {}
        labels = []
        image_files = []
        for subdir, dirs, files in os.walk('{}/lfw'.format(os.path.dirname(filepath))):
            for file in files:
                label = subdir.split('/')[-1]
                if label not in label_map:
                    label_map[label] = len(label_map) + 1
                image_files.append(os.path.join(subdir, file))
                labels.append(label_map[label])

        return create_image_dataset(image_files, labels)


if __name__ == '__main__':
    lfw = DataLoader.create('lfw')
    print(lfw.load_dataset())
