from util.registry.data_loader import DataLoader
from util.dataset import extract_tgz
from util.dataset import download
from util.config import CONFIG

import tensorflow as tf
import os


class Cub200DataLoader(DataLoader):
    name = 'cub200'

    def prepare_files(self):
        data_directory = os.path.join(CONFIG['dataset']['data_dir'], self.name)
        if tf.gfile.Exists(data_directory):
            count = 0
            for root, dirnames, filenames in os.walk(data_directory):
                for filename in filenames:
                    if filename.lower().endswith('.png'):
                        count += 1
            if count == 11788:
                return
        filepath = download(
            'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz',
            os.path.join(CONFIG['dataset']['temp_dir'], self.name)
        )
        extract_tgz(filepath, os.path.join(CONFIG['dataset']['temp_dir'], self.name))

        extract_path = os.path.join(CONFIG['dataset']['temp_dir'], self.name, 'CUB_200_2011')
        data_directory = os.path.join(CONFIG['dataset']['data_dir'])
        train_data_directory = os.path.join(data_directory, self.name, 'train')
        test_data_directory = os.path.join(data_directory, self.name, 'test')
        if not tf.gfile.Exists(train_data_directory):
            tf.gfile.MakeDirs(train_data_directory)
        if not tf.gfile.Exists(test_data_directory):
            tf.gfile.MakeDirs(test_data_directory)

        bounding_boxes = {}
        with open(os.path.join(extract_path, 'bounding_boxes.txt'), 'r') as f:
            for line in f:
                image_id, x, y, width, height = line.rstrip().split(' ')
                bounding_boxes[int(image_id)] = (
                    int(float(x)),
                    int(float(y)),
                    int(float(width)),
                    int(float(height)))

        image_ids = {}
        with open(os.path.join(extract_path, 'images.txt'), 'r') as f:
            for line in f:
                image_id, filename = line.rstrip().split(' ')
                image_ids[filename.split('/')[1]] = int(image_id)

        for root, dirnames, filenames in os.walk(os.path.join(extract_path, 'images')):
            for dirname in dirnames:
                if not tf.gfile.Exists(os.path.join(root, dirname)):
                    tf.gfile.MakeDirs(os.path.join(root, dirname))
            for filename in filenames:
                full_file_path = str(os.path.join(root, filename))
                image_string = tf.read_file(full_file_path)
                image_decoded = tf.image.decode_jpeg(image_string, channels=3)
                image_id = image_ids[filename]
                x, y, width, height = bounding_boxes[image_id]
                image_width = int(image_decoded.shape[1])
                image_height = int(image_decoded.shape[0])
                image_cropped = tf.image.crop_to_bounding_box(
                    image_decoded,
                    y, x,
                    min(image_height - y, height), min(image_width - x, width))
                image_encoded = tf.image.encode_png(image_cropped)
                png_filename = '{}.png'.format(filename.rsplit('.', 1)[0])

                class_name = str(root).split('/')[-1]
                if int(class_name.split('.')[0]) <= 100:
                    tf.write_file(os.path.join(train_data_directory, class_name, png_filename), image_encoded)
                else:
                    tf.write_file(os.path.join(test_data_directory, class_name, png_filename), image_encoded)

    def _image_parse_function(self, filename):
        width = self.conf['image']['width']
        height = self.conf['image']['height']
        channel = self.conf['image']['channel']
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=channel)
        image_resized = tf.image.resize_image_with_pad(image_decoded, target_height=height, target_width=width)
        image_normalized = (image_resized / 255. - 0.5) * 2
        image_normalized = tf.reshape(image_normalized, [width, height, channel])
        return image_normalized
