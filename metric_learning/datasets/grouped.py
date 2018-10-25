from util.registry.dataset import Dataset

from collections import defaultdict

import tensorflow as tf
import random


class GroupedDataset(Dataset):
    name = 'grouped'

    def create_dataset(self, image_files, labels):
        data = list(zip(image_files, labels, range(len(image_files))))
        random.shuffle(data)

        group_size = self.conf['dataset']['dataset']['group_size']
        num_groups = self.conf['dataset']['dataset']['num_groups']
        data_map = defaultdict(list)
        for image_file, label, image_id in data:
            data_map[label].append((image_file, image_id))

        data_map = dict(filter(lambda x: len(x[1]) >= group_size, data_map.items()))
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
        images_ds = tf.data.Dataset.from_tensor_slices(
            tf.constant(image_files_grouped)
        ).map(self.data_loader.image_parse_function)
        if 'random_flip' in self.conf['image'] and self.conf['image']['random_flip']:
            images_ds = images_ds.map(self.data_loader.random_flip)
        if 'random_crop' in self.conf['image']:
            images_ds = images_ds.map(self.data_loader.random_crop)
        labels_ds = tf.data.Dataset.from_tensor_slices(tf.constant(labels_grouped))
        image_ids_ds = tf.data.Dataset.from_tensor_slices(tf.constant(image_ids_grouped))

        return tf.data.Dataset.zip((images_ds, labels_ds, image_ids_ds)), len(image_files_grouped)

