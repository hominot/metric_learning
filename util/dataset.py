from tqdm import tqdm

import math
import random
import requests
import tarfile
import tensorflow as tf
import os


def download(url, directory, filename=None):
    if filename is None:
        filename = url.split('/')[-1]
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0));
    block_size = 1024
    wrote = 0
    with open(filepath, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size//block_size) , unit='KB', unit_scale=True):
            wrote = wrote  + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print('ERROR, something went wrong')
    return filepath


def extract(filepath, directory):
    tar = tarfile.open(filepath, 'r')
    for item in tar:
        tar.extract(item, directory)


def create_dataset_from_directory(directory):
    label_map = {}
    labels = []
    image_files = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            label = subdir.split('/')[-1]
            if label not in label_map:
                label_map[label] = len(label_map) + 1
            image_files.append(os.path.join(subdir, file))
            labels.append(label_map[label])
    return image_files, labels


def split_train_test(images, labels, split_test_ratio=0.2):
    data = list(zip(images, labels))
    random.shuffle(data)
    num_test = int(split_test_ratio * len(data))
    training_data = data[:len(data) - num_test]
    testing_data = data[-num_test:]
    return training_data, testing_data
