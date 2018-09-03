from collections import defaultdict
from tqdm import tqdm

import math
import random
import requests
import tarfile
import tensorflow as tf
import os


def download(url, directory='/tmp/research', filename=None):
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


def extract(filepath):
    extract_path = os.path.dirname(filepath)
    tar = tarfile.open(filepath, 'r')
    for item in tar:
        tar.extract(item, extract_path)


def sample_pairs(images, labels, num_positives, num_negatives):
    data_map = defaultdict(list)
    positive_pair_candidates = []
    for image, label in zip(images, labels):
        data_map[int(label)].append(image)
    for label, image_list in data_map.items():
        if len(image_list) >= 2:
            positive_pair_candidates.append(label)

    images_1 = []
    images_2 = []
    pair_labels = []

    if positive_pair_candidates:
        for _ in range(num_positives):
            label = random.choice(positive_pair_candidates)
            x, y = random.sample(data_map[label], 2)
            images_1.append(x)
            images_2.append(y)
            pair_labels.append(1)

    for _ in range(num_negatives):
        x, y = random.sample(data_map.keys(), 2)
        images_1.append(random.choice(data_map[x]))
        images_2.append(random.choice(data_map[y]))
        pair_labels.append(0)
    images_1 = tf.stack(images_1)
    images_2 = tf.stack(images_2)
    pair_labels = pair_labels
    return images_1, images_2, pair_labels
