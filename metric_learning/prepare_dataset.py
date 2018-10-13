import tensorflow as tf

from util.registry.data_loader import DataLoader
from util.config import CONFIG

import argparse
import random
import shutil
import os


if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser(
        description='Download dataset for training/testing models')
    parser.add_argument('--dataset', help='Name of the dataset')

    args = parser.parse_args()

    directory = os.path.join(CONFIG['dataset']['experiment_dir'], args.dataset)

    data_loader: DataLoader = DataLoader.create(args.dataset)
    data_loader.prepare_files()

    # copy test data
    shutil.rmtree(os.path.join(directory, 'test'), ignore_errors=True)
    shutil.copytree(
        os.path.join(CONFIG['dataset']['data_dir'], args.dataset, 'test'),
        os.path.join(directory, 'test'))

    # cross validation splits
    shutil.rmtree(os.path.join(directory, 'train'), ignore_errors=True)
    splits = list(range(CONFIG['dataset']['cross_validation_splits']))
    for k in splits:
        if not tf.gfile.Exists(os.path.join(directory, 'train', str(k))):
            tf.gfile.MakeDirs(os.path.join(directory, 'train', str(k)))
    for root, dirnames, filenames in os.walk(os.path.join(CONFIG['dataset']['data_dir'], args.dataset, 'train')):
        for filename in filenames:
            split = random.choice(splits)
            dest_filename = os.path.join(root.split('/')[-1], filename)
            dest_dir = os.path.join(directory, 'train', str(split))
            if not tf.gfile.Exists(os.path.dirname(os.path.join(dest_dir, dest_filename))):
                tf.gfile.MakeDirs(os.path.dirname(os.path.join(dest_dir, dest_filename)))
            shutil.copy(os.path.join(root, filename), os.path.join(directory, 'train', str(split), dest_filename))
