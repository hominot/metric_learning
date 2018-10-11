import tensorflow as tf

from util.registry.data_loader import DataLoader
from util.config import CONFIG

import argparse
import shutil
import os


if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser(
        description='Prepare dataset for training/testing models')
    parser.add_argument('--dataset', help='Name of the dataset')
    parser.add_argument('--dir', help='Directory to save training/testing datasets')

    args = parser.parse_args()

    directory = args.dir
    if directory is None:
        directory = os.path.join(CONFIG['dataset']['experiment_dir'], args.dataset)

    data_loader: DataLoader = DataLoader.create(args.dataset)
    data_loader.prepare_files()

    shutil.rmtree(os.path.join(directory, 'train'), ignore_errors=True)
    shutil.rmtree(os.path.join(directory, 'test'), ignore_errors=True)

    for split in ['train', 'test']:
        shutil.copytree(
            os.path.join(CONFIG['dataset']['data_dir'], args.dataset, split),
            os.path.join(directory, split))
