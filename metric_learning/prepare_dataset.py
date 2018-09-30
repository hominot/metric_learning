import tensorflow as tf

from util.registry.data_loader import DataLoader
from util.dataset import split_train_test_by_label
from util.dataset import save_image_files

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
        directory = os.path.join('/tmp/research/experiment', args.dataset)

    data_loader: DataLoader = DataLoader.create({'name': args.dataset}, None)
    data_loader.prepare_files()

    shutil.rmtree(os.path.join(directory, 'train'), ignore_errors=True)
    shutil.rmtree(os.path.join(directory, 'test'), ignore_errors=True)
    image_files, labels = data_loader.load_image_files()
    training_data, testing_data = split_train_test_by_label(image_files, labels)
    save_image_files(list(zip(*training_data))[0], os.path.join(directory, 'train'))
    save_image_files(list(zip(*testing_data))[0], os.path.join(directory, 'test'))
