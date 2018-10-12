import tensorflow as tf

import argparse
import boto3
import json
import os

from util.config import CONFIG
from util.dataset import create_dataset_from_directory

from util.registry.model import Model
from util.registry.metric import Metric
from util.registry.data_loader import DataLoader


s3 = boto3.client('s3')


def get_config(experiment):
    if CONFIG['tensorboard'].getboolean('s3_upload'):
        ret = s3.get_object(
            Bucket='hominot',
            Key='research/metric_learning/experiments/{}/config.json'.format(experiment))
        return json.loads(ret['Body'].read().decode('utf-8'))

    config_dir = os.path.join(CONFIG['tensorboard']['local_dir'], 'experiments', experiment)
    with open(os.path.join(config_dir, 'config.json'), 'r') as f:
        return json.load(f)


def get_checkpoint(experiment, step=None):
    if CONFIG['tensorboard'].getboolean('s3_upload'):
        if step is None:
            return tf.train.latest_checkpoint(
                's3://hominot/research/metric_learning/experiments/{}/checkpoints'.format(experiment))
        return 's3://hominot/research/metric_learning/experiments/{}/checkpoints/ckpt-{}'.format(experiment, step)

    if step is None:
        prefix = '{}/experiments/{}/checkpoints'.format(
            CONFIG['tensorboard']['local_dir'],
            experiment)
        return tf.train.latest_checkpoint(prefix)
    return '{}/experiments/{}/checkpoints/ckpt-{}'.format(
        CONFIG['tensorboard']['local_dir'],
        experiment,
        step)


METRICS = [
    {
        'name': 'accuracy',
        'compute_period': 10,
        'batch_size': 48,
    },
    {
        'name': 'recall',
        'k': [1, 2],
        'compute_period': 10,
        'batch_size': 48,
    },
]

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser(description='Train using a specified config')
    parser.add_argument('--experiment', help='experiment id')
    parser.add_argument('--step', help='batch number')
    args = parser.parse_args()

    conf = get_config(args.experiment)
    optimizer = tf.train.AdamOptimizer(learning_rate=conf['optimizer']['learning_rate'])
    model = Model.create(conf['model']['name'], conf)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=model,
                                     optimizer_step=tf.train.get_or_create_global_step())
    checkpoint.restore(get_checkpoint(args.experiment))

    testing_files, testing_labels = create_dataset_from_directory(
        conf['dataset']['test']['data_directory']
    )
    data_loader = DataLoader.create(conf['dataset']['name'], conf)
    test_datasets = {}
    if 'identification' in conf['dataset']['test']:
        dataset, num_testcases = data_loader.create_identification_test_dataset(
            testing_files, testing_labels)
        test_datasets['identification'] = {
            'dataset': dataset,
            'num_testcases': num_testcases,
        }
    if 'recall' in conf['dataset']['test']:
        images_ds, labels_ds, num_testcases = data_loader.create_recall_test_dataset(
            testing_files, testing_labels)
        test_datasets['recall'] = {
            'dataset': (images_ds, labels_ds),
            'num_testcases': num_testcases,
        }
    for metric_conf in METRICS:
        metric = Metric.create(metric_conf['name'], conf)
        score = metric.compute_metric(model, test_datasets[metric.dataset]['dataset'], test_datasets[metric.dataset]['num_testcases'])
        print(metric_conf['name'], score)
