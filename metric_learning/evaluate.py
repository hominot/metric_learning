import tensorflow as tf

import argparse
import boto3
import json
import os
import tempfile

from util.config import CONFIG
from util.dataset import load_images_from_directory

from util.registry.model import Model
from util.registry.metric import Metric
from util.registry.data_loader import DataLoader
from util.registry.batch_design import BatchDesign


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


def get_checkpoint(temp_dir, experiment, step=None):
    if CONFIG['tensorboard'].getboolean('s3_upload'):
        if step is None:
            checkpoint_path = tf.train.latest_checkpoint(
                's3://hominot/research/metric_learning/experiments/{}/checkpoints'.format(experiment))
        else:
            checkpoint_path = 's3://hominot/research/metric_learning/experiments/{}/checkpoints/ckpt-{}'.format(
                experiment, step)
        for data in s3.list_objects(Bucket=CONFIG['tensorboard']['s3_bucket'],
                                    Prefix=checkpoint_path.split('/', 3)[-1])['Contents']:
            s3.download_file(
                Bucket=CONFIG['tensorboard']['s3_bucket'],
                Key=data['Key'],
                Filename=os.path.join(temp_dir, data['Key'].split('/')[-1]))
        return os.path.join(temp_dir, checkpoint_path.split('/')[-1])

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
        'name': 'recall',
        'k': [1, 2, 4, 8],
        'compute_period': 10,
        'batch_size': 48,
        'dataset': 'test',
    },
]

if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser(description='Train using a specified config')
    parser.add_argument('--experiment', help='experiment id')
    parser.add_argument('--step', help='batch number')
    args = parser.parse_args()

    conf = get_config(args.experiment)
    optimizer = tf.train.AdamOptimizer(learning_rate=conf['trainer']['learning_rate'])
    model = Model.create(conf['model']['name'], conf)

    test_dir = os.path.join(
        CONFIG['dataset']['experiment_dir'], conf['dataset']['name'], 'test')
    testing_files, testing_labels = load_images_from_directory(test_dir)

    data_loader = DataLoader.create(conf['dataset']['name'], conf)
    vanilla_ds = BatchDesign.create(
        'vanilla',
        conf,
        {'data_loader': data_loader})
    test_datasets = {
        'test': vanilla_ds.create_dataset(
            testing_files, testing_labels, testing=True),
    }

    checkpoint = tf.train.Checkpoint(model=model)
    with tempfile.TemporaryDirectory() as temp_dir:
        c = get_checkpoint(temp_dir, args.experiment, args.step)
        checkpoint.restore(c)

        for metric_conf in METRICS:
            metric = Metric.create(metric_conf['name'], conf)
            test_dataset, test_num_testcases = test_datasets[metric_conf['dataset']]
            score = metric.compute_metric(model, test_dataset, test_num_testcases)
            print(metric_conf['name'], score)
