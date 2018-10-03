import tensorflow as tf
import os

import boto3

tensorboard_dir = '/tmp/tensorflow/metric_learning'


def set_tensorboard_writer(model, data_loader):
    if not tf.gfile.Exists(tensorboard_dir):
        tf.gfile.MakeDirs(tensorboard_dir)
    run_name = '{}_{}'.format(data_loader, model)
    run_dir = '{}_0001'.format(run_name)
    runs = list(filter(
        lambda x: '_' in x and x.rsplit('_', 1)[0] == run_name,
        tf.gfile.ListDirectory(tensorboard_dir)
    ))
    if runs:
        next_run = int(max(runs).split('_')[-1]) + 1
        run_dir = '{}_{:04d}'.format(run_name, next_run)
    writer = tf.contrib.summary.create_file_writer(
        os.path.join(tensorboard_dir, run_dir),
        flush_millis=10000)
    return writer


def upload_to_s3(run_name):
    s3 = boto3.client('s3')
    run_dir = os.path.join(tensorboard_dir, run_name)
    for filename in os.listdir(run_dir):
        s3.meta.client.upload_file(
            filename,
            'hominot',
            'research/metric_learning/tensorboard/{}/{}'.format(run_name, filename))

