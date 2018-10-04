import tensorflow as tf
import os

import boto3

from util.config import config


def set_tensorboard_writer(model, data_loader):
    run_name = '{}_{}'.format(data_loader, model)
    run_dir = '{}_0001'.format(run_name)

    local_tensorboard_dir = config['tensorboard']['local_dir']
    runs = list(filter(
        lambda x: '_' in x and x.rsplit('_', 1)[0] == run_name,
        tf.gfile.ListDirectory(local_tensorboard_dir)
    ))

    if config['tensorboard']['s3_upload']:
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(
            Bucket=config['tensorboard']['s3_bucket'],
            Prefix='{}/{}'.format(config['tensorboard']['s3_key'], run_name))
        keys = [x['Key'][:-1]
                for x in response.get('Contents', [])
                if x['Key'].endswith('/') and len(x['Key']) == len(config['tensorboard']['s3_key']) + len(run_name) + 7]
        runs += list(filter(
            lambda x: '_' in x and x.rsplit('_', 1)[0] == run_name,
            keys
        ))
    if runs:
        next_run = int(max(runs).split('_')[-1]) + 1
        run_dir = '{}_{:04d}'.format(run_name, next_run)
    if not tf.gfile.Exists(local_tensorboard_dir):
        tf.gfile.MakeDirs(local_tensorboard_dir)

    if config['tensorboard']['s3_upload']:
        s3 = boto3.client('s3')
        s3.put_object(
            Bucket=config['tensorboard']['s3_bucket'],
            Body='',
            Key='{}/{}/'.format(config['tensorboard']['s3_key'], run_dir)
        )

    print('Starting {}'.format(run_dir))
    writer = tf.contrib.summary.create_file_writer(
        os.path.join(local_tensorboard_dir, run_dir),
        flush_millis=10000)
    return writer, run_dir


def upload_to_s3(run_name):
    s3 = boto3.client('s3')
    run_dir = os.path.join(config['tensorboard']['local_dir'], run_name)
    for filename in os.listdir(run_dir):
        s3.upload_file(
            os.path.join(run_dir, filename),
            config['tensorboard']['s3_bucket'],
            '{}/{}/{}'.format(config['tensorboard']['s3_key'], run_name, filename))
