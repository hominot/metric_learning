import tensorflow as tf
import os

import boto3

from util.config import CONFIG


s3 = boto3.client('s3')


def set_tensorboard_writer(model, data_loader):
    run_name = '{}_{}'.format(data_loader, model)
    run_dir = '{}_0001'.format(run_name)

    local_tensorboard_dir = os.path.join(CONFIG['tensorboard']['local_dir'], 'tensorboard')
    if not tf.gfile.Exists(local_tensorboard_dir):
        tf.gfile.MakeDirs(local_tensorboard_dir)
    runs = list(filter(
        lambda x: '_' in x and x.rsplit('_', 1)[0] == run_name,
        tf.gfile.ListDirectory(local_tensorboard_dir)
    ))

    if CONFIG['tensorboard'].getboolean('s3_upload'):
        prefix='{}/tensorboard/{}'.format(CONFIG['tensorboard']['s3_key'], run_name)
        response = s3.list_objects_v2(
            Bucket=CONFIG['tensorboard']['s3_bucket'],
            Prefix=prefix)
        keys = [x['Key'][:-1]
                for x in response.get('Contents', [])
                if x['Key'].endswith('/')]
        runs += list(filter(
            lambda x: '_' in x and x.rsplit('_', 1)[0].rsplit('/', 1)[1] == run_name,
            keys
        ))
    if runs:
        next_run = max([int(run.split('_')[-1]) for run in runs]) + 1
        run_dir = '{}_{:04d}'.format(run_name, next_run)

    if CONFIG['tensorboard'].getboolean('s3_upload'):
        s3.put_object(
            Bucket=CONFIG['tensorboard']['s3_bucket'],
            Body='',
            Key='{}/tensorboard/{}/'.format(CONFIG['tensorboard']['s3_key'], run_dir)
        )

    print('Starting {}'.format(run_dir))
    writer = tf.contrib.summary.create_file_writer(
        os.path.join(local_tensorboard_dir, run_dir),
        flush_millis=10000)
    return writer, run_dir


def upload_tensorboard_log_to_s3(run_name):
    run_dir = os.path.join(CONFIG['tensorboard']['local_dir'], 'tensorboard', run_name)
    for filename in os.listdir(run_dir):
        s3.upload_file(
            os.path.join(run_dir, filename),
            CONFIG['tensorboard']['s3_bucket'],
            '{}/tensorboard/{}/{}'.format(CONFIG['tensorboard']['s3_key'], run_name, filename))


def upload_file_to_s3(file_path, bucket, key):
    s3.upload_file(file_path, bucket, key)


def upload_string_to_s3(body, bucket, key):
    s3.put_object(Bucket=bucket, Body=body, Key=key)


def create_checkpoint(checkpoint, run_name):
    prefix = '{}/experiments/{}/checkpoints/ckpt'.format(
        CONFIG['tensorboard']['local_dir'],
        run_name)
    if not tf.gfile.Exists(os.path.dirname(prefix)):
        tf.gfile.MakeDirs(os.path.dirname(prefix))
    checkpoint.save(file_prefix=prefix)
    if CONFIG['tensorboard'].getboolean('s3_upload'):
        for root, dirnames, filenames in os.walk(os.path.dirname(prefix)):
            for filename in filenames:
                s3.upload_file(
                    os.path.join(root, filename),
                    CONFIG['tensorboard']['s3_bucket'],
                    '{}/experiments/{}/checkpoints/{}'.format(
                        CONFIG['tensorboard']['s3_key'],
                        run_name,
                        filename
                    )
                )
                os.remove(os.path.join(root, filename))
