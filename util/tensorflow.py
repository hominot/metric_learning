import tensorboard as tf
import os

tensorboard_dir = '/tmp/tensorflow/metric_learning'


def set_tensorboard_writer(conf):
    if not tf.gfile.Exists(tensorboard_dir):
        tf.gfile.MakeDirs(tensorboard_dir)
    run_name = '{}_{}'.format(
        conf['dataset']['name'],
        conf['model']['name'],
    )
    run_dir = '{}_0001'.format(run_name)
    runs = list(filter(
        lambda x: '_' in x and x.rsplit('_', 1)[0] == run_name,
        next(os.walk(tensorboard_dir))[1]
    ))
    if runs:
        next_run = int(max(runs).split('_')[-1]) + 1
        run_dir = '{}_{:04d}'.format(run_name, next_run)
    writer = tf.contrib.summary.create_file_writer(
        os.path.join(tensorboard_dir, run_dir),
        flush_millis=10000)
    writer.set_as_default()
