import tensorflow as tf
import numpy as np

import os

from util.data_loader import DataLoader
from util.dataset import split_train_test_by_label
from util.model import Model
from util.metric import Metric


tf.enable_eager_execution()


conf = {
    'dataset': {
        'name': 'lfw',
    },
    'model': {
        'name': 'center_embedding',
        'child_model': {
            'name': 'simple_conv',
            'k': 4,
        },
    },
    'metrics': [
        {
            'name': 'accuracy',
            'compute_period': 10,
            'conf': {
                'sampling_rate': 0.1,
            }
        }
    ]
}


tensorboard_dir = '/tmp/tensorflow/metric_learning'
if not tf.gfile.Exists(tensorboard_dir):
    tf.gfile.MakeDirs(tensorboard_dir)

data_loader: DataLoader = DataLoader.create(conf['dataset'])
image_files, labels = data_loader.load_image_files()
training_data, testing_data = split_train_test_by_label(image_files, labels)

num_labels = max(labels)

extra_info = {
    'num_labels': num_labels,
}
grid_points = np.random.random([num_labels, 16]) * 2 - 1

test_ds = data_loader.create_dataset(*zip(*testing_data)).batch(256)

step_counter = tf.train.get_or_create_global_step()
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
model = Model.create(conf['model'], extra_info)

device = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'

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

for _ in range(10):
    train_ds = data_loader.create_grouped_dataset(
        *zip(*training_data),
        group_size=16,
        num_groups=4,
    ).batch(64)
    with tf.device(device):
        for (batch, (images, labels)) in enumerate(train_ds):
            with tf.contrib.summary.record_summaries_every_n_global_steps(
                    10, global_step=step_counter):
                with tf.GradientTape() as tape:
                    loss_value = model.loss(images, labels)
                    tf.contrib.summary.scalar('loss', loss_value)

                for metric_conf in conf['metrics']:
                    if int(tf.train.get_global_step()) % metric_conf.get('compute_period', 10) == 0:
                        metric = Metric.create(metric_conf)
                        score = metric.compute_metric(model, test_ds, **metric_conf['conf'])
                        tf.contrib.summary.scalar(metric_conf['name'], score)
                grads = tape.gradient(loss_value, model.variables)
                optimizer.apply_gradients(
                    zip(grads, model.variables), global_step=step_counter)
