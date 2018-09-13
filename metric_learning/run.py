import tensorflow as tf
import numpy as np

import random
import os

from collections import defaultdict
from util.data_loader import DataLoader
from util.dataset import split_train_test_by_label, split_train_test
from util.loss_function import LossFunction
from metric_learning.models.simple_conv import create_model


tf.enable_eager_execution()


def compute_verification_accuracy(model, testing_ds, sampling_rate=1.0):
    data_map = defaultdict(list)
    for images, labels in testing_ds:
        embeddings = model(images, training=False)
        for index in range(embeddings.shape[0]):
            data_map[int(labels[index])].append(embeddings[index])

    failure = 0.
    total = 0.
    norm_sum = 0.
    norm_total = 0.
    for label in data_map.keys():
        if len(data_map[label]) < 2:
            continue
        for index, anchor_embedding in enumerate(data_map[label]):
            if random.random() > sampling_rate:
                continue
            norm_sum += float(tf.norm(anchor_embedding))
            total += 1
            positive_indices = random.sample(range(len(data_map[label])), 2)
            positive_index = positive_indices[0] if positive_indices[0] != index else positive_indices[1]
            p_d = tf.norm(anchor_embedding - data_map[label][positive_index])
            num_negative_examples = min(5, len(data_map) - 1)
            negative_labels = random.sample(set(data_map.keys()) - {label}, num_negative_examples)
            negative_embeddings = []
            for negative_label in negative_labels:
                negative_embeddings.append(random.choice(data_map[negative_label]))

            for negative_embedding in negative_embeddings:
                if tf.norm(negative_embedding - anchor_embedding) <= p_d:
                    failure += 1
                    break

    return (total - failure) / total, norm_sum / total


conf = {
    'dataset': 'lfw',
    'loss': {
        'name': 'grid',
        'conf': {
        },
    },
}


tensorboard_dir = '/tmp/tensorflow/metric_learning'
if not tf.gfile.Exists(tensorboard_dir):
    tf.gfile.MakeDirs(tensorboard_dir)

data_loader: DataLoader = DataLoader.create(conf['dataset'])
image_files, labels = data_loader.load_image_files()
training_data, testing_data = split_train_test_by_label(image_files, labels)
num_labels = max(labels)
grid_points = np.random.random([num_labels, 16]) * 2 - 1

test_ds = data_loader.create_dataset(*zip(*testing_data)).batch(256)

step_counter = tf.train.get_or_create_global_step()
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
model = create_model()

device = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'

loss_function: LossFunction = LossFunction.create(conf['loss']['name'])

run_name = '{}_simple_conv_{}_loss'.format(
    conf['dataset'],
    conf['loss']['name'],
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
        group_size=2,
        num_groups=4,
    ).batch(64)
    with tf.device(device):
        for (batch, (images, labels)) in enumerate(train_ds):
            with tf.contrib.summary.record_summaries_every_n_global_steps(
                    10, global_step=step_counter):
                with tf.GradientTape() as tape:
                    embeddings = model(images, training=True)
                    loss_value = loss_function.loss(embeddings, labels, grid_points=grid_points, **conf['loss']['conf'])
                    tf.contrib.summary.scalar('loss', loss_value)

                if int(tf.train.get_global_step()) % 10 == 0:
                    accuracy, norm_avg = compute_verification_accuracy(model, test_ds, 0.1)
                    tf.contrib.summary.scalar('accuracy', accuracy)
                    tf.contrib.summary.scalar('norm', norm_avg)
                grads = tape.gradient(loss_value, model.variables)
                optimizer.apply_gradients(
                    zip(grads, model.variables), global_step=step_counter)
