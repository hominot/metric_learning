import tensorflow as tf
import numpy as np

tfe = tf.contrib.eager

import time
import random
import os

from collections import defaultdict
from util.data_loader import DataLoader
from util.dataset import split_train_test
from metric_learning.models.simple_dense import create_model
from metric_learning.loss_functions.grid_loss import grid_loss


tf.enable_eager_execution()


def compute_verification_accuracy(model, testing_ds, sampling_rate=1.0):
    data_map = defaultdict(list)
    for images, labels in testing_ds:
        embeddings = model(images, training=False)
        for index in range(embeddings.shape[0]):
            data_map[int(labels[index])].append(embeddings[index])

    failure = 0.
    total = 0.
    for label in data_map.keys():
        if len(data_map[label]) < 2:
            continue
        for index, anchor_embedding in enumerate(data_map[label]):
            if random.random() > sampling_rate:
                continue
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
                if tf.norm(negative_embedding - anchor_embedding) < p_d:
                    failure += 1
                    break

    return (total - failure) / total


tensorboard_dir = '/tmp/tensorflow/metric_learning'
if not tf.gfile.Exists(tensorboard_dir):
    tf.gfile.MakeDirs(tensorboard_dir)

run_name = 'mnist_simple_dense_variable_grid_loss'
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

data_loader: DataLoader = DataLoader.create('mnist')
image_files, labels = data_loader.load_image_files()
training_data, testing_data = split_train_test(image_files, labels)
num_labels = max(labels)
grid_points = np.random.random([num_labels, 8]) * 2 - 1

train_ds = data_loader.create_dataset(*zip(*training_data)) \
    .shuffle(3000).batch(256)
test_ds = data_loader.create_dataset(*zip(*testing_data)).batch(256)

step_counter = tf.train.get_or_create_global_step()
optimizer = tf.train.AdamOptimizer()
model = create_model()

device = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'

start = time.time()

with tf.device(device):
    for _ in range(10):
        for (batch, (images, labels)) in enumerate(train_ds):
            with tf.contrib.summary.record_summaries_every_n_global_steps(
                    10, global_step=step_counter):
                with tf.GradientTape() as tape:
                    embeddings = model(images, training=True)
                    loss_value = grid_loss(embeddings, labels, grid_points)
                    tf.contrib.summary.scalar('loss', loss_value)

                if int(tf.train.get_global_step()) % 10 == 0:
                    tf.contrib.summary.scalar('accuracy', compute_verification_accuracy(model, test_ds, 0.1))
                grads = tape.gradient(loss_value, model.variables)
                optimizer.apply_gradients(
                    zip(grads, model.variables), global_step=step_counter)
