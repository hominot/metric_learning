import tensorflow as tf
import time
import random
import os

from collections import defaultdict
from util.data_loader import DataLoader
from util.dataset import split_train_test
from metric_learning.models.simple_dense import create_model
from metric_learning.loss_functions.triplet_loss import triplet_loss


tf.enable_eager_execution()


def compute_verification_accuracy(embeddings, labels):
    data_map = defaultdict(set)
    for index in range(embeddings.shape[0]):
        data_map[int(labels[index])].add(index)

    failure = 0.
    total = 0.
    for index in range(embeddings.shape[0]):
        label = int(labels[index])
        if len(data_map[label]) < 2:
            continue
        total += 1
        anchor_embedding = embeddings[index]
        positive_indices = random.sample(data_map[label], 2)
        positive_index = positive_indices[0] if positive_indices[0] != index else positive_indices[1]
        p_d = tf.norm(anchor_embedding - embeddings[positive_index])

        negative_candidates = set(range(embeddings.shape[0])) - data_map[int(labels[index])]
        negative_indices = random.sample(negative_candidates, min(len(negative_candidates), 5))

        for negative_index in negative_indices:
            if tf.norm(embeddings[negative_index] - anchor_embedding) < p_d:
                failure += 1
                break

    return (total - failure) / total


tensorboard_dir = '/tmp/tensorflow/metric_learning'
if not tf.gfile.Exists(tensorboard_dir):
    tf.gfile.MakeDirs(tensorboard_dir)

run_name = 'triplet_loss'
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

train_ds = data_loader.create_dataset(*zip(*training_data)) \
    .shuffle(10000).batch(256)

step_counter = tf.train.get_or_create_global_step()
optimizer = tf.train.AdamOptimizer()
model = create_model()

start = time.time()
for _ in range(10):
    for (batch, (images, labels)) in enumerate(train_ds):
        with tf.contrib.summary.record_summaries_every_n_global_steps(
                10, global_step=step_counter):
            with tf.GradientTape() as tape:
                embeddings = model(images, training=True)
                loss_value = triplet_loss(embeddings, labels)
                tf.contrib.summary.scalar('loss', loss_value)
                tf.contrib.summary.scalar('accuracy', compute_verification_accuracy(embeddings, labels))
            grads = tape.gradient(loss_value, model.variables)
            optimizer.apply_gradients(
                zip(grads, model.variables), global_step=step_counter)
