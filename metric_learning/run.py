import tensorflow as tf
import numpy as np
import time

from metric_learning.dataset.lfw import create_dataset
from metric_learning.models.simple_conv import create_model
from metric_learning.loss_functions.contrastive_loss import contrastive_loss


tf.enable_eager_execution()


def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
    labels = tf.cast(labels, tf.int64)
    batch_size = int(logits.shape[0])
    return tf.reduce_sum(
        tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size


writer = tf.contrib.summary.create_file_writer(
    '/tmp/tensorflow/metric_learning',
    flush_millis=10000)
writer.set_as_default()

image_dataset = create_dataset()
train_ds = image_dataset.shuffle(1024).batch(32)

step_counter = tf.train.get_or_create_global_step()
optimizer = tf.train.AdamOptimizer()
model = create_model()

start = time.time()
for (batch, (images, labels)) in enumerate(train_ds):
    with tf.contrib.summary.record_summaries_every_n_global_steps(
            10, global_step=step_counter):
        with tf.GradientTape() as tape:
            embeddings = model(images, training=True)
            loss_value = contrastive_loss(embeddings, labels)
            tf.contrib.summary.scalar('loss', loss_value)
            #tf.contrib.summary.scalar('accuracy', compute_accuracy(embeddings, labels))
        grads = tape.gradient(loss_value, model.variables)
        optimizer.apply_gradients(
            zip(grads, model.variables), global_step=step_counter)
