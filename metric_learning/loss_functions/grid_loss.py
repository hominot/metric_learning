import tensorflow as tf


def loss(embeddings, labels, grid_points):
    grid_points_for_labels = grid_points[labels - 1, :]
    d = tf.reduce_sum(tf.square(embeddings - grid_points_for_labels), axis=1)
    return sum(d)
