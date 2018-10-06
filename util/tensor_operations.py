import tensorflow as tf


def pairwise_euclidean_distance_squared(first, second):
    return tf.reduce_sum(
        tf.square(first[None] - second[:, None]),
        axis=2)


def pairwise_dot_product(first, second):
    return tf.reduce_sum(
        tf.multiply(first[None], second[:, None]),
        axis=2)


def pairwise_matching_matrix(labels):
    return tf.equal(labels[None], labels[:, None])


def upper_triangular_part(matrix):
    a = tf.linalg.band_part(tf.ones(matrix.shape), -1, 0)
    return tf.boolean_mask(matrix, 1 - a)
