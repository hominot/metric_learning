import tensorflow as tf


def pairwise_euclidean_distance_squared(first, second):
    return tf.reduce_sum(
        tf.square(second[None] - first[:, None]),
        axis=2)


def pairwise_dot_product(first, second):
    return tf.reduce_sum(
        tf.multiply(second[None], first[:, None]),
        axis=2)


def pairwise_difference(first, second):
    return -second[None] + first[:, None]


def pairwise_matching_matrix(first, second):
    return tf.equal(second[None], first[:, None])


def repeat_columns(labels):
    return tf.tile(labels[:, None], [1, labels.shape[0]])


def upper_triangular_part(matrix):
    a = tf.linalg.band_part(tf.ones(matrix.shape), -1, 0)
    return tf.boolean_mask(matrix, 1 - a)