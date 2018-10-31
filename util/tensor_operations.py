import tensorflow as tf


def pairwise_euclidean_distance_squared(first, second):
    return tf.reduce_sum(
        tf.square(second[None] - first[:, None]),
        axis=2)


def pairwise_euclidean_distance(first, second):
    return stable_sqrt(pairwise_euclidean_distance_squared(first, second))


def pairwise_dot_product(first, second):
    return tf.reduce_sum(
        tf.multiply(second[None], first[:, None]),
        axis=2)


def pairwise_cosine_similarity(first, second):
    first_norm = first / tf.norm(first, axis=1, keep_dims=True)
    second_norm = second / tf.norm(second, axis=1, keep_dims=True)
    return pairwise_dot_product(first_norm, second_norm)


def pairwise_difference(first, second):
    return -second[None] + first[:, None]


def pairwise_matching_matrix(first, second):
    return tf.equal(second[None], first[:, None])


def repeat_columns(labels):
    return tf.tile(labels[:, None], [1, labels.shape[0]])


def upper_triangular_part(matrix):
    a = tf.linalg.band_part(tf.ones(matrix.shape), -1, 0)
    return tf.boolean_mask(matrix, 1 - a)


def off_diagonal_part(matrix):
    return tf.boolean_mask(matrix, 1 - tf.eye(int(matrix.shape[0])))


def stable_sqrt(tensor):
    return tf.sqrt(tf.maximum(tensor, 1e-12))


def get_n_blocks(tensor, n):
    r = tf.range(tensor.shape[0])
    mask = tf.equal(r[None] // n, r[:, None] // n)
    return tf.reshape(tf.boolean_mask(tensor, mask), [-1, n])
