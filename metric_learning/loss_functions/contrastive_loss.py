import tensorflow as tf
import numpy as np

from util.dataset import sample_pairs


def contrastive_loss(embeddings, labels):
    images_1, images_2, pair_labels = sample_pairs(
        embeddings, labels, labels.shape[0], labels.shape[0])

    pair_labels = np.array(pair_labels)

    positive_images_1 = tf.boolean_mask(images_1, pair_labels == 1)
    positive_images_2 = tf.boolean_mask(images_2, pair_labels == 1)
    negative_images_1 = tf.boolean_mask(images_1, pair_labels == 0)
    negative_images_2 = tf.boolean_mask(images_2, pair_labels == 0)

    d_p = tf.reduce_sum(tf.square(positive_images_1 - positive_images_2), axis=1)
    d_n = tf.reduce_sum(tf.square(negative_images_1 - negative_images_2), axis=1)
    loss_value = sum(d_p) + sum(tf.maximum(0, 1 - d_n))

    return loss_value
