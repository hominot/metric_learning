import tensorflow as tf
import random

from collections import defaultdict
from util.loss_function import LossFunction


def sample_triples(images, labels):
    data_map = defaultdict(list)
    positive_pair_candidates = []
    for index, label in enumerate(labels):
        data_map[int(label)].append(index)
    for label, index_list in data_map.items():
        if len(index_list) >= 2:
            positive_pair_candidates.append(label)

    anchor_images = []
    positive_images = []
    negative_images = []

    for label in positive_pair_candidates:
        for index in data_map[int(label)]:
            anchor_images.append(images[index])
            x, y = random.sample(data_map[int(label)], 2)
            positive_index = x if index != x else y
            positive_images.append(images[positive_index])
            x, y = random.sample(data_map.keys(), 2)
            negative_label = x if int(label) != x else y
            negative_images.append(images[random.choice(data_map[negative_label])])

    return (
        tf.stack(anchor_images),
        tf.stack(positive_images),
        tf.stack(negative_images),
    )


class TripletLossFunction(LossFunction):
    name = 'triplet'

    def loss(self, embeddings, labels):
        anchor_images, positive_images, negative_images = sample_triples(embeddings, labels)

        d_p = tf.reduce_sum(tf.square(anchor_images - positive_images), axis=1)
        d_n = tf.reduce_sum(tf.square(anchor_images - negative_images), axis=1)
        loss_value = sum(tf.maximum(0, 1 + d_p - d_n))

        return loss_value
