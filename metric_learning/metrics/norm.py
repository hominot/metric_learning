from util.registry.metric import Metric
from collections import defaultdict

import tensorflow as tf


class Norm(Metric):
    name = 'norm'

    def compute_metric(self, model, test_ds, *args, **kwargs):
        num_batch = 0
        norm = 0.
        for anchor_images, positive_images, negative_images_group in test_ds:
            anchor_embeddings = model(anchor_images, training=False)
            norm += float(tf.reduce_mean(tf.norm(anchor_embeddings, axis=1)))
            num_batch += 1

        return norm / num_batch
