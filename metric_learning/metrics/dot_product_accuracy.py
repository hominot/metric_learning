from util.registry.metric import Metric

import tensorflow as tf


class DotProductAccuracy(Metric):
    name = 'dot_product_accuracy'

    def compute_metric(self, model, test_ds, *args, **kwargs):

        total = 0.
        success = 0.
        for anchor_images, positive_images, negative_images_group in test_ds:
            anchor_embeddings = model(anchor_images, training=False)
            positive_embeddings = model(positive_images, training=False)
            negative_embeddings = tf.stack([model(negative_images) for negative_images in negative_images_group])
            p = tf.reduce_sum(tf.multiply(anchor_embeddings, positive_embeddings), axis=1)
            n = tf.reduce_min(tf.reduce_sum(tf.multiply(negative_embeddings, anchor_embeddings), axis=2), axis=0)
            success += sum(tf.cast(p > n, tf.float32))
            total += int(anchor_images.shape[0])

        return success / total
