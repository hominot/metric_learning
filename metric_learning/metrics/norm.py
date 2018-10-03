from util.registry.metric import Metric

from tqdm import tqdm

import random
import tensorflow as tf


class Norm(Metric):
    name = 'norm'

    def compute_metric(self, model, test_ds, num_testcases):
        num_batch = 0
        norm = 0.
        batch_size = self.conf['batch_size']
        test_ds = test_ds.batch(batch_size).prefetch(batch_size)
        for anchor_images, positive_images, negative_images_group in tqdm(
                test_ds, total=num_testcases // batch_size, desc=self.name):
            if random.random() > self.conf.get('sampling_rate', 1.0):
                continue
            anchor_embeddings = model(anchor_images, training=False)
            norm += float(tf.reduce_mean(tf.norm(anchor_embeddings, axis=1)))
            num_batch += 1

        return norm / num_batch
