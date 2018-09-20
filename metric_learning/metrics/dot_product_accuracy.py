from util.registry.metric import Metric
from collections import defaultdict

import random
import tensorflow as tf


class DotProductAccuracy(Metric):
    name = 'dot_product_accuracy'

    def compute_metric(self, model, test_ds, *args, **kwargs):

        data_map = defaultdict(list)
        for images, labels in test_ds:
            embeddings = model(images, training=False)
            for index in range(embeddings.shape[0]):
                data_map[int(labels[index])].append(embeddings[index])

        failure = 0.
        total = 0.
        for label in data_map.keys():
            if len(data_map[label]) < 2:
                continue
            for index, anchor_embedding in enumerate(data_map[label]):
                if 'sampling_rate' in kwargs and random.random() > kwargs['sampling_rate']:
                    continue
                total += 1
                positive_indices = random.sample(range(len(data_map[label])), 2)
                positive_index = positive_indices[0] if positive_indices[0] != index else positive_indices[1]
                p_d = tf.reduce_sum(tf.multiply(anchor_embedding, data_map[label][positive_index]))
                num_negative_examples = min(5, len(data_map) - 1)
                negative_labels = random.sample(set(data_map.keys()) - {label}, num_negative_examples)
                negative_embeddings = []
                for negative_label in negative_labels:
                    negative_embeddings.append(random.choice(data_map[negative_label]))

                for negative_embedding in negative_embeddings:
                    if tf.reduce_sum(tf.multiply(negative_embedding, anchor_embedding)) >= p_d:
                        failure += 1
                        break

        return (total - failure) / total
