from util.registry.metric import Metric

from collections import defaultdict
from tqdm import tqdm

import math
import tensorflow as tf

import sklearn.cluster
import sklearn.metrics.cluster


class NMI(Metric):
    name = 'nmi'

    def compute_metric(self, model, ds, num_testcases, embedding_cache):
        if self.metric_conf['dataset'] in embedding_cache:
            embeddings_list, labels_list = zip(*embedding_cache[self.metric_conf['dataset']])
        else:
            batch_size = self.metric_conf['batch_design']['batch_size']
            ds = ds.batch(batch_size)
            batches = tqdm(
                ds,
                total=math.ceil(num_testcases / batch_size),
                desc='embedding',
                dynamic_ncols=True)
            embeddings_list = []
            labels_list = []
            for images, labels in batches:
                embeddings = model(images, training=False)
                embeddings_list.append(embeddings)
                labels_list.append(labels)
            embedding_cache[self.metric_conf['dataset']] = zip(embeddings_list, labels_list)
        embeddings = tf.concat(embeddings_list, axis=0)
        labels = tf.concat(labels_list, axis=0)
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[int(label)] += 1
        clusters = sklearn.cluster.KMeans(len(label_counts)).fit(embeddings).labels_
        return sklearn.metrics.cluster.normalized_mutual_info_score(clusters, labels)
