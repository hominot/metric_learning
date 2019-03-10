from __future__ import print_function

import tensorflow as tf
from tqdm import tqdm

from util.registry.metric import Metric
from util.tensor_operations import compute_pairwise_distances
from metric_learning.constants.distance_function import get_distance_function


class LFW(Metric):
    name = 'lfw'

    def compute_metric(self, model, ds, num_testcases):
        embeddings_list, labels_list = self.get_embeddings(model, ds, num_testcases)
        distance_function = get_distance_function(self.conf['loss']['distance_function'])
        data = list(zip(embeddings_list, labels_list))
        batches = tqdm(
            data, total=len(embeddings_list), desc='lfw', dynamic_ncols=True)
        for i, (embeddings, labels) in enumerate(batches):
            for j, (test_embeddings, test_labels) in enumerate(data):
                pairwise_distances = compute_pairwise_distances(
                    embeddings, test_embeddings, distance_function)

        return {'lfw': 1.0}
