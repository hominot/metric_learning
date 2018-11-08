from util.registry.metric import Metric

from tqdm import tqdm
from util.registry.batch_design import BatchDesign
from metric_learning.constants.distance_function import DistanceFunction

import math
import tensorflow as tf


class AUC(Metric):
    name = 'auc'

    def compute_metric(self, model, ds, num_testcases):
        batch_size = self.metric_conf['batch_design']['batch_size']
        ds = ds.batch(batch_size)
        conf_copy = {}
        conf_copy.update(self.conf)
        conf_copy['batch_design'] = self.metric_conf['batch_design']
        batch_design = BatchDesign.create(
            self.metric_conf['batch_design']['name'],
            conf_copy,
            {})
        parametrization = self.conf['loss']['parametrization']
        if parametrization == 'dot_product':
            distance_function = DistanceFunction.DOT_PRODUCT
        else:
            distance_function = DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED

        scores = []
        batches = tqdm(
            ds,
            total=math.ceil(num_testcases / batch_size),
            desc='auc',
            dynamic_ncols=True)
        for batch in batches:
            distances, match, _ = batch_design.get_pairwise_distances(
                batch, model, distance_function, training=False)
            num_pairs = int(distances.shape[0]) // 2
            scores.append(tf.reduce_mean(tf.cast(
                distances[:num_pairs] < distances[num_pairs:],
                tf.float32
            )))

        return float(sum(scores)) / len(scores)
