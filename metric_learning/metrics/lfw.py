from __future__ import print_function

import os
import sys
import copy
import operator

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tqdm import trange

from util.registry.metric import Metric
from util.registry.data_loader import DataLoader
from util.registry.batch_design import BatchDesign
from util.tensor_operations import compute_elementwise_distances
from metric_learning.constants.distance_function import get_distance_function
from util.config import CONFIG


def _make_pairs_dataset():
    pairs, labels = [], []
    with open(os.path.join(CONFIG['dataset']['data_dir'], 'lfw', 'pairs.txt')) as f:
        num_groups, num_examples = map(int, f.readline().rstrip().split('\t'))
        label = True
        for line in f:
            if label is True:
                name1, n1, n2 = line.rstrip().split('\t')
                name2 = name1
            else:
                name1, n1, name2, n2 = line.rstrip().split('\t')
            pair_lhs = _make_full_image_path(name1, n1)
            pair_rhs = _make_full_image_path(name2, n2)
            pairs.append([pair_lhs, pair_rhs])
            labels.append(label)
            if len(pairs) % num_examples == 0:
                label = not label
    return pairs, labels


def _make_full_image_path(name, n):
    filename = '{}_{:04d}.jpg'.format(name, int(n))
    return os.path.join(CONFIG['dataset']['data_dir'], 'lfw', 'test', name, filename)


def _flatten_image_files(pairs):
    image_files, pair_indices = [], []
    for i, (path1, path2) in enumerate(pairs):
        image_files.append(path1); image_files.append(path2)
        pair_indices.append(i); pair_indices.append(i)
    return image_files, pair_indices


def _make_embedding_pairs(embeddings_list):
    embedding_pairs = []
    for embeddings in embeddings_list:
        for i in range(0, embeddings.shape[0].value, 2):
            embedding_pairs.append([embeddings[i], embeddings[i+1]])
    return embedding_pairs


def _calculate_accuracy(distances, labels):
    data = sorted(zip(distances, labels), key=operator.itemgetter(0))
    labels = [l for d, l in data]
    best_accuracy = 0.
    for k in range(len(data)+1):
        predicts = [True]*k + [False]*(len(data)-k)
        corrects = sum(p == l for p, l in zip(predicts, labels))
        accuracy = corrects / len(data)
        best_accuracy = max(accuracy, best_accuracy)
    return best_accuracy


class LFW(Metric):
    name = 'lfw'

    def compute_metric(self, model, ds, num_testcases):
        Metric.cache.clear()

        data_loader = DataLoader.create(self.conf['dataset']['name'], self.conf)
        batch_design = BatchDesign.create(
            self.metric_conf['batch_design']['name'],
            self.conf,
            {'data_loader': data_loader})

        pairs, labels = _make_pairs_dataset()
        image_files, pair_indices = _flatten_image_files(pairs)

        test_dataset, num_testcases = batch_design.create_dataset(
            model, image_files, pair_indices,
            self.metric_conf['batch_design'], testing=True)

        embeddings_list, pair_indices_list = self.get_embeddings(
            model, test_dataset, num_testcases)
        embedding_pairs = _make_embedding_pairs(embeddings_list)

        distances = []
        distance_function = get_distance_function(self.conf['loss']['distance_function'])
        for i, embedding_pair in enumerate(tqdm(
            embedding_pairs, desc='lfw:distance', dynamic_ncols=True)):
            distance = compute_elementwise_distances(
                tf.reshape(embedding_pair[0], [1, -1]),
                tf.reshape(embedding_pair[1], [1, -1]),
                distance_function)
            distances.append(float(distance))

        accuracies = []
        for i in trange(0, 6000, 600, desc='lfw:accuracy', dynamic_ncols=True):
            accuracy = _calculate_accuracy(distances[i:i+600], labels[i:i+600])
            accuracies.append(accuracy)

        Metric.cache.clear()

        return {
            'lfw_acc_avg': float(np.mean(accuracies)),
            'lfw_acc_std': float(np.std(accuracies))
        }
