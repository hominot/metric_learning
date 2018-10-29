import tensorflow as tf

from metric_learning.constants.distance_function import DistanceFunction
from util.registry.dataset import Dataset
from metric_learning.datasets.grouped import GroupedDataset
from metric_learning.datasets.grouped import get_npair_distances

tf.enable_eager_execution()


class GroupedDatasetTest(tf.test.TestCase):
    def testGroupedDataset(self):
        image_files = ['a', 'b', 'c', 'd', 'e', 'f']
        labels = [3, 1, 2, 3, 1, 1]

        conf = {
            'dataset': {
                'dataset': {
                    'group_size': 2,
                    'num_groups': 2,
                }
            }
        }
        dataset: GroupedDataset = Dataset.create('grouped', conf, {'data_loader': None})
        batch = dataset.get_next_batch(image_files, labels)
        self.assertAllEqual(batch, [
            ('b', 1),
            ('f', 1),
            ('d', 3),
            ('a', 3),
        ])

    def testGetNpairDistances(self):
        embeddings = tf.constant([
            [1.],
            [2.],
            [5.],
            [7.],
        ])
        distances, matches = get_npair_distances(embeddings, DistanceFunction.EUCLIDEAN_DISTANCE)
        self.assertAllEqual(distances, [
            [1., 6.],
            [3., 2.],
        ])


if __name__ == '__main__':
    tf.test.main()
