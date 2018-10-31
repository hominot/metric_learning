import tensorflow as tf

from metric_learning.constants.distance_function import DistanceFunction
from util.registry.batch_design import BatchDesign
from metric_learning.batch_designs.grouped import GroupedBatchDesign
from metric_learning.batch_designs.grouped import get_npair_distances

tf.enable_eager_execution()


class GroupedBatchDesignTest(tf.test.TestCase):
    def testGroupedDataset(self):
        image_files = ['a', 'b', 'c', 'd', 'e', 'f']
        labels = [3, 1, 2, 3, 1, 1]

        conf = {
            'batch_design': {
                'name': 'grouped',
                'group_size': 2,
                'num_groups': 2,
            }
        }
        dataset: GroupedBatchDesign = BatchDesign.create('grouped', conf, {'data_loader': None})
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
        distances, matches = get_npair_distances(
            embeddings, 2, DistanceFunction.EUCLIDEAN_DISTANCE)
        self.assertAllEqual(distances, [
            [1., 6.],
            [3., 2.],
        ])


if __name__ == '__main__':
    tf.test.main()
