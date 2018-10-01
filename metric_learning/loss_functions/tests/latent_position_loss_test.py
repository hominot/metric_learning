import tensorflow as tf

from metric_learning.loss_functions.latent_position_loss import pairwise_euclidean_distance_squared
from metric_learning.loss_functions.latent_position_loss import pairwise_matching_matrix
from metric_learning.loss_functions.latent_position_loss import upper_triangular_part
from metric_learning.loss_functions.latent_position_loss import pairwise_dot_product

tf.enable_eager_execution()


class LatentPositionLossTest(tf.test.TestCase):
    def testPairwiseDifference(self):
        embeddings = tf.constant([
            [0, 1],
            [0, 2],
            [0, 3],
        ])
        y = pairwise_euclidean_distance_squared(embeddings)
        self.assertAllEqual(y, [
            [0, 1, 4],
            [1, 0, 1],
            [4, 1, 0],
        ])

    def testPairwiseDotProduct(self):
        embeddings = tf.constant([
            [0, 1],
            [0, 2],
            [0, 3],
        ])
        y = pairwise_dot_product(embeddings)
        self.assertAllEqual(y, [
            [1, 2, 3],
            [2, 4, 6],
            [3, 6, 9],
        ])

    def testPairwiseMatching(self):
        labels = tf.constant([1, 1, 2, 2, 2, 1])
        y = pairwise_matching_matrix(labels)
        self.assertAllEqual(y, [
            [ 1., 1., -1., -1., -1., 1.],
            [ 1.,  1., -1., -1., -1., 1.],
            [-1., -1., 1., 1., 1., -1.],
            [-1., -1., 1., 1., 1., -1.],
            [-1., -1., 1., 1., 1., -1.],
            [ 1., 1., -1., -1., -1., 1.],
        ])

    def testUpperTriangularPart(self):
        a = tf.constant([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
        b = upper_triangular_part(a)
        self.assertAllEqual(b, [2, 3, 6])


if __name__ == '__main__':
    tf.test.main()
