import tensorflow as tf

from util.tensor_operations import _pairwise_euclidean_distance_squared
from util.tensor_operations import pairwise_matching_matrix
from util.tensor_operations import upper_triangular_part
from util.tensor_operations import pairwise_dot_product
from util.tensor_operations import repeat_columns
from util.tensor_operations import pairwise_difference
from util.tensor_operations import pairwise_cosine_similarity
from util.tensor_operations import off_diagonal_part
from util.tensor_operations import get_n_blocks
from util.tensor_operations import pairwise_product

tf.enable_eager_execution()


class TensorOperationsTest(tf.test.TestCase):
    def testPairwiseEuclideanDifference(self):
        embeddings = tf.constant([
            [0, 1],
            [0, 2],
            [0, 3],
        ])
        y = _pairwise_euclidean_distance_squared(embeddings, embeddings)
        self.assertAllEqual(y, [
            [0, 1, 4],
            [1, 0, 1],
            [4, 1, 0],
        ])

    def testPairwiseDifference2(self):
        first = tf.constant([
            [0, 1],
            [0, 2],
            [0, 3],
        ])
        second = tf.constant([
            [0, 1],
        ])
        y = _pairwise_euclidean_distance_squared(first, second)
        self.assertAllEqual(y, [
            [0], [1], [4],
        ])

    def testPairwiseDotProduct(self):
        embeddings = tf.constant([
            [0, 1],
            [0, 2],
            [0, 3],
        ])
        y = pairwise_dot_product(embeddings, embeddings)
        self.assertAllEqual(y, [
            [1, 2, 3],
            [2, 4, 6],
            [3, 6, 9],
        ])

    def testPairwiseMatching(self):
        labels = tf.constant([1, 1, 2, 2, 2, 1])
        y = pairwise_matching_matrix(labels, labels)
        self.assertAllEqual(y, [
            [True, True, False, False, False, True],
            [True, True, False, False, False, True],
            [False, False, True, True, True, False],
            [False, False, True, True, True, False],
            [False, False, True, True, True, False],
            [True, True, False, False, False, True],
        ])

    def testPairwiseMatching2(self):
        first = tf.constant([1, 1, 2])
        second = tf.constant([2, 1])
        y = pairwise_matching_matrix(first, second)
        self.assertAllEqual(y, [
            [False, True],
            [False, True],
            [True, False],
        ])

    def testUpperTriangularPart(self):
        a = tf.constant([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
        b = upper_triangular_part(a)
        self.assertAllEqual(b, [2, 3, 6])

    def testRepeatColumn(self):
        a = tf.constant([1, 2, 3])
        b = repeat_columns(a)
        self.assertAllEqual(b, [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ])

    def testPairwiseDifference(self):
        a = tf.constant([1, 2, 3])
        b = tf.constant([1, 2])
        c = pairwise_difference(a, b)
        self.assertAllEqual(c, [
            [0, -1],
            [1, 0],
            [2, 1],
        ])

    def testPairwiseCosineSimilarity(self):
        embeddings = tf.constant([
            [0., 1.],
            [1., 0.],
        ])
        c = pairwise_cosine_similarity(embeddings, embeddings)
        self.assertAllEqual(c, [
            [1., 0.],
            [0., 1.],
        ])

    def testOffDiagonalPart(self):
        a = tf.constant([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
        b = off_diagonal_part(a)
        self.assertAllEqual(b, [2, 3, 4, 6, 7, 8])

    def testNBlocks(self):
        a = tf.constant([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ])
        self.assertAllEqual(get_n_blocks(a, 2), [
            [1, 2],
            [5, 6],
            [11, 12],
            [15, 16],
        ])
        self.assertAllEqual(get_n_blocks(a, 1), [
            [1],
            [6],
            [11],
            [16],
        ])

        b = tf.constant([
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8],
            [4, 5, 6, 7, 8, 9],
            [5, 6, 7, 8, 9, 10],
            [6, 7, 8, 9, 10, 11],
        ])
        self.assertAllEqual(get_n_blocks(b, 2), [
            [1, 2],
            [2, 3],
            [5, 6],
            [6, 7],
            [9, 10],
            [10, 11],
        ])
        self.assertAllEqual(get_n_blocks(b, 3), [
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [7, 8, 9],
            [8, 9, 10],
            [9, 10, 11],
        ])

    def testPairwiseProduct(self):
        a = tf.constant([1, 2, 3])
        b = tf.constant([2, 3, 4])
        self.assertAllEqual(pairwise_product(a, b), [
            [2, 3, 4],
            [4, 6, 8],
            [6, 9, 12],
        ])


if __name__ == '__main__':
    tf.test.main()
