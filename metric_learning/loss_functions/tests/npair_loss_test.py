import tensorflow as tf

from metric_learning.loss_functions.npair_loss import sample_npair
from metric_learning.loss_functions.npair_loss import compute_dot_product_exponents
from metric_learning.loss_functions.npair_loss import compute_euclidean_distance_exponents


tf.enable_eager_execution()


class NpairLossTest(tf.test.TestCase):
    def testNpairSampling(self):
        images = tf.constant([
            [0., 1.],
            [0., 2.],
            [0., 3.],
            [0., 4.],
            [0., 5.],
            [0., 6.],
            [0., 7.],
            [0., 8.],
        ])
        labels = tf.constant([
            0, 0, 1, 1, 2, 2, 3, 3, 4, 4
        ])
        first_images, second_images = sample_npair(images, labels, 4)[0]
        self.assertAllEqual(first_images, [[0., 1.], [0., 3.], [0., 5.], [0., 7.]])
        self.assertAllEqual(second_images, [[0., 2.], [0., 4.], [0., 6.], [0., 8.]])

    def testComputeDotProductExponents(self):
        images = tf.constant([
            [0., 1.],
            [0., 2.],
            [0., 3.],
            [0., 4.],
            [0., 5.],
            [0., 6.],
            [0., 7.],
            [0., 8.],
        ])
        labels = tf.constant([
            0, 0, 1, 1, 2, 2, 3, 3, 4, 4
        ])
        first_images, second_images = sample_npair(images, labels, 4)[0]
        exponents = compute_dot_product_exponents(first_images, second_images)
        self.assertAllEqual(exponents, [
            [0., 2., 4., 6.],
            [0., -6., 6., 12.],
            [0., -20., -10., 10.],
            [0., -42., -28., -14.],
        ])

    def testComputeEuclideanDistanceExponents(self):
        images = tf.constant([
            [0., 1.],
            [0., 2.],
            [0., 3.],
            [0., 4.],
            [0., 5.],
            [0., 6.],
            [0., 7.],
            [0., 8.],
        ])
        labels = tf.constant([
            0, 0, 1, 1, 2, 2, 3, 3, 4, 4
        ])
        first_images, second_images = sample_npair(images, labels, 4)[0]
        exponents = compute_euclidean_distance_exponents(first_images, second_images)
        self.assertAllEqual(exponents, [
            [  0.,   0.,  -8., -24.],
            [  0.,  -8.,   0.,  -8.],
            [  0., -24.,  -8.,   0.],
            [  0., -48., -24.,  -8.],
        ])


if __name__ == '__main__':
    tf.test.main()
