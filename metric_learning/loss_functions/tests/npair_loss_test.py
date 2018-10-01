import tensorflow as tf

from metric_learning.loss_functions.npair_loss import sample_npair
from metric_learning.loss_functions.npair_loss import compute_exponents


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

    def testComputeExponents(self):
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
        exponents = compute_exponents(first_images, second_images)
        self.assertAllEqual(exponents, [
            [0., 2., 4., 6.],
            [-6., 0., 6., 12.],
            [-20., -10., 0., 10.],
            [-42., -28., -14., 0.],
        ])


if __name__ == '__main__':
    tf.test.main()
