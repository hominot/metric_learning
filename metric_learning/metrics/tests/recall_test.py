import tensorflow as tf

from metric_learning.metrics.recall import compute_recall
from metric_learning.metrics.recall import count_singletons


tf.enable_eager_execution()


class RecallMetricTest(tf.test.TestCase):
    def testSingletonCount(self):
        labels = [1, 1, 2, 2, 2, 3, 4, 4, 5]
        ret = count_singletons(labels)
        self.assertEqual(ret, 2)

    def testRecall(self):
        embeddings = tf.constant([
            [1., 0.],
            [1., 0.],
            [0., 1.],
            [0., 1.],
        ])
        labels = tf.constant([1, 1, 2, 2], tf.int64)
        data = [(embeddings, labels)]
        for parametrization in ['euclidean_distance', 'dot_product']:
            r = compute_recall(data, [1], parametrization)
            self.assertEqual(r, {1: 1.0})

    def testRecall2(self):
        embeddings = tf.constant([
            [1., 0.],
            [4., 0.],
            [0., 1.],
            [1., 2.],
        ])
        labels = tf.constant([1, 1, 2, 2], tf.int64)
        data = [(embeddings, labels)]
        self.assertEqual(
            compute_recall(data, [1, 2, 3], 'euclidean_distance'),
            {1: 0.5, 2: 0.75, 3: 1.0}
        )
        self.assertEqual(
            compute_recall(data, [1, 2, 3], 'dot_product'),
            {1: 1.0, 2: 1.0, 3: 1.0}
        )

    def testRecallWithSingleton(self):
        embeddings = tf.constant([
            [1., 0.],
            [1., 0.],
            [0., 1.],
            [0., 1.],
            [0., 1.],
        ])
        labels = tf.constant([1, 1, 2, 3, 4], tf.int64)
        data = [(embeddings, labels)]
        r = compute_recall(data, [1], 'euclidean_distance')
        self.assertEqual(r, {1: 1.0})


if __name__ == '__main__':
    tf.test.main()
