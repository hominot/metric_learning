import tensorflow as tf

from util.distribution import factor_expansion
from util.distribution import wallenius

tf.enable_eager_execution()


class TensorOperationsTest(tf.test.TestCase):
    def testFactorExpansion(self):
        exponents = [1, 2, 3]
        print(factor_expansion(exponents))
        self.assertTrue(True)

    def testWallenius(self):
        weights = [1, 1, 1, 1, 1, 1]
        self.assertAllClose(wallenius([0], weights), 1 / 6)
        self.assertAllClose(wallenius([0, 1], weights), 1 / 15)
        self.assertAllClose(wallenius([0, 1, 2], weights), 1 / 20)
        self.assertAllClose(wallenius([0, 1, 2, 3], weights), 1 / 15)
        self.assertAllClose(wallenius([0, 1, 2, 3, 4], weights), 1 / 6)


if __name__ == '__main__':
    tf.test.main()
