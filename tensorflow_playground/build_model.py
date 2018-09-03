import tensorflow as tf


class MySimpleLayer(tf.keras.layers.Layer):
    def __init__(self, output_units):
        super(MySimpleLayer, self).__init__()
        self.output_units = output_units

    def build(self, input_shape):
        self.kernel = self.add_variable(
            'kernel', [input_shape[-1], self.output_units]
        )

    def call(self, input):
        return tf.matmul(input, self.kernel)


class MNISTModel(tf.keras.Model):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units=10)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, input):
        result = self.dense1(input)
        result = self.dense2(result)
        result = self.dense2(result)
        return result


model = MNISTModel()