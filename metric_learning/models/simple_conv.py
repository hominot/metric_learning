import tensorflow as tf


def create_model():
    data_format = 'channels_last'

    layers = tf.keras.layers
    max_pool = layers.MaxPooling2D(
        (2, 2), (2, 2), padding='same', data_format=data_format)

    return tf.keras.Sequential(
        [
            layers.Conv2D(
                32,
                3,
                padding='same',
                data_format=data_format,
                activation=tf.nn.relu),
            max_pool,
            layers.Conv2D(
                64,
                3,
                padding='same',
                data_format=data_format,
                activation=tf.nn.relu),
            max_pool,
            layers.Flatten(),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dropout(0.4),
            layers.Dense(16),
        ])
