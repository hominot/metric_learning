import tensorflow as tf

tf.enable_eager_execution()

from mnist.dataset import dataset
from mnist.model import create_model


"""
def vectorize(input, weights, biases):
    return input * weights + biases


def loss(weights, biases):
    error = prediction(training_inputs, weights, biases) - training_outputs
    return tf.reduce_mean(tf.square(error))


def grad(weights, biases):
    with tf.GradientTape() as tape:
        loss_value = loss(weights, biases)
    return tape.gradient(loss_value, [weights, biases])
"""


def loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))


def compute_accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
    labels = tf.cast(labels, tf.int64)
    batch_size = int(logits.shape[0])
    return tf.reduce_sum(
        tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size


def train(directory):
    ds = dataset(
        directory,
        'train-images-idx3-ubyte',
        'train-labels-idx1-ubyte')

    optimizer = tf.train.MomentumOptimizer(0.01, 0.5)

    model = create_model('channels_last')
    step_counter = tf.train.get_or_create_global_step()

    train_ds = ds.shuffle(60000).batch(500)
    for (batch, (images, labels)) in enumerate(train_ds):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss_value = loss(logits, labels)
            tf.contrib.summary.scalar('loss', loss_value)
            tf.contrib.summary.scalar('accuracy', compute_accuracy(logits, labels))
        grads = tape.gradient(loss_value, model.variables)
        optimizer.apply_gradients(
            zip(grads, model.variables), global_step=step_counter)
        print('Step #%d\tLoss: %.6f' % (batch, loss_value))

    return ds


if __name__ == '__main__':
    train('__data__')
