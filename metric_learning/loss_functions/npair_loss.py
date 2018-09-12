import tensorflow as tf

from collections import defaultdict
from util.loss_function import LossFunction


def sample_npair(images, labels, n):
    ret = []
    for batch in range(int(images.shape[0]) // (n * 2)):
        cur_labels = labels[batch * n * 2: (batch + 1) * n * 2]
        cur_images = images[batch * n * 2: (batch + 1) * n * 2]
        data_map = defaultdict(list)
        for index, label in enumerate(cur_labels):
            data_map[int(label)].append(index)

        anchor_images = []
        positive_images = []
        negative_images = []
        for label in data_map.keys():
            anchor_images.append(cur_images[data_map[label][0]])
            positive_images.append(cur_images[data_map[label][1]])
            for negative_label, negative_index in data_map.items():
                if label == negative_label:
                    continue
                negative_images.append(cur_images[data_map[negative_label][1]])

        ret.append((
            tf.stack(anchor_images),
            tf.stack(positive_images),
            tf.stack(negative_images),
        ))
    return ret


class NPairLossFunction(LossFunction):
    name = 'npair'

    def loss(self, embeddings, labels, *args, **kwargs):
        sampled_data = sample_npair(embeddings, labels, kwargs['n'])

        losses = []
        for anchor_images, positive_images, negative_images in sampled_data:
            losses.append(tf.reduce_mean(
                tf.reduce_sum(
                    tf.log(
                        1 + \
                        tf.exp(
                            tf.matmul(anchor_images, tf.transpose(negative_images)) - \
                            tf.reduce_sum(tf.multiply(anchor_images, positive_images), axis=1, keepdims=True)
                        )
                    ),
                    axis=1
                )
            ))
        return sum(losses)

