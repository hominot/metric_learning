import tensorflow as tf

tf.enable_eager_execution()

from util.registry.model import Model
from util.registry.data_loader import DataLoader

from util.dataset import create_dataset_from_directory

import time

model_conf = {
    'name': 'inception',
    'loss': {
        'name': 'latent_position',
        'method': 'distance',
        'parametrization': 'bias',
    }
}

dataset_conf = {
    'name': 'lfw',
    'train': {
        'data_directory': '/tmp/research/experiment/lfw/train',
        'batch_size': 16,
        'group_size': 2,
        'num_groups': 8,
        'min_class_size': 8,
    },
    'test': {
        'data_directory': '/tmp/research/experiment/lfw/test',
        'num_negative_examples': 5,
    },
}

now = time.time()

print(0, 'start')

model = Model.create(model_conf, {})

print(int(time.time() - now), 'model loaded')

print(int(time.time() - now), 'model loaded')

data_loader: DataLoader = DataLoader.create(dataset_conf)

testing_files, testing_labels = create_dataset_from_directory(
    dataset_conf['test']['data_directory']
)
test_ds = data_loader.create_verification_test_dataset(testing_files, testing_labels).batch(32)

print(int(time.time() - now), 'test data loaded')

count = 0
for a, p, n in test_ds:
    a = model(a, training=False)
    p = model(p, training=False)
    n = tf.stack([model(nn, training=False) for nn in n])
    count += 1

print(int(time.time() - now), 'test data iteration done')
print(count)
