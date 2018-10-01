import tensorflow as tf

tf.enable_eager_execution()

from util.registry.model import Model
from util.registry.data_loader import DataLoader

from util.dataset import create_dataset_from_directory

import time

tfe = tf.contrib.eager

conf = {
    'model': {
        'name': 'inception',
        'loss': {
            'name': 'latent_position',
            'method': 'distance',
            'parametrization': 'bias',
        },
    },
    'image': {
        'width': 250,
        'height': 250,
        'channel': 3,
    },
    'dataset': {
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
    },
}

now = time.time()

print(0, 'start')

model = Model.create(conf['model']['name'], conf)
model.call = tfe.defun(model.call)

print(int(time.time() - now), 'model loaded')

data_loader: DataLoader = DataLoader.create(conf['dataset']['name'], conf)

testing_files, testing_labels = create_dataset_from_directory(
    conf['dataset']['test']['data_directory']
)
test_ds = data_loader.create_verification_test_dataset(testing_files, testing_labels).batch(32).prefetch(32)

print(int(time.time() - now), 'test data loaded')

count = 0
for _ in range(3):
    for a, p, n in test_ds:
        a = model(a, training=False)
        p = model(p, training=False)
        n = tf.stack([model(nn) for nn in n])
        count += 1
    print(int(time.time() - now), 'test data iteration done')

print(count)
