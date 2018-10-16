from util.config import render_jinja_config
from util.config import generate_config

configs = {
    'stanford_inception_latent_position': {
        'image': render_jinja_config('image'),
        'dataset': render_jinja_config('dataset', dataset='stanford_online_product'),
        'model': render_jinja_config('model', name='inception'),
        'loss': render_jinja_config('loss', name='latent_position'),
        'metrics': render_jinja_config('metrics'),
        'trainer': render_jinja_config('trainer'),
    },
    'stanford_inception_npair': {
        'image': render_jinja_config('image'),
        'dataset': render_jinja_config('dataset', dataset='stanford_online_product'),
        'model': render_jinja_config('model', name='inception'),
        'loss': render_jinja_config('loss', name='npair'),
        'metrics': render_jinja_config('metrics'),
        'trainer': render_jinja_config('trainer'),
    },
    'cub200': generate_config({
        'image': {},
        'dataset': {'dataset': 'cub200', 'num_groups': 16, 'batch_size': 64, 'group_size': 4},
        'model': {'name': 'resnet50', 'dimension': 128},
        'loss': {'name': 'margin', 'parametrization': 'euclidean_distance'},
        'metrics': {},
        'trainer': {},
    }),
    'mnist_simple_dense': generate_config({
        'image': {},
        'dataset': {'dataset': 'mnist', 'num_groups': 4},
        'model': {'name': 'simple_dense', 'dimension': 2},
        'loss': {'name': 'margin'},
        'metrics': {'compute_period': 10, 'recall_k': [1, 2]},
        'trainer': {},
    }),
}
