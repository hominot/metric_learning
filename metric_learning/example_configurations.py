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
        'model': {'name': 'resnet50', 'dimension': 128, 'l2_normalize': True},
        'loss': {'name': 'contrastive', 'parametrization': 'euclidean_distance', 'alpha': 4.0},
        'metrics': {},
        'trainer': {'learning_rate': 0.00001, 'num_epochs': 200},
    }),
    'mnist_simple_dense': generate_config({
        'image': {},
        'dataset': {'dataset': 'mnist', 'num_groups': 2, 'group_size': 2, 'batch_size': 16},
        'model': {'name': 'simple_dense', 'dimension': 2},
        'loss': {'name': 'npair', 'parametrization': 'dot_product', 'alpha': 4, 'npair': True},
        'metrics': {'recall_k': [1, 2]},
        'trainer': {'lr_decay_steps': 10},
    }),
}
