from util.config import render_jinja_config
from util.config import generate_config

configs = {
    'stanford_inception_latent_position': {
        'image': render_jinja_config('image'),
        'dataset': render_jinja_config('dataset', dataset='stanford_online_product'),
        'model': render_jinja_config('model_latent_position'),
        'metrics': render_jinja_config('metrics'),
        'trainer': render_jinja_config('trainer'),
    },
    'stanford_inception_npair': {
        'image': render_jinja_config('image'),
        'dataset': render_jinja_config('dataset', dataset='stanford_online_product'),
        'model': render_jinja_config('model_tuplet', parametrization='euclidean_distance'),
        'metrics': render_jinja_config('metrics'),
        'trainer': render_jinja_config('trainer'),
    },
    'mnist_simple_dense': generate_config([
        ('image', 'image', {}),
        ('dataset', 'dataset', {'dataset': 'mnist', 'num_groups': 4, 'num_testcases': 1000}),
        ('model', 'model_simple_dense', {}),
        ('metrics', 'metrics', {'accuracy': True, 'compute_period': 10, 'recall_k': [1, 2]}),
        ('trainer', 'trainer', {}),
    ]),
}
