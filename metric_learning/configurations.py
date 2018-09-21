configs = {
    'mnist_npair': {
        'dataset': {
            'name': 'mnist',
            'train': {
                'data_directory': '/tmp/research/experiment/mnist/train',
                'batch_size': 64,
                'group_size': 2,
                'num_groups': 8,
                'min_class_size': 8,
            },
            'test': {
                'data_directory': '/tmp/research/experiment/mnist/test',
                'num_negative_examples': 1,
            },
        },
        'model': {
            'name': 'simple_dense',
            'k': 4,
            'loss': {
                'name': 'npair',
                'n': 4
            }
        },
        'metrics': [
            {
                'name': 'accuracy',
                'compute_period': 100,
            },
        ]
    },
    'lfw_latent_position': {
        'dataset': {
            'name': 'lfw',
            'train': {
                'data_directory': '/tmp/research/experiment/lfw/train',
                'batch_size': 64,
                'group_size': 4,
                'num_groups': 8,
                'min_class_size': 8,
            },
            'test': {
                'data_directory': '/tmp/research/experiment/lfw/test',
                'num_negative_examples': 5,
            },
        },
        'model': {
            'name': 'latent_position',
            'method': 'projection',
            'child_model': {
                'name': 'simple_conv',
                'k': 8,
            },
        },
        'metrics': [
            {
                'name': 'accuracy',
                'compute_period': 100,
            },
            {
                'name': 'norm',
                'compute_period': 100,
            },
        ]
    },
    'lfw_npair': {
        'dataset': {
            'name': 'lfw',
            'train': {
                'data_directory': '/tmp/research/experiment/lfw/train',
                'batch_size': 64,
                'group_size': 2,
                'num_groups': 8,
                'min_class_size': 8,
            },
            'test': {
                'data_directory': '/tmp/research/experiment/lfw/test',
                'num_negative_examples': 5,
            },
        },
        'model': {
            'name': 'simple_conv',
            'k': 8,
            'loss': {
                'name': 'npair',
                'n': 8,
            },
        },
        'metrics': [
            {
                'name': 'accuracy',
                'compute_period': 100,
            },
            {
                'name': 'norm',
                'compute_period': 100,
            },
        ]
    },
}
