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
                'compute_period': 10,
                'conf': {
                    'sampling_rate': 0.1,
                }
            },
        ]
    },
    'mnist_latent_position': {
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
            'name': 'latent_position',
            'method': 'projection',
            'child_model': {
                'name': 'simple_dense',
                'k': 4,
            },
        },
        'metrics': [
            {
                'name': 'accuracy',
                'compute_period': 10,
                'conf': {
                    'sampling_rate': 0.1,
                }
            },
        ]
    },
}
