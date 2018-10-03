configs = {
    'lfw_inception_latent_position_bias': {
        'image': {
            'width': 250,
            'height': 250,
            'channel': 3,
            'random_crop': {
                'width': 224,
                'height': 224,
                'n': 8,
            },
        },
        'dataset': {
            'name': 'lfw',
            'train': {
                'data_directory': '/tmp/research/experiment/lfw/train',
                'batch_size': 32,
                'group_size': 2,
                'num_groups': 8,
                'min_class_size': 2,
            },
            'test': {
                'data_directory': '/tmp/research/experiment/lfw/test',
                'identification': {
                    'num_negative_examples': 5,
                    'num_testcases': 10000,
                    'batch_size': 48,
                },
                'recall': {
                    'num_examples_per_class': 5,
                    'num_testcases': 10000,
                    'batch_size': 48,
                },
            },
        },
        'model': {
            'name': 'inception',
            'loss': {
                'name': 'latent_position',
                'parametrization': 'bias',
                'alpha': 8,
            },
        },
        'metrics': [
            {
                'name': 'recall',
                'compute_period': 200,
                'k': [1, 3, 5, 10],
            },
            {
                'name': 'accuracy',
                'compute_period': 200,
            },
        ],
        'optimizer': {
            'learning_rate': 0.0001,
        },
        'num_epochs': 50,
    },
    'lfw_inception_npair': {
        'dataset': {
            'name': 'lfw',
            'image': {
                'width': 250,
                'height': 250,
                'channel': 3,
            },
            'train': {
                'data_directory': '/tmp/research/experiment/lfw/train',
                'batch_size': 32,
                'group_size': 2,
                'num_groups': 8,
                'min_class_size': 2,
            },
            'test': {
                'data_directory': '/tmp/research/experiment/lfw/test',
                'identification': {
                    'num_testcases': 10000,
                    'num_negative_examples': 5,
                    'batch_size': 48,
                },
                'recall': {
                    'num_examples_per_class': 5,
                    'num_testcases': 10000,
                    'batch_size': 48,
                },
            },
        },
        'model': {
            'name': 'inception',
            'loss': {
                'name': 'npair',
                'n': 8,
            },
            'image': {
                'width': 250,
                'height': 250,
                'channel': 3,
            },
        },
        'metrics': [
            {
                'name': 'recall',
                'k': [1, 2, 4, 8],
                'compute_period': 200,
            },
            {
                'name': 'accuracy',
                'compute_period': 200,
            },
        ],
        'optimizer': {
            'learning_rate': 0.00005,
        },
        'num_epochs': 100,
    },
    'mnist_simple_dense': {
        'image': {
            'width': 28,
            'height': 28,
            'channel': 1,
            'random_crop': {
                'width': 26,
                'height': 26,
                'n': 2,
            }
        },
        'dataset': {
            'name': 'mnist',
            'train': {
                'data_directory': '/tmp/research/experiment/mnist/train',
                'batch_size': 32,
                'group_size': 2,
                'num_groups': 4,
                'min_class_size': 2,
            },
            'test': {
                'data_directory': '/tmp/research/experiment/mnist/test',
                'identification': {
                    'num_negative_examples': 1,
                    'num_testcases': 1000,
                },
                'recall': {
                    'num_examples_per_class': 5,
                    'num_testcases': 1000,
                },
            },
        },
        'model': {
            'name': 'simple_dense',
            'k': 8,
            'loss': {
                'name': 'latent_position',
                'parametrization': 'bias',
                'alpha': 4.0,
            },
        },
        'metrics': [
            {
                'name': 'accuracy',
                'compute_period': 10,
                'batch_size': 48,
            },
            {
                'name': 'recall',
                'k': [1, 2],
                'compute_period': 10,
                'batch_size': 48,
            },
        ],
        'optimizer': {
            'learning_rate': 0.0001,
        },
        'num_epochs': 100,
    },
}
