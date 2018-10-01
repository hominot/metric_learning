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
                'num_negative_examples': 5,
            },
        },
        'model': {
            'name': 'inception',
            'loss': {
                'name': 'latent_position',
                'parametrization': 'bias',
                'alpha': 4,
            },
        },
        'metrics': [
            {
                'name': 'accuracy',
                'compute_period': 200,
            },
        ],
        'optimizer': {
            'learning_rate': 0.001,
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
                'num_negative_examples': 5,
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
                'name': 'accuracy',
                'compute_period': 200,
                'skip_steps': 10000,
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
                'num_negative_examples': 1,
            },
        },
        'model': {
            'name': 'simple_dense',
            'k': 8,
            'loss': {
                'name': 'latent_position',
                'parametrization': 'dot_product',
                'alpha': -2.0,
            },
        },
        'metrics': [
            {
                'name': 'accuracy',
                'compute_period': 10,
            },
        ],
        'optimizer': {
            'learning_rate': 0.0001,
        },
        'num_epochs': 100,
    },
}
