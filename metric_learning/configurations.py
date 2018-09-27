configs = {
    'lfw_latent_position_distance': {
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
            'method': 'distance',
            'child_model': {
                'name': 'simple_conv',
                'k': 8,
            },
        },
        'metrics': [
            {
                'name': 'accuracy',
                'compute_period': 200,
            },
        ],
        'optimizer': {
            'learning_rate': 0.0001,
        },
        'num_epochs': 200,
    },
    'lfw_latent_position_projection': {
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
                'compute_period': 200,
            },
        ],
        'optimizer': {
            'learning_rate': 0.0001,
        },
        'num_epochs': 200,
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
            }
        },
        'metrics': [
            {
                'name': 'accuracy',
                'compute_period': 200,
            },
        ],
        'optimizer': {
            'learning_rate': 0.0001,
        },
        'num_epochs': 200,
    },
    'lfw_inception_latent_position': {
        'dataset': {
            'name': 'lfw',
            'train': {
                'data_directory': '/tmp/research/experiment/lfw/train',
                'batch_size': 32,
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
            'name': 'inception',
            'loss': {
                'name': 'latent_position',
                'method': 'distance',
            }
        },
        'metrics': [
            {
                'name': 'accuracy',
                'compute_period': 200,
                'sampling_rate': 0.1,
            },
        ],
        'optimizer': {
            'learning_rate': 0.001,
        },
        'num_epochs': 200,
    },
    'lfw_inception_npair': {
        'dataset': {
            'name': 'lfw',
            'train': {
                'data_directory': '/tmp/research/experiment/lfw/train',
                'batch_size': 32,
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
            'name': 'inception',
            'loss': {
                'name': 'npair',
                'n': 8,
            }
        },
        'metrics': [
        ],
        'optimizer': {
            'learning_rate': 0.001,
        },
        'num_epochs': 200,
    },
}
