from util.config import CONFIG

experiment_dir = CONFIG['dataset']['experiment_dir']

configs = {
    'stanford_inception_latent_position_bias': {
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
            'name': 'stanford_online_product',
            'train': {
                'data_directory': '{}/stanford_online_product/train'.format(experiment_dir),
                'batch_size': 32,
                'group_size': 2,
                'num_groups': 16,
                'min_class_size': 2,
            },
            'test': {
                'data_directory': '{}/stanford_online_product/test'.format(experiment_dir),
                'identification': {
                    'num_negative_examples': 5,
                    'num_testcases': 10000,
                },
                'recall': {
                    'num_examples_per_class': 5,
                    'num_testcases': 10000,
                },
            },
        },
        'model': {
            'name': 'inception',
            'loss': {
                'name': 'latent_position',
                'parametrization': 'bias',
                'npair': {
                    'n': 16,
                },
                'alpha': 8,
            },
            'metric': 'euclidean_distance',
        },
        'metrics': [
            {
                'name': 'recall',
                'compute_period': 200,
                'k': [1, 2, 4, 8],
                'batch_size': 48,
            },
            {
                'name': 'accuracy',
                'compute_period': 200,
                'batch_size': 48,
            },
        ],
        'optimizer': {
            'learning_rate': 0.0001,
        },
        'num_epochs': 50,
    },
    'stanford_inception_npair': {
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
            'name': 'stanford_online_product',
            'train': {
                'data_directory': '{}/stanford_online_product/train'.format(experiment_dir),
                'batch_size': 32,
                'group_size': 2,
                'num_groups': 16,
                'min_class_size': 2,
            },
            'test': {
                'data_directory': '{}/stanford_online_product/test'.format(experiment_dir),
                'identification': {
                    'num_testcases': 10000,
                    'num_negative_examples': 5,
                },
                'recall': {
                    'num_examples_per_class': 5,
                    'num_testcases': 10000,
                },
            },
        },
        'model': {
            'name': 'inception',
            'loss': {
                'name': 'npair',
                'n': 16,
                'parametrization': 'dot_product',
            },
            'metric': 'cosine_similarity',
        },
        'metrics': [
            {
                'name': 'recall',
                'compute_period': 200,
                'k': [1, 2, 4, 8],
                'batch_size': 48,
            },
        ],
        'optimizer': {
            'learning_rate': 0.0001,
        },
        'num_epochs': 50,
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
            },
            'random_flip': True,
        },
        'dataset': {
            'name': 'mnist',
            'train': {
                'data_directory': '{}/mnist/train'.format(experiment_dir),
                'batch_size': 8,
                'group_size': 2,
                'num_groups': 4,
                'min_class_size': 2,
            },
            'test': {
                'data_directory': '{}/mnist/test'.format(experiment_dir),
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
                'name': 'contrastive',
                'parametrization': 'bias',
                'alpha': 4.0,
                'npair': {
                    'n': 4,
                },
            },
            'metric': 'euclidean_distance',
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
