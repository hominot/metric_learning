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
                'data_directory': '/tmp/research/experiment/stanford_online_product/train',
                'batch_size': 32,
                'group_size': 2,
                'num_groups': 8,
                'min_class_size': 2,
            },
            'test': {
                'data_directory': '/tmp/research/experiment/stanford_online_product/test',
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
                'data_directory': '/tmp/research/experiment/stanford_online_product/train',
                'batch_size': 32,
                'group_size': 2,
                'num_groups': 16,
                'min_class_size': 2,
            },
            'test': {
                'data_directory': '/tmp/research/experiment/stanford_online_product/test',
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
                'name': 'npair',
                'n': 4,
                'parametrization': 'euclidean_distance',
                'alpha': 4.0,
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
