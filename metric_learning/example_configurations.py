from util.config import generate_config

configs = {
    'mnist_simple_dense': generate_config({
        'image': {},
        'dataset': {'name': 'mnist'},
        'batch_design': {'name': 'grouped', 'group_size': 2, 'batch_size': 4, 'npair': 2, 'negative_class_mining': True},
        'model': {'name': 'simple_dense', 'dimension': 2},
        'loss': {'name': 'npair', 'alpha': 4, 'importance_sampling': True},
        'metrics': {'vrf': True, 'vrf_k': [1, 2]},
        'trainer': {'lr_decay_steps': 10},
    }),
    'cub200_npair': generate_config({
        'image': {},
        'dataset': {'name': 'cub200'},
        'batch_design': {'name': 'grouped', 'group_size': 2, 'batch_size': 64, 'npair': 32},
        'model': {'name': 'resnet50', 'dimension': 128},
        'loss': {'name': 'npair', 'lambda': 0.1, 'importance_sampling': False},
        'metrics': {'auc': True, 'recall': True, 'nmi': True},
        'trainer': {'lr_decay_steps': 100, 'learning_rate': 0.0001, 'lr_embedding': 0.0001, 'num_epochs': 20, 'lr_decay_rate': 0.9},
    }),
}
