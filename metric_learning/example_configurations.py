from util.config import generate_config

configs = {
    'mnist_simple_dense': generate_config({
        'image': {},
        'dataset': {'name': 'mnist'},
        'batch_design': {'name': 'grouped', 'group_size': 2, 'batch_size': 16},
        'model': {'name': 'simple_dense', 'dimension': 2},
        'loss': {'name': 'contrastive', 'gamma': 1, 'importance_sampling': True},
        'metrics': {'vrf': True, 'vrf_k': [1, 2]},
        'trainer': {'lr_decay_steps': 10},
    }),
    'cars196_npair': generate_config({
        'image': {},
        'dataset': {'name': 'cars196'},
        'batch_design': {'name': 'grouped', 'group_size': 2, 'batch_size': 64, 'npair': 32},
        'model': {'name': 'resnet50', 'dimension': 128},
        'loss': {'name': 'npair', 'lambda': 0.001},
        'metrics': {'auc': True, 'recall': True, 'nmi': True, 'vrf': True},
        'trainer': {'lr_decay_steps': 100, 'learning_rate': 0.0001, 'lr_embedding': 0.0001, 'num_epochs': 20, 'lr_decay_rate': 0.9},
    }),
    'cub200_npair': generate_config({
        'image': {},
        'dataset': {'name': 'cub200'},
        'batch_design': {'name': 'grouped', 'group_size': 2, 'batch_size': 64, 'npair': 32},
        'model': {'name': 'resnet50', 'dimension': 128},
        'loss': {'name': 'npair', 'lambda': 0.001, 'importance_sampling': True},
        'metrics': {'auc': True, 'recall': True, 'nmi': True},
        'trainer': {'lr_decay_steps': 100, 'learning_rate': 0.0001, 'lr_embedding': 0.0001, 'num_epochs': 20, 'lr_decay_rate': 0.9},
    }),
    'cub200_margin': generate_config({
        'image': {},
        'dataset': {'name': 'cub200'},
        'batch_design': {'name': 'grouped', 'group_size': 4, 'batch_size': 64},
        'model': {'name': 'resnet50', 'dimension': 128, 'l2_normalize': True},
        'loss': {'name': 'margin'},
        'metrics': {'auc': True, 'recall': True, 'nmi': True},
        'trainer': {'lr_decay_steps': 100, 'learning_rate': 0.0001, 'lr_embedding': 0.0001, 'num_epochs': 20, 'lr_decay_rate': 0.9},
    }),
    'cub200_triplet': generate_config({
        'image': {},
        'dataset': {'name': 'cub200'},
        'batch_design': {'name': 'grouped', 'group_size': 4, 'batch_size': 64},
        'model': {'name': 'resnet50', 'dimension': 128, 'l2_normalize': True},
        'loss': {'name': 'triplet', 'alpha': 1.0},
        'metrics': {'auc': True, 'recall': True, 'nmi': True},
        'trainer': {'lr_decay_steps': 100, 'learning_rate': 0.00003, 'num_epochs': 20, 'lr_decay_rate': 0.9},
    }),
}
