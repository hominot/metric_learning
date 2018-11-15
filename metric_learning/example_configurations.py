from util.config import generate_config

configs = {
    'mnist_simple_dense': generate_config({
        'image': {},
        'dataset': {'name': 'mnist'},
        'batch_design': {'name': 'grouped', 'group_size': 2, 'batch_size': 16, 'negative_class_mining': True},
        'model': {'name': 'simple_dense', 'dimension': 2},
        'loss': {'name': 'contrastive', 'alpha': 4, 'importance_sampling': True},
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
    'cub200_contrastive': generate_config({
        'image': {},
        'dataset': {'name': 'cub200', 'multiple': 1},
        'batch_design': {'name': 'grouped', 'group_size': 32, 'batch_size': 64},
        'model': {'name': 'resnet50', 'dimension': 128},
        'loss': {'name': 'contrastive', 'alpha': 10.0, 'new_importance_sampling': True, 'l': 256},
        'metrics': {'auc': True, 'recall': True},
        'trainer': {'lr_decay_steps': 100, 'learning_rate': 0.0001, 'num_epochs': 20, 'lr_decay_rate': 0.90},
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
