from util.config import generate_config

configs = {
    'mnist_simple_dense': generate_config({
        'image': {},
        'dataset': {'name': 'mnist'},
        'batch_design': {'name': 'grouped', 'group_size': 2, 'batch_size': 16, 'npair': 2},
        'model': {'name': 'simple_dense', 'dimension': 2},
        'loss': {'name': 'contrastive', 'alpha': 4, 'importance_sampling': True},
        'metrics': {'auc': True},
        'trainer': {'lr_decay_steps': 10},
    }),
}
