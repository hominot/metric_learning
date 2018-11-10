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
}
