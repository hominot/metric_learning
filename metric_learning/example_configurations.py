from util.config import generate_config

configs = {
    'mnist_simple_dense': generate_config({
        'image': {},
        'dataset': {'name': 'mnist'},
        'batch_design': {'name': 'pair', 'num_groups': 2, 'group_size': 2, 'batch_size': 16},
        'model': {'name': 'simple_dense', 'dimension': 2},
        'loss': {'name': 'logistic', 'alpha': 4},
        'metrics': {'recall_k': [1, 2]},
        'trainer': {'lr_decay_steps': 10},
    }),
}
