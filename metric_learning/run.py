import tensorflow as tf

import argparse
import json

from util.registry.data_loader import DataLoader
from util.dataset import split_train_test_by_label
from util.dataset import create_dataset_from_directory
from util.registry.model import Model
from util.registry.metric import Metric
from util.logging import set_tensorboard_writer
from util.logging import upload_tensorboard_log_to_s3
from util.logging import upload_checkpoint_to_s3
from util.logging import upload_string_to_s3
from metric_learning.configurations import configs
from util.config import config


def train(conf):
    data_loader: DataLoader = DataLoader.create(conf['dataset']['name'], conf)
    if conf['dataset']['train']['data_directory'] and conf['dataset']['test']['data_directory']:
        training_files, training_labels = create_dataset_from_directory(
            conf['dataset']['train']['data_directory']
        )
        testing_files, testing_labels = create_dataset_from_directory(
            conf['dataset']['test']['data_directory']
        )
    else:
        image_files, labels = data_loader.load_image_files()
        training_data, testing_data = split_train_test_by_label(image_files, labels)
        training_files, training_labels = zip(*training_data)
        testing_files, testing_labels = zip(*testing_data)

    extra_info = {
        'num_labels': max(training_labels),
    }

    test_datasets = {}
    if 'identification' in conf['dataset']['test']:
        dataset, num_testcases = data_loader.create_identification_test_dataset(
            testing_files, testing_labels)
        test_datasets['identification'] = {
            'dataset': dataset,
            'num_testcases': num_testcases,
        }
    if 'recall' in conf['dataset']['test']:
        images_ds, labels_ds, num_testcases = data_loader.create_recall_test_dataset(
            testing_files, testing_labels)
        test_datasets['recall'] = {
            'dataset': (images_ds, labels_ds),
            'num_testcases': num_testcases,
        }

    step_counter = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=conf['optimizer']['learning_rate'])
    model = Model.create(conf['model']['name'], conf, extra_info)

    writer, run_name = set_tensorboard_writer(model, data_loader)
    writer.set_as_default()

    device = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'

    if config['tensorboard']['s3_upload']:
        upload_string_to_s3(
            bucket=config['tensorboard']['s3_bucket'],
            body=json.dumps(conf, indent=4),
            key='{}/experiments/{}/config.json'.format(config['tensorboard']['s3_key'], run_name)
        )
    for _ in range(conf['num_epochs']):
        train_ds = data_loader.create_grouped_dataset(
            training_files, training_labels,
            group_size=conf['dataset']['train']['group_size'],
            num_groups=conf['dataset']['train']['num_groups'],
            min_class_size=conf['dataset']['train']['min_class_size'],
        ).batch(conf['dataset']['train']['batch_size'])
        with tf.device(device):
            for (batch, (images, labels)) in enumerate(train_ds):
                with tf.contrib.summary.record_summaries_every_n_global_steps(
                        10, global_step=step_counter):
                    current_step = int(step_counter)
                    for metric_conf in conf['metrics']:
                        if current_step % metric_conf.get('compute_period', 10) == 0 and \
                            current_step >= metric_conf.get('skip_steps', 0):
                            metric = Metric.create(metric_conf['name'], conf)
                            score = metric.compute_metric(model, test_datasets[metric.dataset]['dataset'], test_datasets[metric.dataset]['num_testcases'])
                            if type(score) is dict:
                                for metric, s in score.items():
                                    tf.contrib.summary.scalar(metric, s)
                            else:
                                tf.contrib.summary.scalar(metric_conf['name'], score)
                    with tf.GradientTape() as tape:
                        loss_value = model.loss(images, labels)
                        tf.contrib.summary.scalar('loss', loss_value)

                    if hasattr(model, 'alpha'):
                        tf.contrib.summary.scalar('alpha', model.alpha)
                    if hasattr(model, 'beta'):
                        tf.contrib.summary.scalar('beta', model.beta)
                    grads = tape.gradient(loss_value, model.variables)
                    optimizer.apply_gradients(
                        zip(grads, model.variables), global_step=step_counter)
                    if config['tensorboard']['s3_upload'] and int(step_counter) % int(config['tensorboard']['s3_upload_period']) == 0:
                        upload_tensorboard_log_to_s3(run_name)
                    if int(step_counter) % int(config['tensorboard']['checkpoint_period']) == 0:
                        upload_checkpoint_to_s3(model, optimizer, step_counter, run_name)


if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser(description='Train using a specified config')
    parser.add_argument('--config', help='config to run')
    args = parser.parse_args()

    train(configs[args.config])
