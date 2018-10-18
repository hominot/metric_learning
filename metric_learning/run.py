import tensorflow as tf

import argparse
import copy
import math
import os

from tqdm import tqdm
from util.registry.data_loader import DataLoader
from util.dataset import load_images_from_directory
from util.dataset import create_test_dataset
from util.registry.model import Model
from util.registry.metric import Metric
from util.logging import set_tensorboard_writer
from util.logging import upload_tensorboard_log_to_s3
from util.logging import create_checkpoint
from util.logging import save_config
from metric_learning.example_configurations import configs
from util.config import CONFIG


def evaluate(model, test_dataset, num_testcases, desc='test'):
    with tf.contrib.summary.always_record_summaries():
        for metric_conf in model.conf['metrics']:
            metric = Metric.create(metric_conf['name'], conf)
            score = metric.compute_metric(model, test_dataset, num_testcases)
            if type(score) is dict:
                for metric, s in score.items():
                    tf.contrib.summary.scalar('test ' + metric, s)
                    print('{} {}: {}'.format(desc, metric, s))
            else:
                tf.contrib.summary.scalar('{} {}'.format(desc, metric_conf['name']), score)
                print('{} {}: {}'.format(desc, metric_conf['name'], score))


def train(conf):
    data_loader = DataLoader.create(conf['dataset']['name'], conf)
    training_files, training_labels = load_images_from_directory(
        os.path.join(CONFIG['dataset']['experiment_dir'], conf['dataset']['name'], 'train'),
        splits=set(range(CONFIG['dataset'].getint('cross_validation_splits'))) - {conf['dataset']['cross_validation_split']},
    )

    extra_info = {
        'num_labels': max(training_labels) + 1,
        'num_images': len(training_files),
    }

    test_dir = os.path.join(CONFIG['dataset']['experiment_dir'],
                            conf['dataset']['name'],
                            'test')
    validation_dir = os.path.join(CONFIG['dataset']['experiment_dir'],
                                  conf['dataset']['name'],
                                  'train',
                                  str(conf['dataset']['cross_validation_split']))
    test_dataset, test_num_testcases = create_test_dataset(
        conf, data_loader, test_dir)
    validation_dataset, validation_num_testcases = create_test_dataset(
        conf, data_loader, validation_dir)

    step_counter = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=conf['trainer']['learning_rate'])
    model = Model.create(conf['model']['name'], conf, extra_info)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     model=model,
                                     optimizer_step=tf.train.get_or_create_global_step())

    writer, run_name = set_tensorboard_writer(conf)
    writer.set_as_default()

    device = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'

    save_config(conf, run_name)

    train_conf = conf['dataset']['train']
    evaluate(model, test_dataset, test_num_testcases, 'test')
    evaluate(model, validation_dataset, validation_num_testcases, 'validate')
    for epoch in range(conf['trainer']['num_epochs']):
        train_ds, num_examples = data_loader.create_grouped_dataset(
            training_files, training_labels,
            group_size=train_conf['group_size'],
            num_groups=train_conf['num_groups'],
            min_class_size=train_conf['min_class_size'],
        )
        train_ds = train_ds.batch(train_conf['batch_size'], drop_remainder=True)
        with tf.device(device):
            batches = tqdm(train_ds,
                           total=math.ceil(num_examples / train_conf['batch_size']),
                           desc='epoch #{}'.format(epoch + 1))
            for (batch, (images, labels, image_ids)) in enumerate(batches):
                with tf.contrib.summary.record_summaries_every_n_global_steps(
                        10, global_step=step_counter):
                    with tf.GradientTape() as tape:
                        loss_value = model.loss(images, labels, image_ids)
                        tf.contrib.summary.scalar('loss', loss_value)

                    grads = tape.gradient(loss_value, model.variables)
                    optimizer.apply_gradients(
                        zip(grads, model.variables), global_step=step_counter)
                    if CONFIG['tensorboard'].getboolean('s3_upload') and int(step_counter) % int(CONFIG['tensorboard']['s3_upload_period']) == 0:
                        upload_tensorboard_log_to_s3(run_name)
            print('epoch #{} checkpoint: {}'.format(epoch + 1, run_name))
            if CONFIG['tensorboard'].getboolean('enable_checkpoint'):
                create_checkpoint(checkpoint, run_name)
        evaluate(model, test_dataset, test_num_testcases, 'test')
        evaluate(model, validation_dataset, validation_num_testcases, 'validate')


if __name__ == '__main__':
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser(description='Train using a specified config')
    parser.add_argument('--config', help='config to run')
    parser.add_argument('--split',
                        help='cross validation split number to use as validation data',
                        default=None,
                        type=int)
    args = parser.parse_args()
    conf = copy.deepcopy(configs[args.config])
    if args.split is not None:
        conf['dataset']['cross_validation_split'] = args.split

    train(conf)
