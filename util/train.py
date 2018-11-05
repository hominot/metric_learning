import numpy as np
import tensorflow as tf

import json
import math
import os

from collections import defaultdict
from decimal import Decimal
from tqdm import tqdm
from util.dataset import create_test_dataset
from util.dataset import get_training_files_labels
from util.registry.data_loader import DataLoader
from util.registry.batch_design import BatchDesign
from util.registry.model import Model
from util.registry.metric import Metric
from util.logging import set_tensorboard_writer
from util.logging import upload_tensorboard_log_to_s3
from util.logging import create_checkpoint
from util.logging import save_config
from util.logging import db
from util.config import CONFIG


def evaluate(conf, model, test_dataset, num_testcases, train_stat):
    data = {}
    with tf.contrib.summary.always_record_summaries():
        for metric_conf in model.conf['metrics']:
            metric = Metric.create(metric_conf['name'], conf)
            score = metric.compute_metric(model, test_dataset, num_testcases)
            if type(score) is dict:
                for metric, s in score.items():
                    tf.contrib.summary.scalar('test ' + metric, s)
                    print('{}: {}'.format(metric, s))
                    data[metric] = Decimal(str(s))
            else:
                tf.contrib.summary.scalar('{}'.format(metric_conf['name']), score)
                print('{}: {}'.format(metric_conf['name'], score))
                data[metric_conf['name']] = Decimal(str(score))
    if CONFIG['tensorboard'].getboolean('dynamodb_upload'):
        table = db.Table('TrainHistory')
        item = {}
        item.update(train_stat)
        item.update(data)
        table.put_item(Item=item)
    return data


def stopping_criteria(metrics):
    if len(metrics) <= 5:
        return False
    recall_1 = [float(x['recall@1']) for x in metrics]
    max_recall_1 = max(recall_1)
    if recall_1[-1] < max_recall_1 * 0.95:
        return True
    if recall_1[-1] < recall_1[-2] * 0.98 and recall_1[-2] < recall_1[-3] * 0.98:
        return True
    if all([x * 1.003 < max_recall_1 for x in recall_1[-4:]]):
        return True
    return False


def get_metric_to_report(metrics):
    recall_1 = [float(x['recall@1']) for x in metrics]
    return metrics[np.argmax(recall_1)]


def compute_all_embeddings(model, conf, training_files):
    data_loader = DataLoader.create(conf['dataset']['name'], conf)
    ds = BatchDesign.create('vanilla', conf, {'data_loader': data_loader})
    train_ds, num_examples = ds.create_dataset(
        training_files, [0] * len(training_files), testing=True)
    train_ds = train_ds.batch(48)
    batches = tqdm(train_ds,
                   total=math.ceil(num_examples / 48),
                   desc='drift')
    embeddings = []
    for (batch, (images, labels, image_ids)) in enumerate(batches):
        embeddings.append(model(images, training=False))
    return embeddings


def train(conf):
    print(json.dumps(conf, indent=4))
    data_loader = DataLoader.create(conf['dataset']['name'], conf)
    training_files, training_labels = get_training_files_labels(conf)
    if conf['dataset']['cross_validation_split'] != -1:
        validation_dir = os.path.join(CONFIG['dataset']['experiment_dir'],
                                      conf['dataset']['name'],
                                      'train',
                                      str(conf['dataset']['cross_validation_split']))
        test_dataset, test_num_testcases = create_test_dataset(
            conf, data_loader, validation_dir)
    else:
        test_dir = os.path.join(CONFIG['dataset']['experiment_dir'],
                                conf['dataset']['name'],
                                'test')
        test_dataset, test_num_testcases = create_test_dataset(
            conf, data_loader, test_dir)

    writer, run_name = set_tensorboard_writer(conf)
    writer.set_as_default()
    save_config(conf, run_name)

    dataset = BatchDesign.create(
        conf['batch_design']['name'], conf, {'data_loader': data_loader})

    label_counts = [0] * (max(training_labels) + 1)
    for label in training_labels:
        label_counts[label] += 1
    extra_info = {
        'num_labels': max(training_labels) + 1,
        'num_images': len(training_files),
        'label_counts': label_counts,
    }
    model = Model.create(conf['model']['name'], conf, extra_info)
    optimizers = {
        k: tf.train.AdamOptimizer(learning_rate=v) for
        k, (v, _) in model.learning_rates().items()
    }
    checkpoint = tf.train.Checkpoint(model=model)

    batch_design_conf = conf['batch_design']
    train_stat = {
        'id': run_name,
        'epoch': 0,
    }
    if CONFIG['train'].getboolean('initial_evaluation'):
        evaluate(conf, model, test_dataset, test_num_testcases, train_stat)
    step_counter = tf.train.get_or_create_global_step()
    step_counter.assign(0)

    drift_history = []
    embeddings_history = []
    metrics = []
    if CONFIG['train'].getboolean('compute_drift'):
        embeddings = compute_all_embeddings(model, conf, training_files)
        embeddings_history.append(embeddings)
    for epoch in range(conf['trainer']['num_epochs']):
        train_ds, num_examples = dataset.create_dataset(training_files, training_labels)
        train_ds = train_ds.batch(batch_design_conf['batch_size'], drop_remainder=True)
        batches = tqdm(train_ds,
                       total=math.ceil(num_examples / batch_design_conf['batch_size']),
                       desc='epoch #{}'.format(epoch + 1))
        losses = []
        batches_combined = 0
        grads = None
        for batch in batches:
            with tf.contrib.summary.record_summaries_every_n_global_steps(
                    CONFIG['tensorboard'].getint('record_every_n_global_steps'),
                    global_step=step_counter):
                with tf.GradientTape() as tape:
                    loss_value = model.loss(batch, model, dataset)
                    losses.append(float(loss_value))
                    batches.set_postfix({'loss': float(loss_value)})
                    tf.contrib.summary.scalar('loss', loss_value)

                if grads is None:
                    grads = tape.gradient(loss_value, model.variables)
                else:
                    for idx, g in enumerate(tape.gradient(loss_value, model.variables)):
                        if g is None:
                            continue
                        if grads[idx] is None:
                            grads[idx] = g
                        else:
                            grads[idx] += g
                batches_combined += 1
                if batches_combined == conf['trainer']['combine_batches']:
                    for optimizer_key, (_, variables) in model.learning_rates().items():
                        filtered_grads = filter(lambda x: x[1] in variables, zip(grads, model.variables))
                        optimizers[optimizer_key].apply_gradients(filtered_grads)
                    batches_combined = 0
                    grads = None
                    step_counter.assign_add(1)
                if CONFIG['train'].getboolean('compute_drift'):
                    # TODO: create a drift calculator class
                    embeddings_before = embeddings_history[-1]
                    embeddings_after = compute_all_embeddings(model, conf, training_files)
                    avg_drift = sum([
                        tf.reduce_sum(tf.norm(before - after, axis=1))
                        for before, after in zip(embeddings_before, embeddings_after)
                    ]) / len(training_files)
                    drift_history.append(float(avg_drift))
                    if len(drift_history) > CONFIG['train'].getint('drift_history_length'):
                        drift_history.pop(0)
                    tf.contrib.summary.scalar('avg_drift', avg_drift)
                    embeddings_history.append(embeddings_after)
                    if len(embeddings_history) > CONFIG['train'].getint('drift_history_length'):
                        initial_embeddings = embeddings_history.pop(0)
                        total_drift = sum(drift_history)
                        final_drift = sum([
                            tf.reduce_sum(tf.norm(before - after, axis=1))
                            for before, after in zip(embeddings_after, initial_embeddings)
                        ]) / len(training_files)
                        tf.contrib.summary.scalar('total_drift', total_drift)
                        tf.contrib.summary.scalar('drift_ratio', final_drift / total_drift)
                if CONFIG['tensorboard'].getboolean('s3_upload') and \
                        int(step_counter) % int(CONFIG['tensorboard']['s3_upload_period']) == 0:
                    upload_tensorboard_log_to_s3(run_name)
        print('epoch #{} checkpoint: {}'.format(epoch + 1, run_name))
        if CONFIG['tensorboard'].getboolean('enable_checkpoint'):
            create_checkpoint(checkpoint, run_name)
        train_stat['epoch'] = epoch + 1
        train_stat['loss'] = Decimal(str(sum(losses) / len(losses)))
        metrics.append(evaluate(conf, model, test_dataset, test_num_testcases, train_stat))
        if stopping_criteria(metrics):
            break
    if CONFIG['tensorboard'].getboolean('dynamodb_upload'):
        final_metrics = get_metric_to_report(metrics)
        table = db.Table('Experiment')
        for metric, score in final_metrics.items():
            table.update_item(
                Key={
                    'id': run_name,
                },
                UpdateExpression='SET #m = :u',
                ExpressionAttributeNames={
                    '#m': 'metric:{}'.format(metric),
                },
                ExpressionAttributeValues={
                    ':u': score,
                },
            )
