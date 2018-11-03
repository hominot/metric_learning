import tensorflow as tf

import boto3
import json

from util.train import train

conn_sqs = boto3.resource('sqs')

if __name__ == '__main__':
    tf.enable_eager_execution()
    queue = conn_sqs.get_queue_by_name(QueueName='experiment-configs')
    while True:
        messages = queue.receive_messages(
            MaxNumberOfMessages=1,
            WaitTimeSeconds=0)

        for message in messages:
            conf = json.loads(message.body)
            message.delete()
            train(conf)
