import tensorflow as tf

from util.registry.class_registry import ClassRegistry


class BatchDesign(object, metaclass=ClassRegistry):
    module_path = 'metric_learning.batch_designs'

    def __init__(self, conf, extra_info):
        super(BatchDesign, self).__init__()
        self.conf = conf
        self.extra_info = extra_info
        self.data_loader = extra_info['data_loader']

    def _create_datasets_from_elements(self, elements, testing=False):
        image_files, labels = zip(*elements)
        images_ds = tf.data.Dataset.from_tensor_slices(
            tf.constant(image_files)
        ).map(self.data_loader.image_parse_function)
        if 'random_flip' in self.conf['image'] and self.conf['image']['random_flip']:
            images_ds = images_ds.map(self.data_loader.random_flip)
        if 'random_crop' in self.conf['image']:
            if testing:
                images_ds = images_ds.map(self.data_loader.center_crop)
            else:
                images_ds = images_ds.map(self.data_loader.random_crop)
        labels_ds = tf.data.Dataset.from_tensor_slices(tf.constant(labels, dtype=tf.int64))
        return images_ds, labels_ds

    def get_next_batch(self, image_files, labels):
        raise NotImplementedError

    def create_dataset(self, image_files, labels, testing=False):
        data = []
        for _ in range(self.conf['batch_design']['num_batches'] * self.conf['batch_design']['combine_batches']):
            elements = self.get_next_batch(
                image_files, labels)
            data += elements

        return tf.data.Dataset.zip(
            self._create_datasets_from_elements(data, testing),
        ), len(data)

    def get_pairwise_distances(self, batch, model, distance_function):
        raise NotImplementedError

    def get_npair_distances(self, batch, model, n, distance_function):
        raise NotImplementedError

    def get_embeddings(self, batch, model, distance_function):
        raise NotImplementedError

    def __str__(self):
        return self.conf['batch_design']['dataset']['name']
