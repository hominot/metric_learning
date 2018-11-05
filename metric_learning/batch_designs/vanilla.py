from util.registry.batch_design import BatchDesign


import tensorflow as tf


class VanillaBatchDesign(BatchDesign):
    name = 'vanilla'

    def create_dataset(self, image_files, labels, testing=False):
        data = list(zip(image_files, labels))
        return tf.data.Dataset.zip(
            self._create_datasets_from_elements(data, testing),
        ), len(data)

    def get_pairwise_distances(self, batch, model, distance_function):
        raise NotImplementedError

    def get_npair_distances(self, batch, model, n, distance_function):
        raise NotImplementedError

    def get_embeddings(self, batch, model, distance_function):
        raise NotImplementedError

    def get_next_batch(self, image_files, labels):
        raise NotImplementedError
