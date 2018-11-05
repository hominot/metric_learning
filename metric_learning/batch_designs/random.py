from util.registry.batch_design import BatchDesign

import random


class RandomBatchDesign(BatchDesign):
    name = 'random'

    def get_next_batch(self, image_files, labels):
        data = list(zip(image_files, labels, range(len(image_files))))
        random.shuffle(data)
        return data[:self.conf['batch_design']['batch_size']]

    def get_pairwise_distances(self, batch, model, distance_function):
        raise NotImplementedError

    def get_npair_distances(self, batch, model, n, distance_function):
        raise NotImplementedError

    def get_embeddings(self, batch, model, distance_function):
        raise NotImplementedError
