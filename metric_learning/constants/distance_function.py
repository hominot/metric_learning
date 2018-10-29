from enum import Enum


class DistanceFunction(Enum):
    EUCLIDEAN_DISTANCE = 1
    EUCLIDEAN_DISTANCE_SQUARED = 2
    DOT_PRODUCT = 3


def distance_function(name):
    if name == 'dot_product':
        return DistanceFunction.DOT_PRODUCT
    if name == 'euclidean_distance':
        return DistanceFunction.EUCLIDEAN_DISTANCE
    if name == 'euclidean_distance_squared':
        return DistanceFunction.EUCLIDEAN_DISTANCE_SQUARED
    raise Exception('Unknown distance function with name {}'.format(name))
