from enum import Enum


class DistanceFunction(Enum):
    EUCLIDEAN_DISTANCE = 1
    EUCLIDEAN_DISTANCE_SQUARED = 2
    DOT_PRODUCT = 3