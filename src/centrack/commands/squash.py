import numpy


def squash(data: numpy.ndarray):
    return data.max(axis=1)
