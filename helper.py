import numpy as np
from numpy import ndarray

NUMBER_OF_EPOCH = 2000
BATCH_SIZE = 30


def assert_same_shape(array: ndarray,
                      array_grad: ndarray):
    assert array.shape == array_grad.shape, \
        f'''
        Two ndarrays should have the same shape;
        instead, first ndarray's shape is {array_grad.shape}
        and second ndarray's shape is {array.shape}.
        '''
    return None


def permute_data(X, y):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]

