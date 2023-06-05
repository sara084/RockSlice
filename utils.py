import numpy as np

def initial_weights(input_size):
    '''Helper function to create a weight variable initialized with
    a normal distribution
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    # W = np.random_normal((input_size, 6), mean=0.0, stddev=0.01)
    W = np.zeros((input_size, 6))
    b = np.array([[1., 0, 0], [0, 1., 0]], np.float32)

    return [W, b.flatten()]


def initial_weights_scale(input_size):
    '''Helper function to create a weight variable initialized with
    a normal distribution
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    # W = np.random_normal((input_size, 6), mean=0.0, stddev=0.01)
    W = np.zeros((input_size, 3))
    b = np.array([1, 0, 0], np.float32)
    # W = np.zeros((input_size, 1))
    # b = np.array([1], np.float32)

    return [W, b.flatten()]


def initial_weights_without_scale(input_size):
    '''Helper function to create a weight variable initialized with
    a normal distribution
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    W = np.zeros((input_size, 2))
    b = np.array([0, 0], np.float32)
    # W = np.zeros((input_size, 1))
    # b = np.array([1], np.float32)

    return W, b.flatten()