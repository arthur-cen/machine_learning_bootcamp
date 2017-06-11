import numpy as np

def sigmoid(x):
    """
    Sigmoid (logistic) function
    :param x: array-like shape(n_sample, n_feature)
    :return: simgoid value (array like)
    """

    # TODO sigmoid function
    return 1. / (1. + np.exp(x))


def dsigmoid(x):
    """
    Derivative of sigmoid function
    :param x: array-like shape(n_sample, n_feature)
    :return: derivative value (array like)
    """
    # TODO dsigmoid function
    return sigmoid(x).dot((1. - sigmoid(x)))  # TODO: check if dimension matches


def tanh(x):
    """
    Tanh function
    :param x: array-like shape(n_sample, n_feature)
    :return: tanh value (array like)
    """
    # TODO tanh function
    return np.tanh(x)


def dtanh(x):
    """
    Derivative of tanh function
    :param x: array-like shape(n_sample, n_feature)
    :return: derivative value (array like)
    """
    # TODO dtanh function
    return 1. - tanh(x) ** 2


def softmax(X):
    """
    softmax function
    :param X:
    :return:
    """
    # TODO softmax function
    expo = np.exp(X)
    return (expo.T / np.sum(expo, axis=1)).T