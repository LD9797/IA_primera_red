import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def re_lu(x):
    return x * (x > 0)


def derivative_re_lu(x):
    return 1. * (x > 0)


def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)


def cross_entropy(x, y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    """
    m = y.shape[0]
    p = softmax(x)
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m
    return loss


def delta_cross_entropy(x, y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    """
    m = y.shape[0]
    grad = softmax(x)
    grad[np.argmax(y)] -= 1
    # grad[range(m), range(y.shape[1])] -= 1
    # grad[range(m), y] -= 1
    grad = grad/m
    return grad


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


def cost_derivative(output_activations, y):
    return 2 * (output_activations - y)

