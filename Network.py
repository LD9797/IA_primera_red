import random
import numpy as np
from test_np import convert_np_np


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            #a = sigmoid(np.dot(w, a) + b)
            a = ReLU(np.dot(w, a) + b)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)  #  mini batches de 10 imagenes
            mini_batches = [training_data[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            # Quitarle el mini_batch pasado al training data y el training data se va reduciendo
            # 1250 batches de 32 imagenes para probar 40 mil imagenes
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            test = zip(nabla_b, delta_nabla_b)
            testa = zip(nabla_w, delta_nabla_w)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  # SUMA TODO
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

# Recomendado mini batches 32
# Epoches 6


    #  X es la imagen
    #  Y es el tag
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):  # agarra los biases de la 1 hidden y de la ultima layer despues
            z = np.dot(w, activation) + b
            zs.append(z)
            # activation = sigmoid(z)
            activation = ReLU(z)
            activations.append(activation)
        primera_derivada = dReLU(zs[-1])
        # primera_derivada = sigmoid_prime(zs[-1])
        costo = delta_cross_entropy(activations[-1], y)
        delta = costo * primera_derivada
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            # sp = sigmoid_prime(z)
            sp = dReLU(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def ReLU(x):
    return x * (x > 0)


def dReLU(x):
    return 1. * (x > 0)


def softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


def cross_entropy(X, y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    """
    m = y.shape[0]
    p = softmax(X)
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m
    return loss


def delta_cross_entropy(X, y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    """
    m = y.shape[0]
    grad = softmax(X)
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


#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#net = Network([784, 30, 10])
#net.sgd(training_data, 30, 10, 3.0, test_data=test_data)




