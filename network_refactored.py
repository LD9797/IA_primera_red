import random
import numpy as np
import functions


class Network(object):

    def __init__(self, sizes, type_cost_f, type_forward_f, type_learning_r, type_mini_b, params_mini_b):
        """type_cost_f = 0 -> MSE, 1 -> Cross entropy
           type_forward_f = 0 -> Sigmoid, 1 -> Relu, 2 -> Softmax
           type_learning_r = 0 -> learning_rate/len(mini_batch), 1 -> normal
           type_mini_batch = 0 -> fix length mini batch using all test_data
           params_mini_b = parameters for the type of mini batch"""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.type_cost_function = type_cost_f
        self.type_forward_function = type_forward_f
        self.type_learning_rate = type_learning_r
        self.type_mini_batch = type_mini_b
        self.params_mini_batch = params_mini_b

    def derivative_cost_function(self, x, y):
        if self.type_cost_function == 0:
            return functions.delta_mse(x, y)
        if self.type_cost_function == 1:
            return functions.delta_cross_entropy(x, y)

    def forward_function(self, z):
        if self.type_forward_function == 0:
            return functions.sigmoid(z)
        if self.type_forward_function == 1:
            return functions.re_lu(z)
        if self.type_forward_function == 2:
            return functions.softmax(z)

    def prime_forward_function(self, z):
        if self.type_forward_function == 0 or self.type_forward_function == 2:
            return functions.sigmoid_prime(z)
        if self.type_forward_function == 1:
            return functions.derivative_re_lu(z)

    def mini_batch_creator(self):
        if self.type_mini_batch == 0:
            return functions.regular_batches(self.params_mini_batch)

    def run_network(self, training_data, epochs, learning_rate, test_data):
        n_test = len(test_data)
        n = len(training_data)
        for epoc in xrange(epochs):
            mini_batches = self.mini_batch_creator()
            # Quitarle el mini_batch pasado al training data y el training data se va reduciendo
            # 1250 batches de 32 imagenes para probar 40 mil imagenes
            for mini_batch in mini_batches:
                self.run_network_on_mini_batch(mini_batch, learning_rate)
            print "Epoch {0}: {1} / {2}".format(epoc, self.evaluate(test_data), n_test)

    def run_network_on_mini_batch(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            no_function_results, activations = self.forward(x)
            delta_nabla_b, delta_nabla_w = self.backpropagation(no_function_results, activations, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  # SUMA TODO
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        if self.type_learning_rate == 0:
            learning_rate = learning_rate / len(mini_batch)
        self.weights = [w - learning_rate * nw for w, nw in zip(self.weights, nabla_w)]  # SGD
        self.biases = [b - learning_rate * nb for b, nb in zip(self.biases, nabla_b)]  # SGD

# Recomendado mini batches 32
# Epoches 6

    def forward(self, x):
        """X es la imagen"""
        no_function_results = []
        activations = [x]
        z = np.dot(self.weights[0], x) + self.biases[0]
        no_function_results.append(z)
        activation = self.forward_function(z)
        activations.append(activation)
        for b, w in zip(self.biases[1:], self.weights[1:]):
            z = np.dot(w, activation) + b
            no_function_results.append(z)
            activation = self.forward_function(z)
            activations.append(activation)
        return no_function_results, activations

    def backpropagation(self, no_function_results, activations, y):
        """X es la imagen, Y es el tag"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        primera_derivada = self.prime_forward_function(no_function_results[-1])
        costo = self.derivative_cost_function(activations[-1], y)
        delta = costo * primera_derivada
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for layer in xrange(2, self.num_layers):
            z = no_function_results[-layer]
            sp = self.prime_forward_function(z)
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        return nabla_b, nabla_w

    def feed_forward(self, a):
        """Return the output of the network if "a" is input."""
        for b, w in zip(self.biases, self.weights):
            a = self.forward_function(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
