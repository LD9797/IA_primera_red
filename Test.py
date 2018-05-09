from Network import *

training_data, \
    validation_data, \
    test_data = mnist_loader.load_data_wrapper()

sizes = [784, 30, 10]
net = Network([784, 30, 10])

net.sgd(training_data, 30, 10, 3.0, test_data=test_data)
