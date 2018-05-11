from network_refactored import *

import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

sizes = [784, 30, 10]
batch_params = [training_data, 10]
net = Network(sizes, type_cost_f=0, type_forward_f=0, type_learning_r=0, type_mini_b=0, params_mini_b=batch_params)

print "Por 0.3"
print "Batch 10"
print "1 hidden layer of 30"
print "Sigmoid"
net.run_network(training_data, 30, 3.0, test_data)
