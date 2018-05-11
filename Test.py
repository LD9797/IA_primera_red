from Network import *
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

sizes = [784, 256, 128, 10]
net = Network(sizes)

print "Por 0.085"
print "Batch 32"
print "2 capas de 256, 128"
print "Relu - Softmax - Cross entropy"
net.sgd(training_data, 30, 32, 2.72, test_data=test_data)

