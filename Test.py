from Network import *
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

sizes = [784, 32, 10]
net = Network(sizes)

print "Por 0.085"
print "Batch 32"
print "1 capa de 32"
print "Relu - Softmax - Cross entropy"
net.sgd(training_data, 30, 32, 2.72, test_data=test_data)

