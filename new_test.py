from network_refactored import *
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

sizes = [784, 30, 10]
batch_params = [training_data, 10]
net = Network(sizes, type_cost_f=1, type_forward_f=1, type_learning_r=1, type_mini_b=0, params_mini_b=batch_params)

print "Por 0.3"
print "Batch 10"
print "1 Hidden de 30"
print "Sigmoid"
net.run_network(30, 0.3, test_data)


#  Creo que no sirve con learning rates altos en cross

#  self.z -> Normal
#  self.z2 -> sigmoid(self.z)
#  self.z3 -> z2 * w2 -> normal
#  o -> sigmoid z3

#  o_delta -> y - o * sigmoidprima(o)
#  z2_error -> o_delta*w2T
