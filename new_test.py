from network_refactored import *
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

sizes = [784, 128, 10]
batch_params = [training_data, 32]
add = "C:\\Users\\User\\Desktop\\Resultados\\"
tipo = "128MSERELUSOFT"
file_names = [add+"Pesos" + tipo + ".pkl", add+"Biases" + tipo + ".pkl", add+"Progreso" + tipo + ".csv"]

net = Network(sizes, type_cost_f=0, type_forward_f=1, type_learning_r=1, params_mini_b=batch_params,
              lmbda=0.002, file_names=file_names)

print "Por 0.0085"
print "Batch 32"
print "1 hidden de 128"
print "Relu - MSE"
net.run_network(10, 0.0085, test_data)


#  Creo que no sirve con learning rates altos en cross

#  self.z -> Normal
#  self.z2 -> sigmoid(self.z)
#  self.z3 -> z2 * w2 -> normal
#  o -> sigmoid z3

#  o_delta -> (y - o) * sigmoidprima(o)
#  z2_error -> o_delta*w2T
