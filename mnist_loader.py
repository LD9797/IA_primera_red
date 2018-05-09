import cPickle
import gzip
import numpy as np


def load_data():
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]  # Pasa las fotos a un vector vertial
    training_results = [vectorized_result(y) for y in tr_d[1]]  # Agarra los labels y crea un vector vertial con un 1 en
                                                                # la posicion a la que pertence el label
    training_data = zip(training_inputs, training_results)  # Junta el vector vertical con el vector de su label en una tupla
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return training_data, validation_data, test_data


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

