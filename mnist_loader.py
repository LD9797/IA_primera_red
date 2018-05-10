import cPickle
import gzip
import numpy as np


def load_data():
    """Carga los datos de mnist y devuelve una tuplca conteniendo los
    datos de entrenamiento, validacion y test"""
    mnist_file = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(mnist_file)
    mnist_file.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    """Pasa los pixels de las fotos a una lista vertical y las une junto con sus labels en una tupla y
       crea una lista que contiene esas tuplas.
       Devuelve una tupla conteniendo las listas de los datos de entrenamiento, validacion y test.
       Con los datos de entrenamiento, pone los labels en una lista vertical."""
    training_data, validation_data, test_data = load_data()
    training_inputs = [np.reshape(picture, (784, 1)) for picture in training_data[0]]
    training_labels = [vectorized_result(label) for label in training_data[1]]
    training_data_wrapped = zip(training_inputs, training_labels)
    validation_inputs = [np.reshape(picture, (784, 1)) for picture in validation_data[0]]
    validation_data_wrapped = zip(validation_inputs, validation_data[1])
    test_inputs = [np.reshape(picture, (784, 1)) for picture in test_data[0]]
    test_data_wrapped = zip(test_inputs, test_data[1])
    return training_data_wrapped, validation_data_wrapped, test_data_wrapped


def vectorized_result(j):
    """Crea un lista vertical y coloca un 1 en la posicion j"""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

