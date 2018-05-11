import numpy as np


def convert_np_np(converting_array):
    a = []
    for x in range(len(converting_array)):
        a.append(np.asarray([converting_array[x]]))
    return np.asarray(a)

