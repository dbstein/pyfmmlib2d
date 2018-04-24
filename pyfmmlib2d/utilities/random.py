import numpy as np

def complex_random(sh):
    return np.random.rand(*sh) + 1j*np.random.rand(*sh)

def float_random(sh):
    return np.random.rand(*sh)
    