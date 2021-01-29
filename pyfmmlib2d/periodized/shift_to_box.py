import numpy as np

def shift_to_box(X, box):
    """
    Shift src or trg into central interval of periodic box
    X has shape [2, N]
    box = [xmin, xmax, ymin, ymax] of periodic box
    """
    XA = np.empty_like(X)
    XA[0, :] = shift_to_interval(X[0], [box[0], box[1]])
    XA[1, :] = shift_to_interval(X[1], [box[2], box[3]])
    return XA

def shift_to_interval(v, interval):
    """
    periodically shift values into interval
    """
    r = interval[1] - interval[0]
    return (v - interval[0]) % r + interval[0]
