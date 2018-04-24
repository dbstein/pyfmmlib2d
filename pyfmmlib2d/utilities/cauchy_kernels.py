import numpy as np

# cauchy potential, gradient, and hessian kernels
# these are simply coded to ensure correctness; they are not fast
# and are used for testing the FMM implementations

def cauchy_kernel(sx, tx, charge, dipstr):
    # same self-interaction as FMM
    is_self = sx is tx
    csx = sx[0] + 1j*sx[1]
    ctx = tx[0] + 1j*tx[1]
    # get difference
    diff = ctx[:,None] - csx
    # charge part
    if is_self:
        np.fill_diagonal(diff, 1.0)
    G = np.log(diff)
    # dipstr part
    if is_self:
        np.fill_diagonal(diff, np.inf)
    Gd = 1.0/diff
    return G.dot(charge) + Gd.dot(dipstr)
def cauchy_kernel_d1(sx, tx, charge, dipstr):
    # no self interaction checking
    csx = sx[0] + 1j*sx[1]
    ctx = tx[0] + 1j*tx[1]
    # get difference
    diff = ctx[:,None] - csx
    G = 1.0/diff
    Gd = -1.0/diff**2
    return G.dot(charge) + Gd.dot(dipstr)
def cauchy_kernel_d2(sx, tx, charge, dipstr):
    # no self interaction checking
    csx = sx[0] + 1j*sx[1]
    ctx = tx[0] + 1j*tx[1]
    # get difference
    diff = ctx[:,None] - csx
    G = -1.0/diff**2
    Gd = 2.0/diff**3
    return G.dot(charge) + Gd.dot(dipstr)
