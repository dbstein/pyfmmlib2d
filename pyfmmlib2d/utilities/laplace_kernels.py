import numpy as np

# laplace potential, gradient, and hessian kernels
# these are simply coded to ensure correctness; they are not fast
# and are used for testing the FMM implementations

def laplace_kernel(sx, tx, charge, dipstr, dipvec):
    # same self-interaction as FMM
    is_self = sx is tx
    # get distance
    dx = tx[0][:,None] - sx[0]
    dy = tx[1][:,None] - sx[1]
    d2 = dx**2 + dy**2
    if is_self:
        np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    # charge part
    G = np.log(d)
    # dipole part
    Gd = -(dipvec[0]*dx+dipvec[1]*dy)/d2
    if is_self:
        np.fill_diagonal(Gd, 0.0)
    return G.dot(charge) + Gd.dot(dipstr)
def laplace_kernel_gradient(sx, tx, charge, dipstr, dipvec):
    # no self interaction checking
    dx = tx[0][:,None] - sx[0]
    dy = tx[1][:,None] - sx[1]
    d2 = dx**2 + dy**2
    d4 = d2**2
    # charge part
    Gx = dx/d2
    Gy = dy/d2
    # dipole part
    Gdx = (dipvec[0]*(dx**2 - dy**2) + dipvec[1]*2*dx*dy)/d4
    Gdy = (dipvec[0]*2*dx*dy + dipvec[1]*(dy**2 - dx**2))/d4
    # add up
    rx = Gx.dot(charge) + Gdx.dot(dipstr)
    ry = Gy.dot(charge) + Gdy.dot(dipstr)
    return rx, ry
def laplace_kernel_hessian(sx, tx, charge, dipstr, dipvec):
    # no self interaction checking
    dx = tx[0][:,None] - sx[0]
    dy = tx[1][:,None] - sx[1]
    d2 = dx**2 + dy**2
    d4 = d2**2
    d6 = d2**3
    # charge part
    Gxx = (dy**2 - dx**2)/d4
    Gxy = -2*dx*dy/d4
    Gyy = (dx**2 - dy**2)/d4
    # dipole part
    Gdxx1 = -4*dx*(dipvec[0]*(dx**2 - dy**2) + dipvec[1]*2*dx*dy)/d6
    Gdxx2 = (2*dipvec[0]*dx + 2*dipvec[1]*dy)/d4
    Gdxx = Gdxx1 + Gdxx2
    Gdxy1 = -4*dy*(dipvec[0]*(dx**2 - dy**2) + dipvec[1]*2*dx*dy)/d6
    Gdxy2 = (-2*dipvec[0]*dy + 2*dipvec[1]*dx)/d4
    Gdxy = Gdxy1 + Gdxy2
    Gdyy1 = -4*dy*(dipvec[0]*2*dx*dy + dipvec[1]*(dy**2 - dx**2))/d6
    Gdyy2 = (2*dipvec[0]*dx + 2*dipvec[1]*dy)/d4
    Gdyy = Gdyy1 + Gdyy2
    # add up
    rxx = Gxx.dot(charge) + Gdxx.dot(dipstr)
    rxy = Gxy.dot(charge) + Gdxy.dot(dipstr)
    ryy = Gyy.dot(charge) + Gdyy.dot(dipstr)
    return rxx, rxy, ryy
