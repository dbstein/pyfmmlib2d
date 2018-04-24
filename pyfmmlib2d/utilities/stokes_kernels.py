import numpy as np

# laplace velocity and stress kernels
# these are simply coded to ensure correctness; they are not fast
# and are used for testing the FMM implementations

def stokes_kernel(sx, tx, forces, dipstr, dipvec):
    # same self-interaction as FMM
    is_self = sx is tx
    # get distance
    dx = tx[0][:,None] - sx[0]
    dy = tx[1][:,None] - sx[1]
    d2 = dx**2 + dy**2
    d4 = d2*d2
    d = np.sqrt(d2)
    if is_self:
        np.fill_diagonal(d, 1.0)
        np.fill_diagonal(d2, np.inf)
        np.fill_diagonal(d4, np.inf)
    logr = np.log(d)
    # forces part
    c = 0.25/np.pi
    G00 = (-logr + dx*dx/d2)
    G01 = dx*dy/d2
    G11 = (-logr + dy*dy/d2)
    u1 = c*(G00.dot(forces[0]) + G01.dot(forces[1]))
    v1 = c*(G01.dot(forces[0]) + G11.dot(forces[1]))
    # dipole part
    c = 1.0/np.pi
    r_dot_n = dx*dipvec[0] + dy*dipvec[1]
    r_dot_n_ir4 = r_dot_n/d4
    Gd00 = r_dot_n_ir4*dx*dx
    Gd01 = r_dot_n_ir4*dx*dy
    Gd11 = r_dot_n_ir4*dy*dy
    u2 = c*(Gd00.dot(dipstr[0]) + Gd01.dot(dipstr[1]))
    v2 = c*(Gd01.dot(dipstr[0]) + Gd11.dot(dipstr[1]))
    return u1+u2, v1+v2
def stokes_kernel_stress(sx, tx, forces, dipstr, dipvec):
    # no self interaction checking
    dx = tx[0][:,None] - sx[0]
    dy = tx[1][:,None] - sx[1]
    d2 = dx**2 + dy**2
    d4 = d2*d2
    d6 = d2*d4
    # forces part, u_x
    c = 0.25/np.pi
    G00 = (dx*dy**2 - dx**3)/d4
    G01 = (dy**3 - dx**2*dy)/d4
    u_x1 = c*(G00.dot(forces[0]) + G01.dot(forces[1]))
    # forces part, u_y
    G00 = (-3*dy*dx**2-dy**3)/d4
    G01 = (dx**3 - dx*dy**2)/d4
    u_y1 = c*(G00.dot(forces[0]) + G01.dot(forces[1]))
    # forces part, v_x
    G01 = (dy**3 - dy*dx**2)/d4
    G11 = (-3*dx*dy**2-dx**3)/d4
    v_x1 = c*(G01.dot(forces[0]) + G11.dot(forces[1]))
    # forces part, v_y
    G01 = (dx**3 - dy**2*dx)/d4
    G11 = (dy*dx**2 - dy**3)/d4
    v_y1 = c*(G01.dot(forces[0]) + G11.dot(forces[1]))
    # forces part, p
    c = 0.5/np.pi
    G0 = dx/d2
    G1 = dy/d2
    p1 = c*(G0.dot(forces[0]) + G1.dot(forces[1]))
    # dipole part, u_x
    c = 1.0/np.pi
    r_dot_n = dx*dipvec[0] + dy*dipvec[1]
    Gd00 = (3*dx**2*dipvec[0] + 2*dx*dy*dipvec[1])/d4 - \
                4*dx**3*r_dot_n/d6
    Gd01 = (2*dx*dy*dipvec[0] + dy**2*dipvec[1])/d4 - \
                4*dx**2*dy*r_dot_n/d6
    u_x2 = c*(Gd00.dot(dipstr[0]) + Gd01.dot(dipstr[1]))
    # dipole part, u_y
    Gd00 = dx**2*dipvec[1]/d4 - 4*dx**2*dy*r_dot_n/d6
    Gd01 = (dx**2*dipvec[0] + 2*dx*dy*dipvec[1])/d4 - \
                4*dx*dy**2*r_dot_n/d6
    u_y2 = c*(Gd00.dot(dipstr[0]) + Gd01.dot(dipstr[1]))
    # dipole part, v_x
    Gd01 = (dy**2*dipvec[1] + 2*dy*dx*dipvec[0])/d4 - \
                4*dy*dx**2*r_dot_n/d6
    Gd11 = dy**2*dipvec[0]/d4 - 4*dy**2*dx*r_dot_n/d6
    v_x2 = c*(Gd01.dot(dipstr[0]) + Gd11.dot(dipstr[1]))
    # dipole part, v_y
    Gd01 = (2*dy*dx*dipvec[1] + dx**2*dipvec[0])/d4 - \
                4*dy**2*dx*r_dot_n/d6
    Gd11 = (3*dy**2*dipvec[1] + 2*dy*dx*dipvec[0])/d4 - \
                4*dy**3*r_dot_n/d6
    v_y2 = c*(Gd01.dot(dipstr[0]) + Gd11.dot(dipstr[1]))
    # dipole part, p
    Gd0 = -dipvec[0]/d2 + 2*r_dot_n*dx/d4
    Gd1 = -dipvec[1]/d2 + 2*r_dot_n*dy/d4
    p2 = c*(Gd0.dot(dipstr[0]) + Gd1.dot(dipstr[1]))
    return u_x1+u_x2, u_y1+u_y2, v_x1+v_x2, v_y1+v_y2, p1+p2


