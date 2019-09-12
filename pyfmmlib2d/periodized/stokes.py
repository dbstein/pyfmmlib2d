import numpy as np
from pyfmmlib2d import SFMM


def stokes_kernel(sx, tx):
    ns = sx.shape[1]
    nt = tx.shape[1]
    dx = tx[0][:,None] - sx[0]
    dy = tx[1][:,None] - sx[1]
    d2 = dx**2 + dy**2
    d = np.sqrt(d2)
    logr = np.log(d)
    c = 0.25/np.pi
    G00 = (-logr + dx*dx/d2)
    G01 = dx*dy/d2
    G11 = (-logr + dy*dy/d2)
    u_mat = np.zeros([nt, 2*ns], dtype=float)
    v_mat = np.zeros([nt, 2*ns], dtype=float)
    u_mat[:, 0*ns:1*ns] = c*G00
    u_mat[:, 1*ns:2*ns] = c*G01
    v_mat[:, 0*ns:1*ns] = c*G01
    v_mat[:, 1*ns:2*ns] = c*G11
    return u_mat, v_mat
def stokes_kernel_stress(sx, tx):
    ns = sx.shape[1]
    nt = tx.shape[1]
    dx = tx[0][:,None] - sx[0]
    dy = tx[1][:,None] - sx[1]
    d2 = dx**2 + dy**2
    d4 = d2*d2
    # forces part, u_x
    c = 0.25/np.pi
    G00 = (dx*dy**2 - dx**3)/d4
    G01 = (dy**3 - dx**2*dy)/d4
    ux_mat = np.zeros([nt, 2*ns])
    ux_mat[:, 0*ns:1*ns] = c*G00
    ux_mat[:, 1*ns:2*ns] = c*G01
    # forces part, u_y
    G00 = (-3*dy*dx**2-dy**3)/d4
    G01 = (dx**3 - dx*dy**2)/d4
    uy_mat = np.zeros([nt, 2*ns])
    uy_mat[:, 0*ns:1*ns] = c*G00
    uy_mat[:, 1*ns:2*ns] = c*G01
    # forces part, v_x
    G01 = (dy**3 - dy*dx**2)/d4
    G11 = (-3*dx*dy**2-dx**3)/d4
    vx_mat = np.zeros([nt, 2*ns])
    vx_mat[:, 0*ns:1*ns] = c*G01
    vx_mat[:, 1*ns:2*ns] = c*G11
    # forces part, v_y
    G01 = (dx**3 - dy**2*dx)/d4
    G11 = (dy*dx**2 - dy**3)/d4
    vy_mat = np.zeros([nt, 2*ns])
    vy_mat[:, 0*ns:1*ns] = c*G01
    vy_mat[:, 1*ns:2*ns] = c*G11
    # forces part, p
    c = 0.5/np.pi
    G0 = dx/d2
    G1 = dy/d2
    p_mat = np.zeros([nt, 2*ns])
    p_mat[:, 0*ns:1*ns] = c*G0
    p_mat[:, 1*ns:2*ns] = c*G1
    return 2*ux_mat - p_mat, uy_mat + vx_mat, 2*vy_mat - p_mat

class periodized_stokes_fmm(object):
    def __init__(self, bounds, p=16, eps=1e-14):
        """
        Class to execute periodized Stokes FMM
        bounds: [xmin, xmax, ymin, ymax] (location of periodic box)
        p:      order of expansion to use on periodic walls
        """
        self.bounds = bounds
        self.p = p
        # compute the location of the collocation nodes
        nodes = 0.5*np.polynomial.chebyshev.chebgauss(p)[0][::-1] + 0.5
        ranx = bounds[1] - bounds[0]
        rany = bounds[3] - bounds[2]
        if np.abs(ranx - rany) > 1e-15:
            raise Exception('For now, periodization bounds must be a square.')
        self.width = ranx
        nodey = nodes*rany + bounds[2]
        rep = lambda x: np.repeat(x, p)
        self.node_left  = np.row_stack([ rep(bounds[0]), nodey ])
        self.node_right = np.row_stack([ rep(bounds[1]), nodey ])
        nodex = nodes*ranx + bounds[0]
        self.node_bottom = np.row_stack([ nodex, rep(bounds[2]) ])
        self.node_top    = np.row_stack([ nodex, rep(bounds[3]) ])
        self.check = np.column_stack([ self.node_left, self.node_right, \
                                       self.node_bottom, self.node_top ])
        # get normals (not outward facing!)
        self.normal_left   = np.row_stack([ rep(1.0), rep(0.0) ])
        self.normal_right  = np.row_stack([ rep(1.0), rep(0.0) ])
        self.normal_bottom = np.row_stack([ rep(0.0), rep(1.0) ])
        self.normal_top    = np.row_stack([ rep(0.0), rep(1.0) ])
        self.normals = np.column_stack([ self.normal_left, self.normal_right,
                                         self.normal_bottom, self.normal_top ])
        # generate sources
        self.n_check = 4*p
        self.n_sources = self.n_check
        self.center = [ 0.5*(self.bounds[0]+self.bounds[1]), 
                        0.5*(self.bounds[2]+self.bounds[3]) ]
        radius = 0.5*np.sqrt(2)*self.width
        adj = np.log(eps)/self.n_sources
        if adj < -0.5:
            raise Exception('Increase p (or decrease eps) to guarantee convergence.')
        Radius = radius/(1 + adj)
        theta = np.linspace(0, 2*np.pi, self.n_check, endpoint=False)
        self.source = np.row_stack([ self.center[0] + Radius*np.cos(theta),
                                     self.center[1] + Radius*np.sin(theta) ])
        # generate source --> targ Stokes velocity matrix
        S2U, S2V = stokes_kernel(self.source, self.check)
        # generate source --> targ Stokes stress matrix
        S2Sxx, S2Sxy, S2Syy = stokes_kernel_stress(self.source, self.check)
        S2SNx = S2Sxx*self.normals[0][:,None] + S2Sxy*self.normals[1][:,None]
        S2SNy = S2Sxy*self.normals[0][:,None] + S2Syy*self.normals[1][:,None]
        # generate the full system that we'll have to solve
        self.MAT = np.zeros([2*self.n_check, 2*self.n_sources], dtype=float)
        self.MAT[0*p:1*p] = S2U[1*p:2*p] - S2U[0*p:1*p]
        self.MAT[1*p:2*p] = S2U[3*p:4*p] - S2U[2*p:3*p]
        self.MAT[2*p:3*p] = S2V[1*p:2*p] - S2V[0*p:1*p]
        self.MAT[3*p:4*p] = S2V[3*p:4*p] - S2V[2*p:3*p]
        self.MAT[4*p:5*p] = S2SNx[1*p:2*p] - S2SNx[0*p:1*p]
        self.MAT[5*p:6*p] = S2SNx[3*p:4*p] - S2SNx[2*p:3*p]
        self.MAT[6*p:7*p] = S2SNy[1*p:2*p] - S2SNy[0*p:1*p]
        self.MAT[7*p:8*p] = S2SNy[3*p:4*p] - S2SNy[2*p:3*p]
        # take the SVD of this matrix
        self.U, D, self.VT = np.linalg.svd(self.MAT, full_matrices=False)
        D[D < eps] = np.Inf
        self.DI = 1.0/D
    def __call__(self, source, target, forces=None, dipstr=None, dipvec=None):
        # get total forces
        tfx = np.sum(forces[0])
        tfy = np.sum(forces[1])
        # get the first set of periodically tiled sources
        source_00 = source.copy()
        source_00[0] -= self.width
        source_00[1] -= self.width
        source_01 = source.copy()
        source_01[0] -= self.width
        source_02 = source.copy()
        source_02[0] -= self.width
        source_02[1] += self.width
        source_10 = source.copy()
        source_10[1] -= self.width
        source_12 = source.copy()
        source_12[1] += self.width
        source_20 = source.copy()
        source_20[0] += self.width
        source_20[1] -= self.width
        source_21 = source.copy()
        source_21[0] += self.width
        source_22 = source.copy()
        source_22[0] += self.width
        source_22[1] += self.width
        big_source = np.column_stack([
                source_00,
                source_01,
                source_02,
                source_10,
                source,
                source_12,
                source_20,
                source_21,
                source_22,
            ])
        # periodically tiled charges, dipstrs, dipvecs
        big_forces = None if forces is None else np.tile(forces, 9)
        big_dipstr = None if dipstr is None else np.tile(dipstr, 9)
        big_dipvec = None if dipvec is None else np.tile(dipvec, (1,9))
        big_target = np.column_stack([ self.check, target ])
        # compute FMM of tiled sources to source and targets
        out1 = SFMM(
                    source = big_source,
                    target = big_target,
                    forces = big_forces,
                    dipstr = big_dipstr,
                    dipvec = big_dipvec,
                    compute_source_velocity = True,
                    compute_source_stress   = True,
                    compute_target_velocity = True,
                    compute_target_stress   = True,
                )
        # extract u and du/dn on the cehck surfaces
        check_u   = out1['target']['u']  [:self.n_check]
        check_v   = out1['target']['v']  [:self.n_check]
        check_ux  = out1['target']['u_x'][:self.n_check]
        check_uy  = out1['target']['u_y'][:self.n_check]
        check_vx  = out1['target']['v_x'][:self.n_check]
        check_vy  = out1['target']['v_y'][:self.n_check]
        check_p   = out1['target']['p']  [:self.n_check]
        check_sxx = 2*check_ux - check_p
        check_sxy = check_uy + check_vx
        check_syy = 2*check_vy - check_p
        check_snx = check_sxx*self.normals[0] + check_sxy*self.normals[1]
        check_sny = check_sxy*self.normals[0] + check_syy*self.normals[1]
        p = self.p
        # get jumps across the domain
        area = self.width**2
        ujumpx = -(check_u[1*p:2*p] - check_u[0*p:1*p])
        ujumpy = -(check_u[3*p:4*p] - check_u[2*p:3*p])
        vjumpx = -(check_v[1*p:2*p] - check_v[0*p:1*p])
        vjumpy = -(check_v[3*p:4*p] - check_v[2*p:3*p])
        snxjumpx = -(check_snx[1*p:2*p] - check_snx[0*p:1*p] + tfx/self.width)
        snxjumpy = -(check_snx[3*p:4*p] - check_snx[2*p:3*p])
        snyjumpx = -(check_sny[1*p:2*p] - check_sny[0*p:1*p])
        snyjumpy = -(check_sny[3*p:4*p] - check_sny[2*p:3*p] + tfy/self.width)
        ujumps = np.concatenate([ujumpx, ujumpy, vjumpx, vjumpy, snxjumpx, snxjumpy, snyjumpx, snyjumpy])
        # solve for sources that set these jumps to 0
        tau = -self.VT.T.dot(self.U.T.dot(ujumps)*self.DI)
        # compute the periodic correction at the sources and targets
        big_target = np.column_stack([ source, target ])
        out2 = SFMM(
                    source = self.source,
                    target = big_target,
                    forces = tau.reshape(2, self.n_sources),
                    compute_target_velocity = True,
                    compute_target_stress   = True,
                )
        # now add the tiling FMM to the correction FMM
        SN = source.shape[1]
        source_dict = {}
        target_dict = {}
        for item in ['u', 'v', 'u_x', 'v_x', 'p']:
            source_dict[item] = out1['source'][item][4*SN:5*SN] + out2['target'][item][:SN]
            target_dict[item] = out1['target'][item][self.n_check:] + out2['target'][item][SN:]

        return { 'source' : source_dict, 'target' : target_dict }









