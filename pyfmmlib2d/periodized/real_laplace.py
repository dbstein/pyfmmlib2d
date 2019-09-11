import numpy as np
from pyfmmlib2d import RFMM

def laplace_kernel(sx, tx):
    dx = tx[0][:,None] - sx[0]
    dy = tx[1][:,None] - sx[1]
    d = np.hypot(dx, dy)
    return np.log(d)
def laplace_kernel_gradient(sx, tx):
    dx = tx[0][:,None] - sx[0]
    dy = tx[1][:,None] - sx[1]
    d2 = dx**2 + dy**2
    return dx/d2, dy/d2

class periodized_laplace_fmm(object):
    def __init__(self, bounds, p=16, eps=1e-14):
        """
        Class to execute periodized FMM
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
        # generate source --> targ Laplace pot matrix
        S2U = laplace_kernel(self.source, self.check)
        # generate source --> targ Laplace grad matrix
        S2Gx, S2Gy = laplace_kernel_gradient(self.source, self.check)
        S2N = S2Gx*self.normals[0][:,None] + S2Gy*self.normals[1][:,None]
        # generate the full system that we'll have to solve
        self.MAT = np.zeros([self.n_sources, self.n_sources], dtype=float)
        self.MAT[0*p:1*p] = S2U[1*p:2*p] - S2U[0*p:1*p]
        self.MAT[1*p:2*p] = S2U[3*p:4*p] - S2U[2*p:3*p]
        self.MAT[2*p:3*p] = S2N[1*p:2*p] - S2N[0*p:1*p]
        self.MAT[3*p:4*p] = S2N[3*p:4*p] - S2N[2*p:3*p]
        # take the SVD of this matrix
        self.U, D, self.VT = np.linalg.svd(self.MAT, full_matrices=False)
        D[D < eps] = np.Inf
        self.DI = 1.0/D
    def __call__(self, source, target, charge=None, dipstr=None, dipvec=None):
        # project charge onto solvability space
        charge = charge - np.mean(charge)
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
        big_charge = None if charge is None else np.tile(charge, 9)
        big_dipstr = None if dipstr is None else np.tile(dipstr, 9)
        big_dipvec = None if dipvec is None else np.tile(dipvec, (1,9))
        big_target = np.column_stack([ self.check, target ])
        # compute FMM of tiled sources to source and targets
        out1 = RFMM(
                    source = big_source,
                    target = big_target,
                    charge = big_charge,
                    dipstr = big_dipstr,
                    dipvec = big_dipvec,
                    compute_source_potential = True,
                    compute_source_gradient  = True,
                    compute_target_potential = True,
                    compute_target_gradient  = True,
                )
        # extract u and du/dn on the cehck surfaces
        check_u   = out1['target']['u'][:self.n_check]
        check_dux = out1['target']['u_x'][:self.n_check]
        check_duy = out1['target']['u_y'][:self.n_check]
        check_dun = check_dux*self.normals[0] + check_duy*self.normals[1]
        p = self.p
        # get jumps across the domain
        ujumpx = check_u[1*p:2*p] - check_u[0*p:1*p]
        ujumpy = check_u[3*p:4*p] - check_u[2*p:3*p]
        unjumpx = check_dun[1*p:2*p] - check_dun[0*p:1*p]
        unjumpy = check_dun[3*p:4*p] - check_dun[2*p:3*p]
        ujumps = np.concatenate([ujumpx, ujumpy, unjumpx, unjumpy])
        # solve for sources that set these jumps to 0
        tau = -self.VT.T.dot(self.U.T.dot(ujumps)*self.DI)
        # compute the periodic correction at the sources and targets
        out3 = RFMM(
                    source = self.source,
                    target = self.check,
                    charge = tau,
                    compute_target_potential = True,
                    compute_target_gradient  = True,
                )
        uu = out3['target']['u']
        uux = out3['target']['u_x']
        uuy = out3['target']['u_y']
        uun = uux*self.normals[0] + uuy*self.normals[1]
        ujx = uu[1*p:2*p] - uu[0*p:1*p]
        ujy = uu[3*p:4*p] - uu[2*p:3*p]
        unjx = uun[1*p:2*p] - uun[0*p:1*p]
        unjy = uun[3*p:4*p] - uun[2*p:3*p]
        big_target = np.column_stack([ source, target ])
        out2 = RFMM(
                    source = self.source,
                    target = big_target,
                    charge = tau,
                    compute_target_potential = True,
                    compute_target_gradient  = True,
                )
        # now add the tiling FMM to the correction FMM
        SN = source.shape[1]
        self_u  = out1['source']['u'] [4*SN:5*SN] + out2['target']['u'] [:SN]
        self_Du = out1['source']['Du'][:,4*SN:5*SN] + out2['target']['Du'][:,:SN]
        target_u  = out1['target']['u'] [self.n_check:] + out2['target']['u'] [SN:]
        target_Du = out1['target']['Du'][:,self.n_check:] + out2['target']['Du'][:,SN:]
        out = {
            'self' : {
                'u'   : self_u,
                'Du'  : self_Du,
                'u_x' : self_Du[0],
                'u_y' : self_Du[1],
            },
            'target' : {
                'u'   : target_u,
                'Du'  : target_Du,
                'u_x' : target_Du[0],
                'u_y' : target_Du[1],
            },
        }
        return out









