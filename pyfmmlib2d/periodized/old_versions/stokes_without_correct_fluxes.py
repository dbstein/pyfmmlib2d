import numpy as np
from pyfmmlib2d import SFMM

def stokes_kernel(sx, tx, w):
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
    u_mat[:, 0*ns:1*ns] = c*G00*w
    u_mat[:, 1*ns:2*ns] = c*G01*w
    v_mat[:, 0*ns:1*ns] = c*G01*w
    v_mat[:, 1*ns:2*ns] = c*G11*w
    return u_mat, v_mat
def stokes_kernel_stress(sx, tx, w):
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
    ux_mat[:, 0*ns:1*ns] = c*G00*w
    ux_mat[:, 1*ns:2*ns] = c*G01*w
    # forces part, u_y
    G00 = (-3*dy*dx**2-dy**3)/d4
    G01 = (dx**3 - dx*dy**2)/d4
    uy_mat = np.zeros([nt, 2*ns])
    uy_mat[:, 0*ns:1*ns] = c*G00*w
    uy_mat[:, 1*ns:2*ns] = c*G01*w
    # forces part, v_x
    G01 = (dy**3 - dy*dx**2)/d4
    G11 = (-3*dx*dy**2-dx**3)/d4
    vx_mat = np.zeros([nt, 2*ns])
    vx_mat[:, 0*ns:1*ns] = c*G01*w
    vx_mat[:, 1*ns:2*ns] = c*G11*w
    # forces part, v_y
    G01 = (dx**3 - dy**2*dx)/d4
    G11 = (dy*dx**2 - dy**3)/d4
    vy_mat = np.zeros([nt, 2*ns])
    vy_mat[:, 0*ns:1*ns] = c*G01*w
    vy_mat[:, 1*ns:2*ns] = c*G11*w
    # forces part, p
    c = 0.5/np.pi
    G0 = dx/d2
    G1 = dy/d2
    p_mat = np.zeros([nt, 2*ns])
    p_mat[:, 0*ns:1*ns] = c*G0*w
    p_mat[:, 1*ns:2*ns] = c*G1*w
    return 2*ux_mat - p_mat, uy_mat + vx_mat, 2*vy_mat - p_mat

# quad weights for the chebyshev nodes i'm using
def fejer_1(n):
    points = -np.cos(np.pi * (np.arange(n) + 0.5) / n)
    N = np.arange(1, n, 2)
    length = len(N)
    m = n - length
    K = np.arange(m)
    v0 = np.concatenate(
        [
            2 * np.exp(1j * np.pi * K / n) / (1 - 4 * K ** 2),
            np.zeros(length + 1),
        ]
    )
    v1 = v0[:-1] + np.conjugate(v0[:0:-1])
    w = np.fft.ifft(v1)
    weights = w.real
    return points, weights
def cheb(n, a=-1, b=1):
    p, w = fejer_1(n)
    ap = 0.5 * (p+1) * (b-a) + a
    return ap, 0.5*w*(b-a)

def generate_square(lx, ly, c, p, n):
    """
    Generate chebyshev discretization around square centered at c=(cx, cy)
    With width 2*lx and height 2*ly
    Discretization is chebyshev; with order p and number of panels n
    """
    ys = []
    yws = []
    xs = []
    xws = []
    wx = 2*lx/n
    wy = 2*ly/n
    for i in range(n):
        y, w = cheb(p, -ly + i*wy, -ly + (i+1)*wy)
        ys.append(y)
        yws.append(w)
        x, w = cheb(p, -lx + i*wx, -lx + (i+1)*wx)
        xs.append(x)
        xws.append(w)
    x = np.concatenate(xs)
    xw = np.concatenate(xws)
    y = np.concatenate(ys)
    yw = np.concatenate(yws)
    # stitch sides together to get nodes
    lxl = np.repeat(lx, p*n)
    lyl = np.repeat(ly, p*n)
    left   = np.row_stack([ -lxl + c[0],  y   + c[1] ])
    right  = np.row_stack([  lxl + c[0],  y   + c[1] ])
    bottom = np.row_stack([  x   + c[0], -lyl + c[1] ])
    top    = np.row_stack([  x   + c[0],  lyl + c[1] ])
    nodes = np.column_stack([ left, right, bottom, top ])
    # stitch together weights
    weights = np.concatenate([ yw, xw, yw, xw ])

    return nodes, weights, nodes.shape[1]

def generate_separated_square(lx, ly, c, p, n):
    """
    Generate chebyshev discretization around square centered at c=(cx, cy)
    With width 2*lx and height 2*ly
    Discretization is chebyshev; with order p and number of panels n
    """
    ys = []
    yws = []
    xs = []
    xws = []
    wx = 2*lx/n
    wy = 2*ly/n
    for i in range(n):
        y, w = cheb(p, -ly + i*wy, -ly + (i+1)*wy)
        ys.append(y)
        yws.append(w)
        x, w = cheb(p, -lx + i*wx, -lx + (i+1)*wx)
        xs.append(x)
        xws.append(w)
    x = np.concatenate(xs)
    xw = np.concatenate(xws)
    y = np.concatenate(ys)
    yw = np.concatenate(yws)
    # stitch sides together to get nodes
    lxl = np.repeat(lx, p*n)
    lyl = np.repeat(ly, p*n)
    left   = np.row_stack([ -lxl + c[0],  y   + c[1] ])
    right  = np.row_stack([  lxl + c[0],  y   + c[1] ])
    bottom = np.row_stack([  x   + c[0], -lyl + c[1] ])
    top    = np.row_stack([  x   + c[0],  lyl + c[1] ])

    return left, right, bottom, top

class periodized_stokes_fmm(object):
    def __init__(self, bounds=[0,2*np.pi,0,2*np.pi], p=16, N=4, expansion_factor=1.5, eps=1e-14):
        """
        Class to execute periodized Stokes FMM
        bounds: [xmin, xmax, ymin, ymax] (location of periodic box)
        p:      order of expansion to use on periodic walls
        """
        self.bounds = bounds
        self.p = p
        self.N = N
        self.ef = expansion_factor
        self.eps = eps
        # center of bounds
        self.center = [ 0.5*(self.bounds[0]+self.bounds[1]), 
                        0.5*(self.bounds[2]+self.bounds[3]) ]
        # ranges in both directions
        self.ranx = self.bounds[1] - self.bounds[0]
        self.rany = self.bounds[3] - self.bounds[2]
        if np.abs(self.ranx - self.rany) > 1e-15:
            raise Exception('For now, periodization bounds must be a square.')

        # get sources
        self.source, self.weights, self.n_sources = \
            generate_square(0.5*self.ranx*self.ef, 0.5*self.rany*self.ef, self.center, self.p, self.N)

        # get check surface
        self.check, self.check_weights, self.n_check = \
            generate_square(0.5*self.ranx, 0.5*self.rany, self.center, self.p, self.N)

        # get dumb check surface
        self.dumb_check, self.dumb_check_weights, _ = \
            generate_square(0.5*self.ranx*3, 0.5*self.rany*3, self.center, self.p, self.N)

        self.check_left, self.check_right, self.check_bottom, self.check_top = \
            generate_separated_square(0.5*self.ranx, 0.5*self.rany, self.center, self.p, self.N)

        # get normals at check surface (not outward facing!)
        rep = lambda x: np.repeat(x, self.p*self.N)
        self.normal_left   = np.row_stack([ rep(1.0), rep(0.0) ])
        self.normal_right  = np.row_stack([ rep(1.0), rep(0.0) ])
        self.normal_bottom = np.row_stack([ rep(0.0), rep(1.0) ])
        self.normal_top    = np.row_stack([ rep(0.0), rep(1.0) ])
        self.normals = np.column_stack([ self.normal_left, self.normal_right,
                                         self.normal_bottom, self.normal_top ])
        # generate source --> targ Stokes velocity matrix
        S2U, S2V = stokes_kernel(self.source, self.check, self.weights)
        # generate source --> targ Stokes stress matrix
        S2Sxx, S2Sxy, S2Syy = stokes_kernel_stress(self.source, self.check, self.weights)
        S2SNx = S2Sxx*self.normals[0][:,None] + S2Sxy*self.normals[1][:,None]
        S2SNy = S2Sxy*self.normals[0][:,None] + S2Syy*self.normals[1][:,None]
        # generate the full system that we'll have to solve
        self.MAT = np.zeros([2*self.n_check, 2*self.n_sources], dtype=float)
        # generate the full system that we'll have to solve
        p = self.p*self.N
        self.MAT[0*p:1*p] = S2U[1*p:2*p] - S2U[0*p:1*p]
        self.MAT[1*p:2*p] = S2U[3*p:4*p] - S2U[2*p:3*p]
        self.MAT[2*p:3*p] = S2V[1*p:2*p] - S2V[0*p:1*p]
        self.MAT[3*p:4*p] = S2V[3*p:4*p] - S2V[2*p:3*p]
        self.MAT[4*p:5*p] = S2SNx[1*p:2*p] - S2SNx[0*p:1*p]
        self.MAT[5*p:6*p] = S2SNx[3*p:4*p] - S2SNx[2*p:3*p]
        self.MAT[6*p:7*p] = S2SNy[1*p:2*p] - S2SNy[0*p:1*p]
        self.MAT[7*p:8*p] = S2SNy[3*p:4*p] - S2SNy[2*p:3*p]
        self.BIG_MAT = np.zeros([2*self.n_check+2, 2*self.n_sources+2], dtype=float)
        self.BIG_MAT[:-2,:-2] = self.MAT
        self.BIG_MAT[-2,0*self.n_sources:1*self.n_sources] = self.weights
        self.BIG_MAT[-1,1*self.n_sources:2*self.n_sources] = self.weights
        self.BIG_MAT[0*self.n_sources:1*self.n_sources,-2] = self.weights
        self.BIG_MAT[1*self.n_sources:2*self.n_sources,-1] = self.weights
        # take the SVD of this matrix
        self.U, D, self.VT = np.linalg.svd(self.BIG_MAT, full_matrices=False)
        D[D < eps] = np.Inf
        self.DI = 1.0/D
        # setup slices for easy evaluations to check surfaces
        NCL = self.check_left.shape[1]
        NCR = self.check_right.shape[1]
        NCB = self.check_bottom.shape[1]
        NCT = self.check_top.shape[1]
        N0 = 0
        N1 = N0 + NCL
        N2 = N1 + NCR
        N3 = N2 + NCB
        N4 = N3 + NCT
        self.slice_left =   slice(N0, N1)
        self.slice_right =  slice(N1, N2)
        self.slice_bottom = slice(N2, N3)
        self.slice_top =    slice(N3, N4)
        # allocate space for check evaluations
        self.check_left_u   = np.zeros(NCL)
        self.check_right_u  = np.zeros(NCR)
        self.check_bottom_u = np.zeros(NCB)
        self.check_top_u    = np.zeros(NCT)
        self.check_left_v   = np.zeros(NCL)
        self.check_right_v  = np.zeros(NCR)
        self.check_bottom_v = np.zeros(NCB)
        self.check_top_v    = np.zeros(NCT)
        self.check_left_snx   = np.zeros(NCL)
        self.check_right_snx  = np.zeros(NCR)
        self.check_bottom_snx = np.zeros(NCB)
        self.check_top_snx    = np.zeros(NCT)
        self.check_left_sny   = np.zeros(NCL)
        self.check_right_sny  = np.zeros(NCR)
        self.check_bottom_sny = np.zeros(NCB)
        self.check_top_sny    = np.zeros(NCT)
    def zero_check(self):
        self.check_left_u[:]     = 0.0
        self.check_right_u[:]    = 0.0
        self.check_bottom_u[:]   = 0.0
        self.check_top_u[:]      = 0.0
        self.check_left_v[:]     = 0.0
        self.check_right_v[:]    = 0.0
        self.check_bottom_v[:]   = 0.0
        self.check_top_v[:]      = 0.0
        self.check_left_snx[:]   = 0.0
        self.check_right_snx[:]  = 0.0
        self.check_bottom_snx[:] = 0.0
        self.check_top_snx[:]    = 0.0
        self.check_left_sny[:]   = 0.0
        self.check_right_sny[:]  = 0.0
        self.check_bottom_sny[:] = 0.0
        self.check_top_sny[:]    = 0.0
    def compute_to_check(self, src, fs, dps, dpv, left, right, bottom, top):
        if src.shape[1] > 0:
            out = SFMM(
                    source = src,
                    target = self.check,
                    forces = fs,
                    dipstr = dps,
                    dipvec = dpv,
                    compute_target_velocity = True,
                    compute_target_stress = True
                )
            u = out['target']['u']
            v = out['target']['v']
            ux = out['target']['u_x']
            uy = out['target']['u_y']
            vx = out['target']['v_x']
            vy = out['target']['v_y']
            p  = out['target']['p']
            sxx = 2*ux - p
            sxy = uy + vx
            syy = 2*vy - p
            if left:
                self.check_left_u += u[self.slice_left]
                self.check_left_v += v[self.slice_left]
                self.check_left_snx += sxx[self.slice_left]*self.normal_left[0]
                self.check_left_sny += sxy[self.slice_left]*self.normal_left[0]
            if right:
                self.check_right_u += u[self.slice_right]
                self.check_right_v += v[self.slice_right]
                self.check_right_snx += sxx[self.slice_right]*self.normal_right[0]
                self.check_right_sny += sxy[self.slice_right]*self.normal_right[0]
            if bottom:
                self.check_bottom_u += u[self.slice_bottom]
                self.check_bottom_v += v[self.slice_bottom]
                self.check_bottom_snx += sxy[self.slice_bottom]*self.normal_bottom[1]
                self.check_bottom_sny += syy[self.slice_bottom]*self.normal_bottom[1]
            if top:
                self.check_top_u += u[self.slice_top]
                self.check_top_v += v[self.slice_top]
                self.check_top_snx += sxy[self.slice_top]*self.normal_top[1]
                self.check_top_sny += syy[self.slice_top]*self.normal_top[1]
    def __call__(self, source, reflection_distance=0.15, forces=None, dipstr=None, dipvec=None, target=None, compute_source_velocity=False, compute_source_stress=False, compute_target_velocity=False, compute_target_stress=False):
        # get total forces
        tfx = np.sum(forces[0])
        tfy = np.sum(forces[1])
        # compute real distances
        dist_x = reflection_distance * self.ranx
        dist_y = reflection_distance * self.rany
        if compute_target_velocity or compute_target_stress:
            assert target is not None, 'Need to give target to compute target velocity or stresses'
        if target is None:
            target = np.zeros([2,1])
        # get where points are too close to boundary
        close_left = source[0] < self.bounds[0] + dist_x
        close_right = source[0] > self.bounds[1] - dist_x
        close_bottom = source[1] < self.bounds[2] + dist_y
        close_top = source[1] > self.bounds[3] - dist_y
        # get all bad locations
        bad_locations = np.logical_or.reduce([close_left, close_right, close_bottom, close_top])
        good_locations = ~bad_locations
        # further divide the close to boundary points
        far_bottom_top = ~np.logical_or(close_bottom, close_top)
        far_left_right = ~np.logical_or(close_left, close_right)
        close_left_bottom  = np.logical_and(close_left,   close_bottom)
        close_left_top     = np.logical_and(close_left,   close_top)
        close_left_only    = np.logical_and(close_left,   far_bottom_top)
        close_right_bottom = np.logical_and(close_right,  close_bottom)
        close_right_top    = np.logical_and(close_right,  close_top)
        close_right_only   = np.logical_and(close_right,  far_bottom_top)
        close_bottom_only  = np.logical_and(close_bottom, far_left_right)
        close_top_only     = np.logical_and(close_top,    far_left_right)
        # reflection 1: points close to only the left boundary reflected to the right
        left_only_bad_source = source[:,close_left_only]
        left_only_reflected_source = np.row_stack([
                left_only_bad_source[0] + self.ranx,
                left_only_bad_source[1],
            ])
        left_only_reflected_forces = forces[:,close_left_only] if forces is not None else None
        left_only_reflected_dipstr = dipstr[:,close_left_only] if dipstr is not None else None
        left_only_reflected_dipvec = dipvec[:,close_left_only] if dipvec is not None else None
        # reflection 2: points close to only the right boundary reflected to the left
        right_only_bad_source = source[:,close_right_only]
        right_only_reflected_source = np.row_stack([
                right_only_bad_source[0] - self.ranx,
                right_only_bad_source[1],
            ])
        right_only_reflected_forces = forces[:,close_right_only] if forces is not None else None
        right_only_reflected_dipstr = dipstr[:,close_right_only] if dipstr is not None else None
        right_only_reflected_dipvec = dipvec[:,close_right_only] if dipvec is not None else None
        # reflection 3: points close to only the bottom boundary reflected to the top
        bottom_only_bad_source = source[:,close_bottom_only]
        bottom_only_reflected_source = np.row_stack([
                bottom_only_bad_source[0],
                bottom_only_bad_source[1] + self.rany,
            ])
        bottom_only_reflected_forces = forces[:,close_bottom_only] if forces is not None else None
        bottom_only_reflected_dipstr = dipstr[:,close_bottom_only] if dipstr is not None else None
        bottom_only_reflected_dipvec = dipvec[:,close_bottom_only] if dipvec is not None else None
        # reflection 4: points close to only the top boundary reflected to  the bottom
        top_only_bad_source = source[:,close_top_only]
        top_only_reflected_source = np.row_stack([
                top_only_bad_source[0],
                top_only_bad_source[1] - self.rany,
            ])
        top_only_reflected_forces = forces[:,close_top_only] if forces is not None else None
        top_only_reflected_dipstr = dipstr[:,close_top_only] if dipstr is not None else None
        top_only_reflected_dipvec = dipvec[:,close_top_only] if dipvec is not None else None
        # reflection 5: points close to the left and the bottom boundaries
        left_bottom_bad_source = source[:,close_left_bottom]
        left_bottom_rb = np.row_stack([
                left_bottom_bad_source[0] + self.ranx,
                left_bottom_bad_source[1],
            ])
        left_bottom_rt = np.row_stack([
                left_bottom_bad_source[0] + self.ranx,
                left_bottom_bad_source[1] + self.rany,
            ])
        left_bottom_lt = np.row_stack([
                left_bottom_bad_source[0],
                left_bottom_bad_source[1] + self.rany,
            ])
        left_bottom_reflected_source = np.column_stack([left_bottom_rb, left_bottom_rt, left_bottom_lt])
        left_bottom_reflected_forces = forces[:,close_left_bottom] if forces is not None else None
        left_bottom_reflected_dipstr = dipstr[:,close_left_bottom] if dipstr is not None else None
        left_bottom_reflected_dipvec = dipvec[:,close_left_bottom] if dipvec is not None else None
        # reflection 6: points close to the left and the top boundaries, and their reflections
        left_top_bad_source = source[:,close_left_top]
        left_top_rt = np.row_stack([
                left_top_bad_source[0] + self.ranx,
                left_top_bad_source[1],
            ])
        left_top_rb = np.row_stack([
                left_top_bad_source[0] + self.ranx,
                left_top_bad_source[1] - self.rany,
            ])
        left_top_lb = np.row_stack([
                left_top_bad_source[0],
                left_top_bad_source[1] - self.rany,
            ])
        left_top_reflected_source = np.column_stack([left_top_rt, left_top_rb, left_top_lb])
        left_top_reflected_forces = forces[:,close_left_top] if forces is not None else None
        left_top_reflected_dipstr = dipstr[:,close_left_top] if dipstr is not None else None
        left_top_reflected_dipvec = dipvec[:,close_left_top] if dipvec is not None else None
        # reflection 7: points close to the right and the top boundaries
        right_top_bad_source = source[:,close_right_top]
        right_top_lt = np.row_stack([
                right_top_bad_source[0] - self.ranx,
                right_top_bad_source[1],
            ])
        right_top_lb = np.row_stack([
                right_top_bad_source[0] - self.ranx,
                right_top_bad_source[1] - self.rany,
            ])
        right_top_rb = np.row_stack([
                right_top_bad_source[0],
                right_top_bad_source[1] - self.rany,
            ])
        right_top_reflected_source = np.column_stack([right_top_lt, right_top_lb, right_top_rb])
        right_top_reflected_forces = forces[:,close_right_top] if forces is not None else None
        right_top_reflected_dipstr = dipstr[:,close_right_top] if dipstr is not None else None
        right_top_reflected_dipvec = dipvec[:,close_right_top] if dipvec is not None else None
        # reflection 8: points close to the right and the bottom boundaries
        right_bottom_bad_source = source[:,close_right_bottom]
        right_bottom_lb = np.row_stack([
                right_bottom_bad_source[0] - self.ranx,
                right_bottom_bad_source[1],
            ])
        right_bottom_lt = np.row_stack([
                right_bottom_bad_source[0] - self.ranx,
                right_bottom_bad_source[1] + self.rany,
            ])
        right_bottom_rt = np.row_stack([
                right_bottom_bad_source[0],
                right_bottom_bad_source[1] + self.rany,
            ])
        right_bottom_reflected_source = np.column_stack([right_bottom_lb, right_bottom_lt, right_bottom_rt])
        right_bottom_reflected_forces = forces[:,close_right_bottom] if forces is not None else None
        right_bottom_reflected_dipstr = dipstr[:,close_right_bottom] if dipstr is not None else None
        right_bottom_reflected_dipvec = dipvec[:,close_right_bottom] if dipvec is not None else None
        # add these into the "all left only" grouping
        all_left_only_source = np.column_stack([ left_only_bad_source,       right_only_reflected_source ])
        all_left_only_forces = np.column_stack([ left_only_reflected_forces, right_only_reflected_forces ]) if forces is not None else None
        all_left_only_dipstr = np.column_stack([ left_only_reflected_dipstr, right_only_reflected_dipstr ]) if dipstr is not None else None
        all_left_only_dipvec = np.column_stack([ left_only_reflected_dipvec, right_only_reflected_dipvec ]) if dipvec is not None else None
        # add these into the "all right only" grouping
        all_right_only_source = np.column_stack([ right_only_bad_source,       left_only_reflected_source ])
        all_right_only_forces = np.column_stack([ right_only_reflected_forces, left_only_reflected_forces ]) if forces is not None else None
        all_right_only_dipstr = np.column_stack([ right_only_reflected_dipstr, left_only_reflected_dipstr ]) if dipstr is not None else None
        all_right_only_dipvec = np.column_stack([ right_only_reflected_dipvec, left_only_reflected_dipvec ]) if dipvec is not None else None
        # add these into the "all bottom only" grouping
        all_bottom_only_source = np.column_stack([ bottom_only_bad_source,       top_only_reflected_source ])
        all_bottom_only_forces = np.column_stack([ bottom_only_reflected_forces, top_only_reflected_forces ]) if forces is not None else None
        all_bottom_only_dipstr = np.column_stack([ bottom_only_reflected_dipstr, top_only_reflected_dipstr ]) if dipstr is not None else None
        all_bottom_only_dipvec = np.column_stack([ bottom_only_reflected_dipvec, top_only_reflected_dipvec ]) if dipvec is not None else None
        # add these into the "all top only" grouping
        all_top_only_source = np.column_stack([ top_only_bad_source,       bottom_only_reflected_source ])
        all_top_only_forces = np.column_stack([ top_only_reflected_forces, bottom_only_reflected_forces ]) if forces is not None else None
        all_top_only_dipstr = np.column_stack([ top_only_reflected_dipstr, bottom_only_reflected_dipstr ]) if dipstr is not None else None
        all_top_only_dipvec = np.column_stack([ top_only_reflected_dipvec, bottom_only_reflected_dipvec ]) if dipvec is not None else None
        # add these into the "left bottom" grouping
        all_left_bottom_source = np.column_stack([ left_bottom_bad_source,       left_top_lb,               right_top_lb,               right_bottom_lb,              ])
        all_left_bottom_forces = np.column_stack([ left_bottom_reflected_forces, left_top_reflected_forces, right_top_reflected_forces, right_bottom_reflected_forces ]) if forces is not None else None
        all_left_bottom_dipstr = np.column_stack([ left_bottom_reflected_dipstr, left_top_reflected_dipstr, right_top_reflected_dipstr, right_bottom_reflected_dipstr ]) if dipstr is not None else None
        all_left_bottom_dipvec = np.column_stack([ left_bottom_reflected_dipvec, left_top_reflected_dipvec, right_top_reflected_dipvec, right_bottom_reflected_dipvec ]) if dipvec is not None else None
        # add these into the "right bottom" grouping
        all_right_bottom_source = np.column_stack([ right_bottom_bad_source,       right_top_rb,               left_top_rb,               left_bottom_rb,              ])
        all_right_bottom_forces = np.column_stack([ right_bottom_reflected_forces, right_top_reflected_forces, left_top_reflected_forces, left_bottom_reflected_forces ]) if forces is not None else None
        all_right_bottom_dipstr = np.column_stack([ right_bottom_reflected_dipstr, right_top_reflected_dipstr, left_top_reflected_dipstr, left_bottom_reflected_dipstr ]) if dipstr is not None else None
        all_right_bottom_dipvec = np.column_stack([ right_bottom_reflected_dipvec, right_top_reflected_dipvec, left_top_reflected_dipvec, left_bottom_reflected_dipvec ]) if dipvec is not None else None
        # add these into the "right top" grouping
        all_right_top_source = np.column_stack([ right_top_bad_source,       left_top_rt,               left_bottom_rt,               right_bottom_rt,              ])
        all_right_top_forces = np.column_stack([ right_top_reflected_forces, left_top_reflected_forces, left_bottom_reflected_forces, right_bottom_reflected_forces ]) if forces is not None else None
        all_right_top_dipstr = np.column_stack([ right_top_reflected_dipstr, left_top_reflected_dipstr, left_bottom_reflected_dipstr, right_bottom_reflected_dipstr ]) if dipstr is not None else None
        all_right_top_dipvec = np.column_stack([ right_top_reflected_dipvec, left_top_reflected_dipvec, left_bottom_reflected_dipvec, right_bottom_reflected_dipvec ]) if dipvec is not None else None
        # add these into the "left top" grouping
        all_left_top_source = np.column_stack([ left_top_bad_source,       left_bottom_lt,               right_bottom_lt,               right_top_lt,              ])
        all_left_top_forces = np.column_stack([ left_top_reflected_forces, left_bottom_reflected_forces, right_bottom_reflected_forces, right_top_reflected_forces ]) if forces is not None else None
        all_left_top_dipstr = np.column_stack([ left_top_reflected_dipstr, left_bottom_reflected_dipstr, right_bottom_reflected_dipstr, right_top_reflected_dipstr ]) if dipstr is not None else None
        all_left_top_dipvec = np.column_stack([ left_top_reflected_dipvec, left_bottom_reflected_dipvec, right_bottom_reflected_dipvec, right_top_reflected_dipvec ]) if dipvec is not None else None
        # sources / forces / dipstrs / dipvecs for direct portion of FMM
        lbr_sr = left_bottom_reflected_source
        rbr_sr = right_bottom_reflected_source
        rtr_sr = right_top_reflected_source
        ltr_sr = left_top_reflected_source
        lbr_ch = np.tile(left_bottom_reflected_forces,  (1,3)) if forces is not None else None
        rbr_ch = np.tile(right_bottom_reflected_forces, (1,3)) if forces is not None else None
        rtr_ch = np.tile(right_top_reflected_forces,    (1,3)) if forces is not None else None
        ltr_ch = np.tile(left_top_reflected_forces,     (1,3)) if forces is not None else None
        lbr_ds = np.tile(left_bottom_reflected_dipstr,  (1,3)) if dipstr is not None else None
        rbr_ds = np.tile(right_bottom_reflected_dipstr, (1,3)) if dipstr is not None else None
        rtr_ds = np.tile(right_top_reflected_dipstr,    (1,3)) if dipstr is not None else None
        ltr_ds = np.tile(left_top_reflected_dipstr,     (1,3)) if dipstr is not None else None
        lbr_dv = np.tile(left_bottom_reflected_dipvec,  (1,3)) if dipvec is not None else None
        rbr_dv = np.tile(right_bottom_reflected_dipvec, (1,3)) if dipvec is not None else None
        rtr_dv = np.tile(right_top_reflected_dipvec,    (1,3)) if dipvec is not None else None
        ltr_dv = np.tile(left_top_reflected_dipvec,     (1,3)) if dipvec is not None else None
        all_source = np.column_stack([ source, left_only_reflected_source, right_only_reflected_source, bottom_only_reflected_source, top_only_reflected_source, lbr_sr, rbr_sr, rtr_sr, ltr_sr ])
        all_forces = np.column_stack([ forces, left_only_reflected_forces, right_only_reflected_forces, bottom_only_reflected_forces, top_only_reflected_forces, lbr_ch, rbr_ch, rtr_ch, ltr_ch ]) if forces is not None else None
        all_dipstr = np.column_stack([ dipstr, left_only_reflected_dipstr, right_only_reflected_dipstr, bottom_only_reflected_dipstr, top_only_reflected_dipstr, lbr_ds, rbr_ds, rtr_ds, ltr_ds ]) if dipstr is not None else None
        all_dipvec = np.column_stack([ dipvec, left_only_reflected_dipvec, right_only_reflected_dipvec, bottom_only_reflected_dipvec, top_only_reflected_dipvec, lbr_dv, rbr_dv, rtr_dv, ltr_dv ]) if dipvec is not None else None
        # set check evaluations to 0
        self.zero_check()
        # now compute things to appropriate portions of the check surfaces
        # first for the left-only sources
        self.compute_to_check(
                all_left_only_source,
                all_left_only_forces,
                all_left_only_dipstr,
                all_left_only_dipvec,
                left=False, right=True, bottom=True, top=True,
            )
        # now for the right-only sources
        self.compute_to_check(
                all_right_only_source,
                all_right_only_forces,
                all_right_only_dipstr,
                all_right_only_dipvec,
                left=True, right=False, bottom=True, top=True,
            )
        # now for the bottom-only sources
        self.compute_to_check(
                all_bottom_only_source,
                all_bottom_only_forces,
                all_bottom_only_dipstr,
                all_bottom_only_dipvec,
                left=True, right=True, bottom=False, top=True,
            )
        # now for the top-only sources
        self.compute_to_check(
                all_top_only_source,
                all_top_only_forces,
                all_top_only_dipstr,
                all_top_only_dipvec,
                left=True, right=True, bottom=True, top=False,
            )
        # now for the left-bottom sources
        self.compute_to_check(
                all_left_bottom_source,
                all_left_bottom_forces,
                all_left_bottom_dipstr,
                all_left_bottom_dipvec,
                left=False, right=True, bottom=False, top=True,
            )
        # now for the right-bottom sources
        self.compute_to_check(
                all_right_bottom_source,
                all_right_bottom_forces,
                all_right_bottom_dipstr,
                all_right_bottom_dipvec,
                left=True, right=False, bottom=False, top=True,
            )
        # now for the right-top sources
        self.compute_to_check(
                all_right_top_source,
                all_right_top_forces,
                all_right_top_dipstr,
                all_right_top_dipvec,
                left=True, right=False, bottom=True, top=False,
            )
        # now for the left-top sources
        self.compute_to_check(
                all_left_top_source,
                all_left_top_forces,
                all_left_top_dipstr,
                all_left_top_dipvec,
                left=False, right=True, bottom=True, top=False,
            )
        # now for the good sources
        good_source = source[:,good_locations]
        good_forces = forces[:,good_locations] if forces is not None else None
        good_dipstr = dipstr[:,good_locations] if dipstr is not None else None
        good_dipvec = dipvec[:,good_locations] if dipvec is not None else None
        self.compute_to_check(
                good_source,
                good_forces,
                good_dipstr,
                good_dipvec,
                left=True, right=True, bottom=True, top=True
            )
        # get jumps across the check surface
        area = self.ranx*self.rany
        ujumpx = (self.check_right_u - self.check_left_u)
        ujumpy = (self.check_top_u   - self.check_bottom_u)
        vjumpx = (self.check_right_v - self.check_left_v)
        vjumpy = (self.check_top_v   - self.check_bottom_v)
        snxjumpx = (self.check_right_snx - self.check_left_snx + tfx/self.ranx)
        snxjumpy = (self.check_top_snx   - self.check_bottom_snx)
        snyjumpx = (self.check_right_sny - self.check_left_sny)
        snyjumpy = (self.check_top_sny   - self.check_bottom_sny + tfy/self.rany)
        ujumps = np.concatenate([ujumpx, ujumpy, vjumpx, vjumpy, snxjumpx, snxjumpy, snyjumpx, snyjumpy])
        # get sum of tiled forces
        force_sum = np.sum(all_forces, axis=1)
        rhs = np.concatenate([-ujumps, -force_sum])
        tau = self.VT.T.dot(self.U.T.dot(rhs)*self.DI)[:-2]
        # weight by quadrature weights
        tau = tau.reshape(2, self.n_sources)*self.weights
        # compute FMM of tiled sources to source and targets
        out1 = SFMM(
                    source = all_source,
                    target = target,
                    forces = all_forces,
                    dipstr = all_dipstr,
                    dipvec = all_dipvec,
                    compute_source_velocity = compute_source_velocity,
                    compute_source_stress   = compute_source_stress,
                    compute_target_velocity = compute_target_velocity,
                    compute_target_stress   = compute_target_stress,
                )
        # compute correction to set jumps to 0
        big_target = np.column_stack([ source, target ])
        out2 = SFMM(
                    source = self.source,
                    target = big_target,
                    forces = tau,
                    compute_target_velocity = compute_source_velocity or compute_target_velocity,
                    compute_target_stress   = compute_source_stress or compute_target_stress,
                )
        # good flux calculation, done painfully
        def sum_check(u, sl):
            w = self.check_weights[sl]
            return np.sum(u*w)/np.sum(w)
        lefties =  all_source[0] <  self.center[0]
        righties = all_source[0] >= self.center[0]
        check_1 = self.check[:,self.slice_right].copy()
        check_1[0] -= 0.25*self.ranx
        out3a = SFMM(
                    source = all_source[:,lefties],
                    target = check_1,
                    forces = all_forces[:,lefties] if forces is not None else None,
                    dipstr = all_dipstr[:,lefties] if dipstr is not None else None,
                    dipvec = all_dipvec[:,lefties] if dipvec is not None else None,
                    compute_target_velocity = True,
                )
        out3b = SFMM(
                    source = all_source[:,righties],
                    target = check_1,
                    forces = all_forces[:,righties] if forces is not None else None,
                    dipstr = all_dipstr[:,righties] if dipstr is not None else None,
                    dipvec = all_dipvec[:,righties] if dipvec is not None else None,
                    compute_target_velocity = True,
                )
        out4 = SFMM(
                    source = self.source,
                    target = check_1,
                    forces = tau,
                    compute_target_velocity = True,
                )
        lefty_ua = sum_check(out3a['target']['u'], self.slice_right)
        lefty_ub = sum_check(out3b['target']['u'], self.slice_right)
        lefty_uc = sum_check(out4['target']['u'],  self.slice_right)
        lefty_u = out3a['target']['u'] + out3b['target']['u'] + out4['target']['u']
        lefty_flux = sum_check(lefty_u, self.slice_right)
        print(lefty_ua, lefty_ub, lefty_ua+lefty_ub, lefty_uc, lefty_ua+lefty_ub+lefty_uc)

        check_2 = self.check[:,self.slice_left].copy()
        check_2[0] += 0.25*self.ranx
        out3a = SFMM(
                    source = all_source[:,lefties],
                    target = check_2,
                    forces = all_forces[:,lefties] if forces is not None else None,
                    dipstr = all_dipstr[:,lefties] if dipstr is not None else None,
                    dipvec = all_dipvec[:,lefties] if dipvec is not None else None,
                    compute_target_velocity = True,
                )
        out3b = SFMM(
                    source = all_source[:,righties],
                    target = check_2,
                    forces = all_forces[:,righties] if forces is not None else None,
                    dipstr = all_dipstr[:,righties] if dipstr is not None else None,
                    dipvec = all_dipvec[:,righties] if dipvec is not None else None,
                    compute_target_velocity = True,
                )
        out4 = SFMM(
                    source = self.source,
                    target = check_2,
                    forces = tau,
                    compute_target_velocity = True,
                )
        righty_ua = sum_check(out3a['target']['u'], self.slice_left)
        righty_ub = sum_check(out3b['target']['u'], self.slice_left)
        righty_uc = sum_check(out4['target']['u'],  self.slice_left)
        righty_u = out3a['target']['u'] + out3b['target']['u'] + out4['target']['u']
        righty_flux = sum_check(righty_u, self.slice_left)
        print(righty_ua, righty_ub, righty_ua+righty_ub, righty_uc, righty_ua+righty_ub+righty_uc)


        # righties = all_source[0] >= self.center[0]
        # check_2 = self.check[:,self.slice_right]
        # # check_2[0] += 0.25*self.ranx
        # out3 = SFMM(
        #             source = all_source[:,lefties],
        #             target = check_2,
        #             forces = all_forces[:,lefties] if forces is not None else None,
        #             dipstr = all_dipstr[:,lefties] if dipstr is not None else None,
        #             dipvec = all_dipvec[:,lefties] if dipvec is not None else None,
        #             compute_target_velocity = True,
        #         )
        # out4 = SFMM(
        #             source = self.source,
        #             target = check_2,
        #             forces = tau,
        #             compute_target_velocity = True,
        #         )
        # righty_u = out3['target']['u']# + out4['target']['u']
        # righty_flux = sum_check(righty_u, self.slice_right)


        # out4 = SFMM(
        #             source = all_source[:,lefties],
        #             target = self.dumb_check,
        #             forces = all_forces[:,lefties] if forces is not None else None,
        #             dipstr = all_dipstr[:,lefties] if dipstr is not None else None,
        #             dipvec = all_dipvec[:,lefties] if dipvec is not None else None,
        #             compute_target_velocity = True,
        #         )
        # out5 = SFMM(
        #             source = all_source[:,righties],
        #             target = self.check,
        #             forces = all_forces[:,righties] if forces is not None else None,
        #             dipstr = all_dipstr[:,righties] if dipstr is not None else None,
        #             dipvec = all_dipvec[:,righties] if dipvec is not None else None,
        #             compute_target_velocity = True,
        #         )
        # out6 = SFMM(
        #             source = all_source[:,righties],
        #             target = self.dumb_check,
        #             forces = all_forces[:,righties] if forces is not None else None,
        #             dipstr = all_dipstr[:,righties] if dipstr is not None else None,
        #             dipvec = all_dipvec[:,righties] if dipvec is not None else None,
        #             compute_target_velocity = True,
        #         )
        # out7 = SFMM(
        #             source = self.source,
        #             target = self.check,
        #             forces = tau,
        #             compute_target_velocity = True,
        #         )
        # def sum_check(u, sl):
        #     w = self.check_weights[sl]
        #     return np.sum(u[sl]*w)/np.sum(w)
        # def sum_dumb_check(u, sl):
        #     w = self.dumb_check_weights[sl]
        #     return np.sum(u[sl]*w)/np.sum(w)
        # # compute the flux from the lefties on the right check wall
        # uc = out3['target']['u']
        # u1 = sum_check(uc, self.slice_right)

        # uc = out5['target']['u']
        # u2 = sum_check(uc, self.slice_left)
        # print('righties --> left', u2)

        # uc = out7['target']['u']
        # u3 = sum_check(uc, self.slice_right)
        # print('source --> right', u3)
        # u4 = sum_check(uc, self.slice_left)
        # print('source --> left', u4)

        # udc = out4['target']['u']
        # u_left = sum_check(uc, self.slice_right) - sum_dumb_check(udc, self.slice_left) + sum_dumb_check(udc, self.slice_right)
        # # compute the flux from the righties on the left check wall
        # uc = out5['target']['u']
        # udc = out6['target']['u']
        # u_right = sum_check(uc, self.slice_left) + sum_dumb_check(udc, self.slice_left) - sum_dumb_check(udc, self.slice_right)
        # # get the total flux
        # u_source = sum_check(out7['target']['u'], self.slice_right)
        # v_source = sum_check(out7['target']['v'], self.slice_top)
        # u_flux = u_left + u_right + u_source
        # print(u_left, u_right, u_source, u_flux)

        # u_flux = out3['target']['u'][self.slice_right] + out4['target']['u'][self.slice_right]
        # v_flux = out3['target']['v'][self.slice_top] + out4['target']['v'][self.slice_top]
        # u_weight = self.check_weights[self.slice_right]
        # v_weight = self.check_weights[self.slice_top]
        # u_flux_sum = -np.sum(u_flux * u_weight) / np.sum(u_weight)
        # v_flux_sum = -np.sum(v_flux * v_weight) / np.sum(v_weight)
        # print(v_flux_sum)

        all_lefty_flux = lefty_flux
        all_righty_fux = righty_flux

        print('mix', lefty_ua+righty_ub, lefty_ub+righty_ua)

        mix_flux1 = lefty_ua + righty_ub + (lefty_uc + righty_uc)/2
        mix_flux2 = lefty_ua + righty_ub + righty_uc
        print('mix2', mix_flux1, mix_flux2)

        u_flux = lefty_flux# + righty_flux
        print(lefty_flux, righty_flux)#, righty_flux, u_flux)

        # now add the tiling FMM to the correction FMM
        SN = source.shape[1]
        out = {}
        if compute_source_velocity or compute_source_stress:
            out_self = {}
            if compute_source_velocity:
                out_self['u'] = out1['source']['u'][:SN] + out2['target']['u'][:SN] - u_flux
                out_self['v'] = out1['source']['v'][:SN] + out2['target']['v'][:SN]# - v_source
            if compute_source_stress:
                out_self['u_x'] = out1['source']['u_x'][:SN] + out2['target']['u_x'][:SN]
                out_self['u_y'] = out1['source']['u_y'][:SN] + out2['target']['u_y'][:SN]
                out_self['v_x'] = out1['source']['v_x'][:SN] + out2['target']['v_x'][:SN]
                out_self['v_y'] = out1['source']['v_y'][:SN] + out2['target']['v_y'][:SN]
                out_self['p']   = out1['source']['p'][:SN]   + out2['target']['p'][:SN]
            out['self'] = out_self
        if compute_target_velocity or compute_target_stress:
            out_target = {}
            if compute_target_velocity:
                out_target['u'] = out1['target']['u'] + out2['target']['u'][SN:]# - u_source
                out_target['v'] = out1['target']['v'] + out2['target']['v'][SN:]# - v_source
            if compute_target_stress:
                out_target['u_x'] = out1['target']['u_x'] + out2['target']['u_x'][SN:]
                out_target['u_y'] = out1['target']['u_y'] + out2['target']['u_y'][SN:]
                out_target['v_x'] = out1['target']['v_x'] + out2['target']['v_x'][SN:]
                out_target['v_y'] = out1['target']['v_y'] + out2['target']['v_y'][SN:]
                out_target['p']   = out1['target']['p']   + out2['target']['p'][SN:]
            out['target'] = out_target
        return out



        # out3 = SFMM(
        #             source = self.source,
        #             target = self.check,
        #             forces = tau.reshape(2, self.n_sources),
        #             compute_target_velocity = True,
        #         )
        # # now add the tiling FMM to the correction FMM
        # SN = source.shape[1]
        # source_dict = {}
        # target_dict = {}
        # ww = self.weights
        # for item in ['u', 'v', 'u_x', 'v_x', 'p']:
        #     if item == 'u':
        #         uu = check_u[1*p:2*p] + out3['target']['u'][1*p:2*p]
        #         adder = -np.sum(uu*ww)/self.width
        #         stopx = adder
        #         # stopx = np.sum(check_u[1*p:2*p]*ww)/self.width
        #         # stopy = np.sum(out3['target']['u'][1*p:2*p]*ww)/self.width
        #     elif item == 'v':
        #         vv = check_v[3*p:4*p] + out3['target']['v'][3*p:4*p]
        #         adder = -np.sum(vv*ww)/self.width
        #         stopy = adder
        #     else:
        #         adder = 0.0
        #     source_dict[item] = out1['source'][item][4*SN:5*SN] + out2['target'][item][:SN] + adder
        #     target_dict[item] = out1['target'][item][self.n_check:] + out2['target'][item][SN:] + adder

        # return { 'source' : source_dict, 'target' : target_dict }, stopx, stopy









